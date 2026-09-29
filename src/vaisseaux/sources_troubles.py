"""Réunit plusieurs jeux publics de fonds d'œil pour le réseau des troubles, avec un test externe.

Entraînement (étiquettes partielles : -1 = inconnu pour ce jeu, ignoré par la perte) :
- ODIR-5K : les 6 troubles (mots-clés par œil, voir `troubles.construire_jeu`) ;
- RFMiD (3 caméras, Inde) : rétinopathie diabétique (DR), excavation papillaire (ODC, signe de
  glaucome), DMLA (ARMD), myopie (MYA ; fond tigré seul = incertain), rétinopathie hypertensive
  (HR) ; le trouble des milieux (MH) rend la cataracte incertaine ;
- SMDG-19 : glaucome, 19 jeux réunis ; on retire ceux déjà présents ailleurs (ODIR, JSIEC,
  sjchoi86) et ceux qui n'ont qu'une classe (un jeu 100 % glaucome apprendrait la caméra) ;
- sjchoi86 « retina_dataset » : normal, cataracte, glaucome.

Test externe, jamais utilisé pour entraîner ni régler : JSIEC-1000 (39 catégories, un autre
hôpital). Doublons : l'empreinte perceptuelle (phash) ne fait que proposer des paires, car tous
les fonds d'œil se ressemblent en gros (disque orangé, papille) ; une paire n'est retenue que si le
dessin fin des vaisseaux (canal vert passe-haut, dans le champ de vue) est corrélé à plus de 0,5,
alors que deux yeux différents ne dépassent pas 0,45. Les images d'entraînement en double d'une
image JSIEC sont retirées ; les doublons entre images d'entraînement partagent un groupe pour la
validation croisée.

Chaque image passe par `preparer_fond_oeil`, la même chaîne que dans la démo, et est mise en
cache en JPEG 448 × 448.

    python -m vaisseaux.sources_troubles --sortie data/troubles
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from vaisseaux.donnees import preparer_fond_oeil
from vaisseaux.troubles import CLES, construire_jeu

KAGGLE = Path.home() / ".cache" / "kagglehub" / "datasets"
ODIR = KAGGLE / "andrewmvd/ocular-disease-recognition-odir5k/versions/2"
RFMID = KAGGLE / "andrewmvd/retinal-disease-classification/versions/1"
JSIEC = KAGGLE / "linchundan/fundusimage1000/versions/4/1000images"
SJCHOI = KAGGLE / "jr2ngb/cataractdataset/versions/2/dataset"
SMDG = "bumbledeep/smdg-full-dataset"
COTE_CACHE = 448
DISTANCE_CANDIDATS = 8  # bits différents sur 64 (empreinte perceptuelle) : paires à vérifier
CORRELATION_DOUBLON = 0.5  # corrélation des détails vasculaires au-delà de laquelle c'est le même œil
COTE_DETAILS = 128
INCONNU = dict.fromkeys(CLES, -1)


# ------------------------------------------------------------------ étiquettes par jeu


def _odir() -> list[dict]:
    jeu = construire_jeu(ODIR)
    return [
        {"source": "odir", "groupe": f"odir-{r.patient}", "origine": str(ODIR / "preprocessed_images" / r.filename)}
        | {c: int(getattr(r, c)) for c in CLES}
        for r in jeu.itertuples()
    ]


def _rfmid() -> list[dict]:
    lignes = []
    for partie, csv, dossier in (
        ("Training_Set", "RFMiD_Training_Labels.csv", "Training"),
        ("Evaluation_Set", "RFMiD_Validation_Labels.csv", "Validation"),
        ("Test_Set", "RFMiD_Testing_Labels.csv", "Test"),
    ):
        base = RFMID / partie / partie
        for r in pd.read_csv(base / csv).itertuples():
            lignes.append(
                {
                    "source": "rfmid",
                    "groupe": f"rfmid-{dossier}-{r.ID}",
                    "origine": str(base / dossier / f"{r.ID}.png"),
                    "diabete": int(r.DR),
                    "glaucome": int(r.ODC),
                    "cataracte": -1 if r.MH else 0,
                    "dmla": int(r.ARMD),
                    "hypertension": int(r.HR),
                    "myopie": 1 if r.MYA else (-1 if r.TSLN else 0),
                }
            )
    return lignes


def _sjchoi() -> list[dict]:
    regles = {
        "1_normal": dict.fromkeys(CLES, 0),
        "2_cataract": INCONNU | {"cataracte": 1},
        "2_glaucoma": INCONNU | {"glaucome": 1},
        "3_retina_disease": INCONNU | {"glaucome": 0, "cataracte": 0},
    }
    return [
        {"source": "sjchoi86", "groupe": f"sjchoi86-{p.stem}", "origine": str(p)} | etiquettes
        for dossier, etiquettes in regles.items()
        for p in sorted((SJCHOI / dossier).glob("*.png"))
    ]


def _smdg() -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    racine = Path(snapshot_download(SMDG, repo_type="dataset"))
    lignes = []
    for fichier in sorted(glob.glob(str(racine / "data" / "*.parquet"))):
        table = pq.read_table(fichier, columns=["label_code", "data_source"]).to_pandas()
        for i, r in enumerate(table.itertuples()):
            lignes.append(
                {
                    "source": "smdg",
                    "sous_source": r.data_source,
                    "groupe": f"smdg-{Path(fichier).stem}-{i}",
                    "origine": fichier,
                    "rang": i,
                    **(INCONNU | {"glaucome": int(r.label_code)}),
                }
            )
    df = pd.DataFrame(lignes)
    exclus = df["sous_source"].str.startswith(("OIA_ODIR", "JSIEC", "sjchoi86"))
    part_glaucome = df[~exclus & (df["glaucome"] >= 0)].groupby("sous_source")["glaucome"].mean()
    une_classe = part_glaucome[(part_glaucome < 0.1) | (part_glaucome > 0.9)].index
    garde = df[~exclus & ~df["sous_source"].isin(une_classe)]
    print(f"SMDG : {len(garde)} images gardées ; jeux à une seule classe retirés : {sorted(une_classe)}")
    return garde.to_dict("records")


# JSIEC-1000 : un seul diagnostic par image. Les catégories ambiguës pour un trouble y sont
# marquées incertaines (-1) plutôt que négatives.
JSIEC_POSITIFS = {
    "diabete": ("0.3.DR1", "1.0.DR2", "1.1.DR3"),
    "glaucome": ("10.0.Possible glaucoma",),
    "dmla": ("6.Maculopathy",),
    "hypertension": ("11.Severe hypertensive retinopathy",),
    "myopie": ("9.Pathological myopia",),
}
JSIEC_INCERTAINS = {
    "diabete": (
        "29.1.Blur fundus with suspected PDR",
        "20.Massive hard exudates",
        "22.Cotton-wool spots",
        "25.Preretinal hemorrhage",
        "27.Laser Spots",
        "18.Vitreous particles",
    ),
    "glaucome": ("0.2.Large optic cup", "10.1.Optic atrophy"),
    "cataracte": ("29.0.Blur fundus without PDR", "29.1.Blur fundus with suspected PDR"),
    "dmla": ("21.Yellow-white spots-flecks", "5.0.CSCR"),
    "hypertension": ("23.Vessel tortuosity", "22.Cotton-wool spots", "20.Massive hard exudates"),
    "myopie": ("0.1.Tessellated fundus", "24.Chorioretinal atrophy-coloboma"),
}


def _jsiec() -> list[dict]:
    lignes = []
    for dossier in sorted(p for p in JSIEC.iterdir() if p.is_dir() and p.name != "1000images"):
        etiquettes = {
            c: 1
            if dossier.name in JSIEC_POSITIFS.get(c, ())
            else (-1 if dossier.name in JSIEC_INCERTAINS.get(c, ()) else 0)
            for c in CLES
        }
        for p in sorted(dossier.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif"):
                lignes.append(
                    {"source": "jsiec", "groupe": f"jsiec-{p.stem}", "origine": str(p), "categorie": dossier.name}
                    | etiquettes
                )
    return lignes


# ------------------------------------------------------------------ préparation des images


def _signature(chemin: str) -> tuple[str, bytes]:
    """Empreinte perceptuelle (hexadécimal) et détails vasculaires normalisés (float16)."""
    import imagehash
    from scipy import ndimage as ndi

    image = Image.open(chemin)
    rgb = np.asarray(image.resize((COTE_DETAILS, COTE_DETAILS), Image.Resampling.BILINEAR), np.float32)
    champ = ndi.binary_erosion(rgb[..., 0] > 20, iterations=10)  # sans le bord du disque, commun à tous
    vert = rgb[..., 1] - ndi.gaussian_filter(rgb[..., 1], 3)
    details = np.zeros(COTE_DETAILS * COTE_DETAILS, np.float32)
    if champ.sum() > 500 and vert[champ].std() >= 1.0:  # sinon image sans détail : jamais un doublon
        d = np.where(champ, (vert - vert[champ].mean()) / vert[champ].std(), 0)
        details = (d / np.sqrt((d**2).mean())).ravel()
    return str(imagehash.phash(image)), details.astype(np.float16).tobytes()


def _preparer(tache: tuple[str, str]) -> tuple[str, tuple[str, bytes] | None]:
    origine, cible = tache
    try:
        if not Path(cible).exists():
            image = Image.open(origine).convert("RGB")
            image.thumbnail((1600, 1600))  # accélère la localisation sur les grandes images
            carre = preparer_fond_oeil(np.asarray(image), COTE_CACHE)
            Path(cible).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(carre).save(cible, quality=95)
        return cible, _signature(cible)
    except Exception as erreur:  # image illisible : écartée, mais signalée
        print(f"illisible : {origine} ({erreur})", flush=True)
        return cible, None


def _preparer_octets(octets: bytes | None, cible: str) -> tuple[str, tuple[str, bytes] | None]:
    """Image SMDG lue du Parquet par le processus principal (None : déjà en cache)."""
    try:
        if octets is not None:
            image = Image.open(io.BytesIO(octets)).convert("RGB")
            image.thumbnail((1600, 1600))
            Path(cible).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(preparer_fond_oeil(np.asarray(image), COTE_CACHE)).save(cible, quality=95)
        return cible, _signature(cible)
    except Exception as erreur:
        print(f"illisible : {cible} ({erreur})", flush=True)
        return cible, None


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_count(a[:, None] ^ b[None, :])


def construire_index(sortie: Path, processus: int) -> pd.DataFrame:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    lignes = _odir() + _rfmid() + _sjchoi() + _smdg() + _jsiec()
    df = pd.DataFrame(lignes)
    df["rang"] = df["rang"].fillna(-1).astype(int)  # position dans le fichier Parquet (SMDG)
    df["chemin"] = [
        str(sortie / "images" / s / f"{hashlib.sha1(f'{o}|{r}'.encode()).hexdigest()[:16]}.jpg")
        for s, o, r in zip(df["source"], df["origine"], df["rang"], strict=True)
    ]
    simples = df[df["rang"] < 0]
    par_fichier: dict[str, list[tuple[int, str]]] = {}
    parquets = df[df["rang"] >= 0]
    for fichier, i, cible in parquets[["origine", "rang", "chemin"]].itertuples(index=False):
        par_fichier.setdefault(fichier, []).append((int(i), cible))
    # empreintes gardées en hexadécimal : un passage par float64 (pandas, valeurs manquantes)
    # tronquerait ces entiers de 64 bits
    empreintes: dict[str, tuple[str, bytes] | None] = {}
    with ProcessPoolExecutor(processus) as ex:
        for k, (cible, h) in enumerate(
            ex.map(_preparer, zip(simples["origine"], simples["chemin"], strict=True), chunksize=16)
        ):
            empreintes[cible] = h
            if k % 2000 == 0:
                print(f"{k}/{len(simples)} images préparées", flush=True)
        # SMDG : lecture en continu par petits lots (un fichier Parquet entier ne tient pas en
        # mémoire dans chaque processus), avec un nombre borné d'images en attente
        import pyarrow.parquet as pq

        en_cours: set = set()
        faites = 0
        for fichier, elements in par_fichier.items():
            cibles = dict(elements)
            rang = 0
            for lot in pq.ParquetFile(fichier).iter_batches(batch_size=16, columns=["image"]):
                for image in lot.column("image").to_pylist():
                    cible = cibles.get(rang)
                    rang += 1
                    if cible is None:
                        continue
                    octets = None if Path(cible).exists() else image["bytes"]
                    en_cours.add(ex.submit(_preparer_octets, octets, cible))
                    if len(en_cours) >= 3 * processus:
                        finies, en_cours = wait(en_cours, return_when=FIRST_COMPLETED)
                        for f in finies:
                            empreintes.__setitem__(*f.result())
                            faites += 1
                            if faites % 1000 == 0:
                                print(f"SMDG : {faites}/{len(parquets)} images préparées", flush=True)
        for f in wait(en_cours).done:
            empreintes.__setitem__(*f.result())
    df = df[df["chemin"].map(lambda c: empreintes.get(c) is not None)].reset_index(drop=True)
    h = np.array([int(empreintes[c][0], 16) for c in df["chemin"]], dtype=np.uint64)
    details = np.stack([np.frombuffer(empreintes[c][1], np.float16) for c in df["chemin"]])
    empreintes.clear()  # libère la mémoire

    # paires candidates (empreinte proche) puis confirmation par les détails vasculaires
    premiers, seconds = [], []
    for debut in range(0, len(df), 2000):
        bloc = np.arange(debut, min(debut + 2000, len(df)))
        a, b = np.nonzero(_distances(h[bloc], h) <= DISTANCE_CANDIDATS)
        garde = bloc[a] < b
        for k0 in range(0, int(garde.sum()), 5000):
            i, j = bloc[a][garde][k0 : k0 + 5000], b[garde][k0 : k0 + 5000]
            correlation = (details[i].astype(np.float32) * details[j].astype(np.float32)).mean(axis=1)
            premiers.append(i[correlation > CORRELATION_DOUBLON])
            seconds.append(j[correlation > CORRELATION_DOUBLON])
    premiers, seconds = np.concatenate(premiers), np.concatenate(seconds)
    source = df["source"].to_numpy()

    # 1. retirer de l'entraînement tout doublon d'une image du test externe
    externe = source == "jsiec"
    fuite = np.zeros(len(df), bool)
    fuite[premiers[externe[seconds] & ~externe[premiers]]] = True
    fuite[seconds[externe[premiers] & ~externe[seconds]]] = True
    print(f"doublons du test externe retirés de l'entraînement : {int(fuite.sum())}")

    # 2. les doublons entre images d'entraînement partagent un groupe (même pli de validation)
    parent = np.arange(len(df))

    def racine(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    fusions = 0
    for i, j in zip(premiers, seconds, strict=True):
        if not (externe[i] or externe[j] or fuite[i] or fuite[j]) and racine(i) != racine(j):
            parent[racine(i)] = racine(j)
            fusions += 1
    df["groupe"] = [df["groupe"].iat[racine(i)] for i in range(len(df))]
    taille = df.loc[~externe].groupby("groupe").size()
    print(f"doublons fusionnés entre images d'entraînement : {fusions} (plus grand groupe : {taille.max()} images)")
    df = df[~fuite].reset_index(drop=True)
    df.to_csv(sortie / "index.csv", index=False)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sortie", type=Path, default=Path("data/troubles"))
    # chaque processus garde numpy, scikit-image et une image en mémoire : 6 tient dans ~8 Go (WSL)
    parser.add_argument("--processus", type=int, default=min(6, os.cpu_count() or 4))
    args = parser.parse_args(argv)
    args.sortie.mkdir(parents=True, exist_ok=True)
    df = construire_index(args.sortie, args.processus)
    resume = df.assign(**{c: df[c] == 1 for c in CLES}).groupby("source")[CLES].sum()
    resume.insert(0, "images", df.groupby("source").size())
    print(resume.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
