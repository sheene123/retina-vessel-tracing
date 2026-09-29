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
hôpital). Les images d'entraînement quasi identiques à une image JSIEC (empreinte perceptuelle)
sont retirées, et les doublons entre jeux d'entraînement partagent un même groupe pour la
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
from concurrent.futures import ProcessPoolExecutor
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
DISTANCE_DOUBLON = 6  # bits différents sur 64 (empreinte perceptuelle)
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


def _preparer(tache: tuple[str, str]) -> tuple[str, str | None]:
    import imagehash

    origine, cible = tache
    try:
        if not Path(cible).exists():
            image = Image.open(origine).convert("RGB")
            image.thumbnail((1600, 1600))  # accélère la localisation sur les grandes images
            carre = preparer_fond_oeil(np.asarray(image), COTE_CACHE)
            Path(cible).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(carre).save(cible, quality=95)
        return cible, str(imagehash.phash(Image.open(cible)))
    except Exception as erreur:  # image illisible : écartée, mais signalée
        print(f"illisible : {origine} ({erreur})", flush=True)
        return cible, None


def _preparer_parquet(tache: tuple[str, list[tuple[int, str]]]) -> list[tuple[str, str | None]]:
    """Les images SMDG sont groupées par fichier Parquet : on lit chaque fichier une seule fois."""
    import imagehash
    import pyarrow.parquet as pq

    fichier, elements = tache
    colonne = None
    resultats = []
    for i, cible in elements:
        try:
            if not Path(cible).exists():
                if colonne is None:
                    colonne = pq.read_table(fichier, columns=["image"]).column("image")
                image = Image.open(io.BytesIO(colonne[i].as_py()["bytes"])).convert("RGB")
                image.thumbnail((1600, 1600))
                Path(cible).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(preparer_fond_oeil(np.asarray(image), COTE_CACHE)).save(cible, quality=95)
            resultats.append((cible, str(imagehash.phash(Image.open(cible)))))
        except Exception as erreur:
            print(f"illisible : {fichier}#{i} ({erreur})", flush=True)
            resultats.append((cible, None))
    return resultats


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
    for fichier, i, cible in df[df["rang"] >= 0][["origine", "rang", "chemin"]].itertuples(index=False):
        par_fichier.setdefault(fichier, []).append((int(i), cible))
    # empreintes gardées en hexadécimal : un passage par float64 (pandas, valeurs manquantes)
    # tronquerait ces entiers de 64 bits
    empreintes: dict[str, str | None] = {}
    with ProcessPoolExecutor(processus) as ex:
        for k, (cible, h) in enumerate(
            ex.map(_preparer, zip(simples["origine"], simples["chemin"], strict=True), chunksize=16)
        ):
            empreintes[cible] = h
            if k % 2000 == 0:
                print(f"{k}/{len(simples)} images préparées", flush=True)
        for resultats in ex.map(_preparer_parquet, par_fichier.items()):
            empreintes.update(resultats)
    df["empreinte"] = df["chemin"].map(empreintes)
    df = df[df["empreinte"].notna()].reset_index(drop=True)
    h = np.array([int(e, 16) for e in df["empreinte"]], dtype=np.uint64)

    # 1. retirer de l'entraînement tout quasi-doublon d'une image du test externe
    test = (df["source"] == "jsiec").to_numpy()
    fuite = (_distances(h[~test], h[test]) <= DISTANCE_DOUBLON).any(axis=1)
    print(f"doublons du test externe retirés de l'entraînement : {int(fuite.sum())}")
    garde = np.ones(len(df), bool)
    garde[np.flatnonzero(~test)[fuite]] = False
    df, h = df[garde].reset_index(drop=True), h[garde]

    # 2. les doublons entre jeux d'entraînement partagent un groupe (même pli de validation)
    entrainement = np.flatnonzero((df["source"] != "jsiec").to_numpy())
    parent = np.arange(len(df))

    def racine(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    paires = 0
    for debut in range(0, len(entrainement), 2000):
        bloc = entrainement[debut : debut + 2000]
        proches = _distances(h[bloc], h[entrainement]) <= DISTANCE_DOUBLON
        for a, b in zip(*np.nonzero(proches), strict=True):
            i, j = bloc[a], entrainement[b]
            if i < j and racine(i) != racine(j):
                parent[racine(i)] = racine(j)
                paires += 1
    df["groupe"] = [df["groupe"].iat[racine(i)] for i in range(len(df))]
    print(f"doublons fusionnés entre jeux d'entraînement : {paires}")
    df = df.drop(columns=["empreinte"])
    df.to_csv(sortie / "index.csv", index=False)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sortie", type=Path, default=Path("data/troubles"))
    parser.add_argument("--processus", type=int, default=min(12, os.cpu_count() or 4))
    args = parser.parse_args(argv)
    args.sortie.mkdir(parents=True, exist_ok=True)
    df = construire_index(args.sortie, args.processus)
    resume = df.assign(**{c: df[c] == 1 for c in CLES}).groupby("source")[CLES].sum()
    resume.insert(0, "images", df.groupby("source").size())
    print(resume.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
