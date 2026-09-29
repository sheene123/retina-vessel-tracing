"""Détection de 6 troubles de l'œil sur ODIR-5K, avec des pourcentages calibrés pour la démo.

Un seul réseau (EfficientNet-B0 pré-entraîné sur ImageNet, 384 × 384) prédit, pour chaque œil,
la probabilité de 6 troubles. Les étiquettes viennent des mots-clés propres à chaque œil ; les
diagnostics « suspects » sont marqués incertains et ignorés pour le trouble concerné.

1. Validation croisée en 5 plis, groupée par patient : prédiction de chaque œil par un modèle qui
   ne l'a jamais vu, AUROC par trouble avec IC 95 % par bootstrap sur les patients.
2. Calibrage (Platt) par trouble sur ces prédictions hors pli : un « 70 % » affiché correspond à
   environ 7 yeux sur 10 réellement atteints, dans une population comme celle d'ODIR.
3. Modèle final entraîné sur toutes les images, exporté en ONNX pour le navigateur.

    python -m vaisseaux.troubles --odir <dossier ODIR-5K de kagglehub>
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

# clé -> (nom affiché, mots-clés positifs, mots-clés incertains)
TROUBLES = {
    "diabete": (
        "Rétinopathie diabétique",
        ("nonproliferative retinopathy", "non proliferative retinopathy", "diabetic retinopathy"),
        ("suspected diabetic retinopathy", "suspicious diabetic retinopathy"),
    ),
    "glaucome": ("Glaucome", ("glaucoma",), ("suspected glaucoma",)),
    "cataracte": ("Cataracte", ("cataract",), ()),
    "dmla": ("DMLA (dégénérescence maculaire liée à l'âge)", ("age-related macular degeneration",), ("drusen",)),
    "hypertension": ("Rétinopathie hypertensive", ("hypertensive retinopathy",), ()),
    "myopie": (
        "Myopie forte",
        ("pathological myopia", "myopia retinopathy", "myopic maculopathy"),
        ("tessellated fundus",),
    ),
}
CLES = list(TROUBLES)
PLIS = 5
GRAINE = 0
TAILLE = 384


def _etiquette(mots: list[str], positifs: tuple[str, ...], incertains: tuple[str, ...]) -> int:
    if any(m in incertains for m in mots):
        return -1
    return int(any(any(p in m for p in positifs) for m in mots))


def construire_jeu(odir: Path) -> pd.DataFrame:
    df = pd.read_csv(odir / "full_df.csv")
    gauche = df["filename"].str.endswith("_left.jpg")
    texte = np.where(gauche, df["Left-Diagnostic Keywords"], df["Right-Diagnostic Keywords"])
    mots = [[m.strip().lower() for m in re.split("[，,]", str(t))] for t in texte]
    jeu = pd.DataFrame({"filename": df["filename"], "patient": df["ID"]})
    for cle, (_, positifs, incertains) in TROUBLES.items():
        # une rétinopathie diabétique « suspectée » ne doit pas compter comme positive
        jeu[cle] = [_etiquette(m, positifs, incertains) for m in mots]
    jeu["normal"] = [int(m == ["normal fundus"]) for m in mots]
    # classe de stratification : premier trouble présent, sinon normal / autre
    premier = np.full(len(jeu), "autre", dtype=object)
    for cle in reversed(CLES):
        premier[jeu[cle].to_numpy() == 1] = cle
    premier[jeu["normal"].to_numpy() == 1] = "normal"
    plis = StratifiedGroupKFold(n_splits=PLIS, shuffle=True, random_state=GRAINE)
    jeu["pli"] = -1
    for k, (_, test) in enumerate(plis.split(jeu, premier, groups=jeu["patient"])):
        jeu.loc[test, "pli"] = k
    return jeu


# ------------------------------------------------------------------ apprentissage


def _modele():
    import timm

    return timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, num_classes=len(CLES))


def _chargeur(odir: Path, lignes: pd.DataFrame, entrainement: bool):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T

    normaliser = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if entrainement:
        transformation = T.Compose(
            [
                T.RandomResizedCrop(TAILLE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(0.15, 0.15, 0.1),
                T.ToTensor(),
                normaliser,
            ]
        )
    else:
        transformation = T.Compose([T.Resize(TAILLE), T.ToTensor(), normaliser])

    class Images(Dataset):
        def __len__(self) -> int:
            return len(lignes)

        def __getitem__(self, i: int):
            ligne = lignes.iloc[i]
            image = Image.open(odir / "preprocessed_images" / ligne["filename"]).convert("RGB")
            return transformation(image), torch.tensor(ligne[CLES].to_numpy(dtype=np.float32))

    return DataLoader(
        Images(), batch_size=32 if entrainement else 64, shuffle=entrainement, num_workers=6, drop_last=entrainement
    )


def entrainer(odir: Path, lignes: pd.DataFrame, epoques: int):
    import torch

    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    modele = _modele().to(appareil)
    y = lignes[CLES].to_numpy()
    positifs, negatifs = (y == 1).sum(0), (y == 0).sum(0)
    poids_positifs = torch.tensor(
        np.clip(negatifs / np.maximum(positifs, 1), 1, 10), dtype=torch.float32, device=appareil
    )
    charge = _chargeur(odir, lignes, entrainement=True)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=3e-4, weight_decay=1e-4)
    planning = torch.optim.lr_scheduler.OneCycleLR(optimiseur, max_lr=3e-4, total_steps=epoques * len(charge))
    for _epoque in range(epoques):
        modele.train()
        for x, cible in charge:
            x, cible = x.to(appareil), cible.to(appareil)
            with torch.autocast(appareil, dtype=torch.bfloat16, enabled=appareil == "cuda"):
                logits = modele(x)
            connu = cible >= 0  # étiquette incertaine : ignorée pour ce trouble
            perte = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), cible.clamp(min=0), pos_weight=poids_positifs, reduction="none"
            )
            perte = (perte * connu).sum() / connu.sum()
            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()
            planning.step()
    return modele.eval()


def logits(odir: Path, modele, lignes: pd.DataFrame) -> np.ndarray:
    import torch

    appareil = next(modele.parameters()).device
    sorties = []
    with torch.no_grad():
        for x, _ in _chargeur(odir, lignes, entrainement=False):
            sorties.append(modele(x.to(appareil)).float().cpu().numpy())
    return np.concatenate(sorties)


# ------------------------------------------------------------------ évaluation et calibrage


def calibrer(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Calibrage de Platt : p = sigmoïde(a z + b), ajusté sur les prédictions hors pli."""
    ok = y >= 0
    regression = LogisticRegression(C=1e4, max_iter=1000).fit(z[ok, None], y[ok])
    return float(regression.coef_[0, 0]), float(regression.intercept_[0])


def evaluer(jeu: pd.DataFrame, z: np.ndarray, calibrage: dict, n: int = 2000) -> dict:
    rng = np.random.default_rng(GRAINE)
    patients = jeu["patient"].unique()
    index = {p: np.flatnonzero(jeu["patient"].to_numpy() == p) for p in patients}
    tirages = [np.concatenate([index[p] for p in rng.choice(patients, len(patients))]) for _ in range(n)]
    resultat = {}
    for j, cle in enumerate(CLES):
        y = jeu[cle].to_numpy()
        ok = y >= 0
        a, b = calibrage[cle]
        p = 1 / (1 + np.exp(-(a * z[:, j] + b)))
        aurocs = []
        for t in tirages:
            t = t[ok[t]]
            if 0 < y[t].sum() < len(t):
                aurocs.append(roc_auc_score(y[t], z[t, j]))
        # à chaque niveau affiché dans la démo : part des yeux atteints parmi ceux classés ainsi
        niveaux = {}
        for nom, bas, haut in (("peu_probable", 0, 0.2), ("possible", 0.2, 0.5), ("probable", 0.5, 1.01)):
            dans = ok & (p >= bas) & (p < haut)
            niveaux[nom] = {"yeux": int(dans.sum()), "part_atteints": float(y[dans].mean()) if dans.any() else None}
        sensibilite = float(((p >= 0.2) & (y == 1)).sum() / max((y == 1).sum(), 1))
        resultat[cle] = {
            "nom": TROUBLES[cle][0],
            "yeux": int(ok.sum()),
            "atteints": int((y == 1).sum()),
            "auroc": float(roc_auc_score(y[ok], z[ok, j])),
            "ic_bas": float(np.quantile(aurocs, 0.025)),
            "ic_haut": float(np.quantile(aurocs, 0.975)),
            "detectes_des_possible": sensibilite,
            "niveaux": niveaux,
        }
    return resultat


def figure(resultat: dict, chemin: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordre = sorted(CLES, key=lambda c: resultat[c]["auroc"])
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    for i, cle in enumerate(ordre):
        r = resultat[cle]
        ax.plot([r["ic_bas"], r["ic_haut"]], [i, i], color="#9ec1dc", lw=6, solid_capstyle="round")
        ax.plot(r["auroc"], i, "o", color="#0b5d8c", ms=8, mec="white", mew=1.5)
        ax.text(r["ic_haut"] + 0.01, i, f"{r['auroc']:.2f}", va="center", fontsize=9, color="#333333")
    ax.set_yticks(
        range(len(ordre)), [f"{resultat[c]['nom'].split(' (')[0]} ({resultat[c]['atteints']} yeux)" for c in ordre]
    )
    ax.axvline(0.5, color="#bbbbbb", lw=1, ls="--")
    ax.set_xlim(0.45, 1.05)
    ax.set_xlabel("AUROC hors pli (point) et IC 95 % par patients (barre)", color="#555555")
    ax.set_title("Qualité de détection par trouble, ODIR-5K", fontsize=10.5, color="#333333", loc="left")
    for bord in ("top", "right", "left"):
        ax.spines[bord].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(chemin, dpi=130)
    plt.close(fig)


def exporter(modele, chemin: Path) -> float:
    import onnxruntime as ort
    import torch

    modele = modele.float().cpu().eval()
    x = torch.randn(1, 3, TAILLE, TAILLE)
    torch.onnx.export(
        modele,
        (x,),
        str(chemin),
        input_names=["image"],
        output_names=["logits"],
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    with torch.no_grad():
        attendu = modele(x).numpy()
    obtenu = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    return float(np.abs(obtenu - attendu).max())


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--odir", type=Path, required=True)
    parser.add_argument("--sortie", type=Path, default=Path("resultats/troubles"))
    parser.add_argument("--modele", type=Path, default=Path("modeles/troubles.onnx"))
    parser.add_argument("--epoques", type=int, default=10)
    args = parser.parse_args(argv)
    debut = time.perf_counter()
    args.sortie.mkdir(parents=True, exist_ok=True)
    args.modele.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(GRAINE)

    jeu = construire_jeu(args.odir)
    print(
        f"{len(jeu)} yeux, {jeu['patient'].nunique()} patients ; atteints : "
        + ", ".join(f"{c} {int((jeu[c] == 1).sum())}" for c in CLES)
    )
    z = np.zeros((len(jeu), len(CLES)), dtype=np.float32)
    for k in range(PLIS):
        app, test = jeu[jeu["pli"] != k], jeu[jeu["pli"] == k]
        modele = entrainer(args.odir, app, args.epoques)
        z[jeu["pli"].to_numpy() == k] = logits(args.odir, modele, test)
        print(
            f"pli {k + 1}/{PLIS} : "
            + ", ".join(
                f"{c} {roc_auc_score(test[c][test[c] >= 0], z[jeu['pli'].to_numpy() == k][:, j][test[c].to_numpy() >= 0]):.2f}"
                for j, c in enumerate(CLES)
            ),
            flush=True,
        )

    calibrage = {c: calibrer(z[:, j], jeu[c].to_numpy()) for j, c in enumerate(CLES)}
    resultat = evaluer(jeu, z, calibrage)
    final = entrainer(args.odir, jeu, args.epoques)
    ecart = exporter(final, args.modele)
    resultat_complet = {
        "troubles": resultat,
        "calibrage": calibrage,
        "parite_onnx": ecart,
        "effectifs": {"yeux": int(len(jeu)), "patients": int(jeu["patient"].nunique())},
        "duree_s": round(time.perf_counter() - debut, 1),
    }
    (args.sortie / "resultats.json").write_text(json.dumps(resultat_complet, indent=1, ensure_ascii=False))
    # fichier lu par la démo : ordre des sorties, noms, calibrage, qualité de détection
    demo = {
        "ordre": CLES,
        "troubles": {
            c: {
                "nom": resultat[c]["nom"],
                "a": calibrage[c][0],
                "b": calibrage[c][1],
                "auroc": resultat[c]["auroc"],
                "ic_bas": resultat[c]["ic_bas"],
                "ic_haut": resultat[c]["ic_haut"],
                "detectes_des_possible": resultat[c]["detectes_des_possible"],
                "niveaux": resultat[c]["niveaux"],
            }
            for c in CLES
        },
    }
    (args.sortie / "troubles_demo.json").write_text(json.dumps(demo, indent=1, ensure_ascii=False))
    jeu.assign(**{f"logit_{c}": z[:, j] for j, c in enumerate(CLES)}).to_csv(
        args.sortie / "predictions.csv", index=False
    )
    figure(resultat, args.sortie / "auroc_par_trouble.png")
    print(
        json.dumps(
            {
                c: {
                    k: v
                    for k, v in r.items()
                    if k in ("auroc", "ic_bas", "ic_haut", "atteints", "detectes_des_possible")
                }
                for c, r in resultat.items()
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    print(f"parité ONNX : {ecart:.2e} ; durée {resultat_complet['duree_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
