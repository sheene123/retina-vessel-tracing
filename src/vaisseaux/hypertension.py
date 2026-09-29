"""Étude : repérer une rétinopathie hypertensive sur ODIR-5K, marqueurs vasculaires contre réseau.

Tâche : yeux dont le diagnostic est une rétinopathie hypertensive seule, contre fonds d'œil
normaux (mots-clés propres à chaque œil). Validation croisée en 5 plis par patient : les deux
yeux d'un même patient ne sont jamais séparés entre apprentissage et test.

Trois modèles, évalués sur les mêmes plis :
- (A) marqueurs : régression logistique sur les 11 marqueurs vasculaires mesurés sur la
  segmentation du U-Net (entraîné sur DRIVE, jamais sur ODIR) ;
- (B) réseau : ResNet-18 pré-entraîné sur ImageNet, affiné sur l'image entière ;
- (C) témoin : âge et sexe seulement, pour vérifier que les modèles lisent bien l'œil.
AUROC sur les prédictions hors pli, intervalle de confiance à 95 % par bootstrap sur les patients.

    python -m vaisseaux.hypertension --odir <dossier ODIR-5K de kagglehub>
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .biomarqueurs import MARQUEURS, mesurer
from .donnees import estimer_masque_fov

PLIS = 5
GRAINE = 0


# ------------------------------------------------------------------ données


def construire_jeu(odir: Path) -> pd.DataFrame:
    df = pd.read_csv(odir / "full_df.csv")
    gauche = df["filename"].str.endswith("_left.jpg")
    mots = np.where(gauche, df["Left-Diagnostic Keywords"], df["Right-Diagnostic Keywords"])
    mots = pd.Series(mots).str.strip().str.lower()
    jeu = df.loc[mots.isin(["hypertensive retinopathy", "normal fundus"]).to_numpy()].copy()
    jeu["hypertension"] = (
        (mots[mots.isin(["hypertensive retinopathy", "normal fundus"])] == "hypertensive retinopathy")
        .astype(int)
        .to_numpy()
    )
    jeu["age"] = jeu["Patient Age"].astype(float)
    jeu["femme"] = (jeu["Patient Sex"] == "Female").astype(int)
    jeu = jeu.rename(columns={"ID": "patient"})[["filename", "patient", "hypertension", "age", "femme"]].reset_index(
        drop=True
    )
    plis = StratifiedGroupKFold(n_splits=PLIS, shuffle=True, random_state=GRAINE)
    jeu["pli"] = -1
    for k, (_, test) in enumerate(plis.split(jeu, jeu["hypertension"], groups=jeu["patient"])):
        jeu.loc[test, "pli"] = k
    return jeu


def _image(odir: Path, nom: str) -> np.ndarray:
    return np.asarray(Image.open(odir / "preprocessed_images" / nom).convert("RGB"))


# ------------------------------------------------------------------ (A) marqueurs


def _mesurer(tache: tuple) -> dict:
    nom, segmentation, masque = tache
    return {"filename": nom, **mesurer(segmentation, masque)}


def marqueurs(odir: Path, jeu: pd.DataFrame, poids_unet: Path, travailleurs: int) -> pd.DataFrame:
    from .unet import charger_modele, predire

    modele = charger_modele(poids_unet)
    noms, lignes = list(jeu["filename"]), []
    with ProcessPoolExecutor(travailleurs) as pool:
        for debut in range(0, len(noms), 200):  # par lots : la mémoire reste bornée
            taches = []
            for nom in noms[debut : debut + 200]:
                rgb = _image(odir, nom)
                masque = estimer_masque_fov(rgb)
                taches.append((nom, (predire(modele, rgb, masque) >= 0.5) & masque, masque))
            lignes += list(pool.map(_mesurer, taches, chunksize=8))
            print(f"marqueurs : {len(lignes)}/{len(noms)} yeux", flush=True)
    return pd.DataFrame(lignes)


def predictions_logistiques(jeu: pd.DataFrame, colonnes: list[str]) -> tuple[np.ndarray, dict]:
    """Prédictions hors pli d'une régression logistique, et coefficients moyens standardisés."""
    proba = np.zeros(len(jeu))
    coefficients = []
    for k in range(PLIS):
        app, test = jeu["pli"] != k, jeu["pli"] == k
        modele = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        )
        modele.fit(jeu.loc[app, colonnes], jeu.loc[app, "hypertension"])
        proba[test.to_numpy()] = modele.predict_proba(jeu.loc[test, colonnes])[:, 1]
        coefficients.append(modele[-1].coef_[0])
    return proba, dict(zip(colonnes, np.mean(coefficients, axis=0).tolist(), strict=True))


# ------------------------------------------------------------------ (B) réseau


def predictions_reseau(odir: Path, jeu: pd.DataFrame, epoques: int = 8, taille: int = 384) -> np.ndarray:
    import timm
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms as T

    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    normaliser = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    augmentation = T.Compose(
        [
            T.RandomResizedCrop(taille, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(0.15, 0.15, 0.1),
            T.ToTensor(),
            normaliser,
        ]
    )
    evaluation = T.Compose([T.Resize(taille), T.ToTensor(), normaliser])

    class Images(Dataset):
        def __init__(self, lignes: pd.DataFrame, transformation):
            self.noms, self.cibles, self.transformation = (
                list(lignes["filename"]),
                list(lignes["hypertension"]),
                transformation,
            )

        def __len__(self) -> int:
            return len(self.noms)

        def __getitem__(self, i: int):
            image = Image.open(odir / "preprocessed_images" / self.noms[i]).convert("RGB")
            return self.transformation(image), torch.tensor(float(self.cibles[i]))

    proba = np.zeros(len(jeu))
    for k in range(PLIS):
        torch.manual_seed(GRAINE + k)
        app, test = jeu[jeu["pli"] != k], jeu[jeu["pli"] == k]
        poids = np.where(
            app["hypertension"] == 1, 1.0 / app["hypertension"].sum(), 1.0 / (1 - app["hypertension"]).sum()
        )
        echantillonneur = WeightedRandomSampler(
            torch.as_tensor(poids, dtype=torch.double), num_samples=len(app), replacement=True
        )
        charge_app = DataLoader(Images(app, augmentation), batch_size=32, sampler=echantillonneur, num_workers=6)
        charge_test = DataLoader(Images(test, evaluation), batch_size=64, num_workers=6)
        modele = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=1).to(appareil)
        optimiseur = torch.optim.AdamW(modele.parameters(), lr=3e-4, weight_decay=1e-4)
        planning = torch.optim.lr_scheduler.CosineAnnealingLR(optimiseur, T_max=epoques * len(charge_app))
        for _epoque in range(epoques):
            modele.train()
            for x, y in charge_app:
                perte = torch.nn.functional.binary_cross_entropy_with_logits(
                    modele(x.to(appareil))[:, 0], y.to(appareil)
                )
                optimiseur.zero_grad()
                perte.backward()
                optimiseur.step()
                planning.step()
        modele.eval()
        sorties = []
        with torch.no_grad():
            for x, _ in charge_test:
                sorties.append(torch.sigmoid(modele(x.to(appareil))[:, 0]).cpu().numpy())
        proba[jeu["pli"].to_numpy() == k] = np.concatenate(sorties)
        print(f"pli {k + 1}/{PLIS} : AUROC {roc_auc_score(test['hypertension'], np.concatenate(sorties)):.3f}")
    return proba


# ------------------------------------------------------------------ évaluation


def auroc_patients(jeu: pd.DataFrame, scores: dict[str, np.ndarray], n: int = 2000) -> dict:
    """AUROC de chaque modèle et différences entre modèles, avec IC 95 % par bootstrap sur les patients."""
    rng = np.random.default_rng(GRAINE)
    patients = jeu["patient"].unique()
    index = {p: np.flatnonzero(jeu["patient"].to_numpy() == p) for p in patients}
    y = jeu["hypertension"].to_numpy()
    tirages = {nom: [] for nom in scores}
    for _ in range(n):
        choix = np.concatenate([index[p] for p in rng.choice(patients, len(patients))])
        if y[choix].min() == y[choix].max():
            continue
        for nom, s in scores.items():
            tirages[nom].append(roc_auc_score(y[choix], s[choix]))
    resultat = {}
    for nom, s in scores.items():
        t = np.array(tirages[nom])
        resultat[nom] = {
            "auroc": float(roc_auc_score(y, s)),
            "ic_bas": float(np.quantile(t, 0.025)),
            "ic_haut": float(np.quantile(t, 0.975)),
        }
    noms = list(scores)
    for i, a in enumerate(noms):
        for b in noms[i + 1 :]:
            d = np.array(tirages[a]) - np.array(tirages[b])
            resultat[f"{a}_moins_{b}"] = {
                "difference": resultat[a]["auroc"] - resultat[b]["auroc"],
                "ic_bas": float(np.quantile(d, 0.025)),
                "ic_haut": float(np.quantile(d, 0.975)),
            }
    return resultat


NOMS = {
    "reseau": "Réseau (ResNet-18, image entière)",
    "marqueurs": "Marqueurs vasculaires (11)",
    "age_sexe": "Témoin : âge et sexe",
}


def figure(jeu: pd.DataFrame, scores: dict[str, np.ndarray], resultat: dict, chemin: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    couleurs = {"reseau": "#0072B2", "marqueurs": "#E69F00", "age_sexe": "#7a7a7a"}
    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls="--")
    for nom, s in scores.items():
        fp, vp, _ = roc_curve(jeu["hypertension"], s)
        r = resultat[nom]
        ax.plot(
            fp,
            vp,
            lw=2,
            color=couleurs[nom],
            label=f"{NOMS[nom]} : {r['auroc']:.2f} [{r['ic_bas']:.2f} ; {r['ic_haut']:.2f}]",
        )
    ax.set_xlabel("Taux de faux positifs (1 - spécificité)", color="#555555")
    ax.set_ylabel("Taux de vrais positifs (sensibilité)", color="#555555")
    ax.set_title(
        "Rétinopathie hypertensive contre fond d'œil normal (ODIR-5K)", fontsize=10.5, color="#333333", loc="left"
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False, title="AUROC [IC 95 %]", title_fontsize=8)
    for bord in ("top", "right"):
        ax.spines[bord].set_visible(False)
    fig.savefig(chemin, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--odir", type=Path, required=True, help="dossier ODIR-5K (contient full_df.csv)")
    parser.add_argument("--unet", type=Path, default=Path("modeles/unet_drive.pt"))
    parser.add_argument("--sortie", type=Path, default=Path("resultats/hypertension"))
    parser.add_argument("--epoques", type=int, default=8)
    parser.add_argument("--travailleurs", type=int, default=10)
    args = parser.parse_args(argv)
    debut = time.perf_counter()
    args.sortie.mkdir(parents=True, exist_ok=True)

    jeu = construire_jeu(args.odir)
    print(
        f"{len(jeu)} yeux, {jeu['hypertension'].sum()} avec rétinopathie hypertensive, {jeu['patient'].nunique()} patients"
    )
    cache = args.sortie / "marqueurs.csv"
    if cache.exists():
        mesures = pd.read_csv(cache)
    else:
        mesures = marqueurs(args.odir, jeu, args.unet, args.travailleurs)
        mesures.to_csv(cache, index=False)
    jeu = jeu.merge(mesures, on="filename")

    scores, coefficients = {}, {}
    scores["marqueurs"], coefficients["marqueurs"] = predictions_logistiques(jeu, list(MARQUEURS))
    scores["age_sexe"], coefficients["age_sexe"] = predictions_logistiques(jeu, ["age", "femme"])
    scores["reseau"] = predictions_reseau(args.odir, jeu, args.epoques)
    scores = {k: scores[k] for k in ("reseau", "marqueurs", "age_sexe")}

    resultat = auroc_patients(jeu, scores)
    resultat["coefficients_marqueurs"] = coefficients["marqueurs"]
    resultat["effectifs"] = {
        "yeux": int(len(jeu)),
        "yeux_hypertension": int(jeu["hypertension"].sum()),
        "patients": int(jeu["patient"].nunique()),
        "patients_hypertension": int(jeu.loc[jeu["hypertension"] == 1, "patient"].nunique()),
    }
    resultat["duree_s"] = round(time.perf_counter() - debut, 1)
    (args.sortie / "resultats.json").write_text(json.dumps(resultat, indent=1, ensure_ascii=False))
    jeu.assign(**{f"score_{k}": v for k, v in scores.items()}).to_csv(args.sortie / "predictions.csv", index=False)
    figure(jeu, scores, resultat, args.sortie / "courbes_roc.png")
    print(
        json.dumps({k: v for k, v in resultat.items() if k != "coefficients_marqueurs"}, indent=1, ensure_ascii=False)
    )
    print(
        "coefficients (marqueurs standardisés) :",
        {k: round(v, 2) for k, v in resultat["coefficients_marqueurs"].items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
