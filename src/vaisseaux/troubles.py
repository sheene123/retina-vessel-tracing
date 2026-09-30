"""Détection de 6 troubles de l'œil, entraînée sur plusieurs jeux publics et testée sur un
hôpital jamais vu, avec des pourcentages calibrés pour la démo.

Un seul réseau (EfficientNet pré-entraîné sur ImageNet, 384 × 384) prédit, pour chaque œil, la
probabilité de 6 troubles. Les images viennent d'ODIR-5K, RFMiD, SMDG-19 et sjchoi86, réunies et
dédoublonnées par `vaisseaux.sources_troubles` ; chaque jeu n'étiquette que certains troubles, les
autres sont ignorés par la perte. Les augmentations imitent les images « du monde réel » : zoom
sur la papille, compression JPEG, flou, basse résolution, dominante de couleur, flèches et traits
d'annotation comme dans un manuel.

1. Validation croisée en 5 plis, groupée par patient (et par doublon entre jeux) : AUROC par
   trouble avec IC 95 % par bootstrap sur les groupes.
2. Calibrage (Platt) par trouble sur ces prédictions hors pli.
3. Modèle final entraîné sur toutes les images d'entraînement, exporté en ONNX.
4. Test externe : JSIEC-1000, un autre hôpital, jamais utilisé pour entraîner ni régler.

    python -m vaisseaux.sources_troubles          # une fois : prépare data/troubles/
    python -m vaisseaux.troubles
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

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

ARCHITECTURE = "efficientnet_b0.ra_in1k"


# Gros plans de la papille (photos prises de près) : une part des images d'entraînement est recadrée
# sur la papille ; sur ces gros plans, seuls les troubles visibles autour de la papille gardent leur
# étiquette (les lésions de la macula ou de la périphérie sortent du cadre).
PART_GROS_PLANS, CADRAGES_GROS_PLAN, TROUBLES_PAPILLE = 0.2, (0.18, 0.6), ("glaucome", "myopie")
# Distillation : la sortie glaucome apprend aussi l'avis d'un professeur (RETFound, voir
# vaisseaux.professeur), donné pour chaque image par un modèle qui ne l'a pas vue à l'entraînement.
POIDS_PROFESSEUR = 0.5
GROS_PLAN_TEST = 0.2  # test : papille qui remplit presque l'image, comme une photo de papille

POIDS_INITIAUX: Path | None = None  # fichier de poids ImageNet, quand la machine n'a pas internet


def _modele(architecture: str = ARCHITECTURE, pretrained: bool = True):
    import timm

    options = {"pretrained_cfg_overlay": {"file": str(POIDS_INITIAUX)}} if pretrained and POIDS_INITIAUX else {}
    return timm.create_model(architecture, pretrained=pretrained, num_classes=len(CLES), **options)


class Annotations:
    """Trace au hasard des flèches, traits et lettres, comme sur une figure de manuel ou une
    capture d'écran annotée : le réseau apprend à ne pas en tenir compte."""

    def __init__(self, p: float = 0.25):
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        import random

        from PIL import ImageDraw

        if random.random() > self.p:
            return image
        image = image.copy()
        dessin = ImageDraw.Draw(image)
        cote = image.size[0]
        for _ in range(random.randint(1, 4)):
            couleur = random.choice([(0, 0, 0), (255, 255, 255), (20, 20, 20), (255, 255, 0), (0, 0, 255)])
            x0, y0 = random.uniform(0, cote), random.uniform(0, cote)
            x1, y1 = x0 + random.uniform(-cote / 3, cote / 3), y0 + random.uniform(-cote / 3, cote / 3)
            dessin.line([(x0, y0), (x1, y1)], fill=couleur, width=random.randint(1, 4))
            if random.random() < 0.5:
                dessin.text((x0, y0), random.choice("ABCDEFabcdef123→*"), fill=couleur)
        return image


class BasseResolution:
    """Réduit puis ré-agrandit l'image : photo prise de loin, miniature web, capture d'écran."""

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        import random

        if random.random() > self.p:
            return image
        cote = image.size[0]
        petit = random.randint(cote // 4, cote // 2)
        return image.resize((petit, petit), Image.Resampling.BILINEAR).resize((cote, cote), Image.Resampling.BILINEAR)


def transformations(entrainement: bool):
    import torch
    from torchvision.transforms import v2 as T

    fin = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    if not entrainement:
        return T.Compose([T.Resize(TAILLE), *fin])
    return T.Compose(
        [
            # zoom : de l'œil entier jusqu'à un gros plan (papille, macula) comme sur les photos web
            T.RandomResizedCrop(TAILLE, scale=(0.35, 1.0), ratio=(0.85, 1.18)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(0.2),
            T.RandomRotation(25),
            T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.03),
            T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.5))], p=0.25),
            BasseResolution(0.25),
            Annotations(0.25),
            T.RandomApply([T.JPEG((25, 90))], p=0.4),
            *fin,
        ]
    )


def _chargeur(
    lignes: pd.DataFrame,
    entrainement: bool,
    images_par_epoque: int | None = None,
    gros_plan: float | None = None,
    colonnes: list[str] | None = None,
    transformation=None,
    taille_lot: int | None = None,
    part_gros_plans: float = PART_GROS_PLANS,
):
    """Images et cibles. À l'entraînement, une part des images devient un gros plan de la papille
    (étiquettes des troubles invisibles ignorées) ; en test, `gros_plan` recadre toutes les images.
    La colonne « prof_glaucome », si présente, est ajoutée en dernière cible (-1 si absente)."""
    import random

    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    from vaisseaux.donnees import gros_plan_papille

    transformation = transformation or transformations(entrainement)
    colonnes = colonnes or CLES
    chemins = lignes["chemin"].tolist()
    valeurs = lignes[colonnes].to_numpy(dtype=np.float32)
    if "prof_glaucome" in lignes:
        valeurs = np.column_stack([valeurs, np.nan_to_num(lignes["prof_glaucome"].to_numpy(np.float32), nan=-1)])
    cibles = torch.tensor(valeurs)
    hors_papille = [j for j, c in enumerate(colonnes) if c not in TROUBLES_PAPILLE]

    class Images(Dataset):
        def __len__(self) -> int:
            return len(chemins)

        def __getitem__(self, i: int):
            image, cible = Image.open(chemins[i]).convert("RGB"), cibles[i].clone()
            part = gros_plan
            if entrainement and random.random() < part_gros_plans:
                part = random.uniform(*CADRAGES_GROS_PLAN)
                cible[hors_papille] = -1
            if part:
                image = Image.fromarray(gros_plan_papille(np.asarray(image), part))
            return transformation(image), cible

    echantillonneur = None
    if entrainement:
        # chaque jeu pèse selon la racine de sa taille : SMDG (glaucome seul) ne noie pas les autres
        taille_source = lignes["source"].map(lignes["source"].value_counts()).to_numpy(dtype=np.float64)
        echantillonneur = WeightedRandomSampler(
            torch.tensor(1 / np.sqrt(taille_source)), num_samples=images_par_epoque or len(lignes), replacement=True
        )
    return DataLoader(
        Images(),
        batch_size=taille_lot or (32 if entrainement else 64),
        sampler=echantillonneur,
        num_workers=min(6, os.cpu_count() or 4),  # plus saturerait la mémoire de WSL (7,5 Go)
        drop_last=entrainement,
        persistent_workers=False,
    )


def entrainer(
    lignes: pd.DataFrame,
    epoques: int,
    images_par_epoque: int,
    architecture: str = ARCHITECTURE,
    progression=None,
    etape: str = "",
    part_gpu: float = 1.0,
):
    """`progression` : barre tqdm partagée par tous les entraînements, avancée à chaque lot.
    `part_gpu` < 1 : après chaque lot, pause proportionnelle au temps de calcul pour que le GPU
    ne travaille en moyenne que cette part du temps (moins de chauffe et de bruit sur un portable,
    où le pilote ne permet pas de plafonner la puissance)."""
    import torch

    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    # format mémoire « channels_last » : +50 % d'images par seconde sur GPU récent en bf16
    modele = _modele(architecture).to(appareil, memory_format=torch.channels_last)
    y = lignes[CLES].to_numpy()
    positifs, negatifs = (y == 1).sum(0), (y == 0).sum(0)
    poids_positifs = torch.tensor(
        np.clip(negatifs / np.maximum(positifs, 1), 1, 10), dtype=torch.float32, device=appareil
    )
    charge = _chargeur(lignes, entrainement=True, images_par_epoque=images_par_epoque)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=3e-4, weight_decay=1e-4)
    planning = torch.optim.lr_scheduler.OneCycleLR(optimiseur, max_lr=3e-4, total_steps=epoques * len(charge))
    bridage = appareil == "cuda" and part_gpu < 1
    # bf16 sur les GPU récents (Ampere et après), sinon fp16 avec mise à l'échelle de la perte (T4)
    bf16 = appareil == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    precision = torch.bfloat16 if bf16 else torch.float16
    echelle = torch.amp.GradScaler("cuda", enabled=appareil == "cuda" and not bf16)
    for epoque in range(epoques):
        modele.train()
        if progression is not None:
            progression.set_description(f"{etape} · époque {epoque + 1}/{epoques}")
        for n, (x, cible) in enumerate(charge):
            if bridage:
                torch.cuda.synchronize()
                debut_lot = time.perf_counter()
            x = x.to(appareil, non_blocking=True, memory_format=torch.channels_last)
            cible = cible.to(appareil, non_blocking=True)
            with torch.autocast(appareil, dtype=precision, enabled=appareil == "cuda"):
                logits = modele(x)
            avis = cible[:, len(CLES)] if cible.shape[1] > len(CLES) else None
            cible = cible[:, : len(CLES)]
            connu = cible >= 0  # trouble non étiqueté par ce jeu, ou incertain : ignoré
            perte = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), cible.clamp(min=0), pos_weight=poids_positifs, reduction="none"
            )
            perte = (perte * connu).sum() / connu.sum().clamp(min=1)
            if avis is not None and (avis >= 0).any():
                # distillation : probabilité du professeur comme cible douce pour le glaucome
                a_un_avis = avis >= 0
                perte = perte + POIDS_PROFESSEUR * torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[a_un_avis, CLES.index("glaucome")].float(), avis[a_un_avis]
                )
            optimiseur.zero_grad()
            echelle.scale(perte).backward()
            echelle.step(optimiseur)
            echelle.update()
            planning.step()
            if bridage:
                torch.cuda.synchronize()
                time.sleep((time.perf_counter() - debut_lot) * (1 / part_gpu - 1))
            if progression is not None:
                progression.update(1)
                if n % 50 == 0:
                    progression.set_postfix_str(f"perte {perte.item():.3f}", refresh=False)
    return modele.eval()


def logits(modele, lignes: pd.DataFrame, gros_plan: float | None = None) -> np.ndarray:
    import torch

    appareil = next(modele.parameters()).device
    sorties = []
    with torch.no_grad():
        for x, _ in _chargeur(lignes, entrainement=False, gros_plan=gros_plan):
            sorties.append(modele(x.to(appareil, memory_format=torch.channels_last)).float().cpu().numpy())
    return np.concatenate(sorties)


# ------------------------------------------------------------------ évaluation et calibrage


# Niveaux affichés dans la démo. Le seuil « probable » est commun (plus d'une chance sur deux) ; le
# seuil « possible » dépend de la maladie : placé pour signaler environ 8 yeux atteints sur 10 en
# validation croisée, borné entre 3 % et 20 %. Sans cela, une maladie rare dans les données
# (rétinopathie hypertensive : 2 % des yeux) n'atteint presque jamais 20 % même quand le réseau
# classe bien les patients. Les pourcentages affichés, eux, restent calibrés.
SENSIBILITE_VISEE, SEUIL_POSSIBLE_MIN, SEUIL_POSSIBLE_MAX, SEUIL_PROBABLE = 0.8, 0.03, 0.2, 0.5


def seuil_possible(p_atteints: np.ndarray) -> float:
    if not len(p_atteints):
        return SEUIL_POSSIBLE_MAX
    s = float(np.quantile(p_atteints, 1 - SENSIBILITE_VISEE))
    return min(SEUIL_POSSIBLE_MAX, max(SEUIL_POSSIBLE_MIN, s))


def calibrer(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Calibrage de Platt : p = sigmoïde(a z + b), ajusté sur les prédictions hors pli."""
    ok = y >= 0
    regression = LogisticRegression(C=1e4, max_iter=1000).fit(z[ok, None], y[ok])
    return float(regression.coef_[0, 0]), float(regression.intercept_[0])


def evaluer(jeu: pd.DataFrame, z: np.ndarray, calibrage: dict, n: int = 2000) -> dict:
    rng = np.random.default_rng(GRAINE)
    # bootstrap sur les groupes (patients) : un tirage = un poids par image, le nombre de fois où
    # son groupe est tiré
    codes, _ = pd.factorize(jeu["groupe"])
    nombre = codes.max() + 1
    tirages = [np.bincount(rng.integers(0, nombre, nombre), minlength=nombre)[codes] for _ in range(n)]
    resultat = {}
    for j, cle in enumerate(CLES):
        y = jeu[cle].to_numpy()
        ok = y >= 0
        a, b = calibrage[cle]
        p = 1 / (1 + np.exp(-(a * z[:, j] + b)))
        aurocs = []
        for w in tirages:
            w = w[ok]
            if 0 < w[y[ok] == 1].sum() < w.sum():
                aurocs.append(roc_auc_score(y[ok], z[ok, j], sample_weight=w))
        # à chaque niveau affiché dans la démo : part des yeux atteints parmi ceux classés ainsi
        niveaux = {}
        seuil = seuil_possible(p[ok & (y == 1)])
        for nom, bas, haut in (
            ("peu_probable", 0, seuil),
            ("possible", seuil, SEUIL_PROBABLE),
            ("probable", SEUIL_PROBABLE, 1.01),
        ):
            dans = ok & (p >= bas) & (p < haut)
            niveaux[nom] = {"yeux": int(dans.sum()), "part_atteints": float(y[dans].mean()) if dans.any() else None}
        sensibilite = float(((p >= seuil) & (y == 1)).sum() / max((y == 1).sum(), 1))
        resultat[cle] = {
            "nom": TROUBLES[cle][0],
            "yeux": int(ok.sum()),
            "atteints": int((y == 1).sum()),
            "auroc": float(roc_auc_score(y[ok], z[ok, j])),
            "ic_bas": float(np.quantile(aurocs, 0.025)),
            "ic_haut": float(np.quantile(aurocs, 0.975)),
            "detectes_des_possible": sensibilite,
            "fausses_alertes": float(((p >= seuil) & (y == 0)).sum() / max((y == 0).sum(), 1)),
            "seuil_possible": seuil,
            "niveaux": niveaux,
        }
    return resultat


def figure(resultat: dict, chemin: Path, externe: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordre = sorted(CLES, key=lambda c: resultat[c]["auroc"])
    fig, ax = plt.subplots(figsize=(7.6, 4.0), constrained_layout=True)
    series = [(resultat, 0.12, "#0b5d8c", "#9ec1dc", "validation croisée")]
    if externe:
        series.append((externe, -0.12, "#c2410c", "#f5c4a8", "test externe JSIEC"))
    for donnees, decalage, couleur, clair, legende in series:
        for i, cle in enumerate(ordre):
            r = donnees[cle]
            if "auroc" not in r:
                continue
            y = i + decalage
            ax.plot([r["ic_bas"], r["ic_haut"]], [y, y], color=clair, lw=5, solid_capstyle="round")
            ax.plot(r["auroc"], y, "o", color=couleur, ms=7, mec="white", mew=1.2, label=legende if i == 0 else None)
            ax.text(r["ic_haut"] + 0.01, y, f"{r['auroc']:.2f}", va="center", fontsize=8.5, color="#333333")
    ax.set_yticks(range(len(ordre)), [resultat[c]["nom"].split(" (")[0] for c in ordre])
    ax.axvline(0.5, color="#bbbbbb", lw=1, ls="--")
    ax.set_xlim(0.45, 1.06)
    ax.set_xlabel("AUROC (point) et IC 95 % (barre)", color="#555555")
    ax.set_title("Qualité de détection par trouble", fontsize=10.5, color="#333333", loc="left")
    if externe:
        ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    for bord in ("top", "right", "left"):
        ax.spines[bord].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(chemin, dpi=130)
    plt.close(fig)


def _avec_cartes(modele):
    """Enveloppe pour l'export : en plus des logits, les cartes d'activation de classe (CAM) de
    chaque trouble, calculées à partir des cartes de caractéristiques et des poids de la couche
    finale. La moyenne de chaque carte redonne exactement le logit : la carte montre quelles zones
    de l'image font monter ou baisser le score, sans approximation."""
    import torch

    class AvecCartes(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.modele = modele

        def forward(self, x):
            caracteristiques = self.modele.forward_features(x)
            tete = self.modele.classifier
            cartes = torch.einsum("bkhw,ck->bchw", caracteristiques, tete.weight) + tete.bias[None, :, None, None]
            return cartes.mean(dim=(2, 3)), cartes

    return AvecCartes()


def exporter(modele, chemin: Path, exemples: np.ndarray) -> float:
    """Export ONNX ; la parité est vérifiée sur de vraies images (un bruit aléatoire, hors
    distribution, amplifie des écarts numériques sans intérêt)."""
    import onnxruntime as ort
    import torch

    modele = modele.float().cpu().to(memory_format=torch.contiguous_format).eval()
    x = torch.from_numpy(exemples)
    avec_cartes = hasattr(modele, "forward_features") and hasattr(modele, "classifier")
    torch.onnx.export(
        _avec_cartes(modele) if avec_cartes else modele,
        (x[:1],),
        str(chemin),
        input_names=["image"],
        output_names=["logits", "cartes"] if avec_cartes else ["logits"],
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    session = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        attendu = modele(x).numpy()
    obtenu = np.concatenate([session.run(None, {"image": exemples[i : i + 1]})[0] for i in range(len(exemples))])
    return float(np.abs(obtenu - attendu).max())


def evaluer_externe(
    externe: pd.DataFrame, z: np.ndarray, calibrage: dict, seuils: dict | None = None, n: int = 2000
) -> dict:
    """Test externe (JSIEC) : AUROC avec IC 95 % par bootstrap sur les images, et ce que voit
    l'utilisateur de la démo : part des yeux atteints signalés au moins « possible » (≥ 20 %) et
    part des yeux sains qui restent « peu probable »."""
    rng = np.random.default_rng(GRAINE)
    resultat = {}
    for j, cle in enumerate(CLES):
        y = externe[cle].to_numpy()
        ok = y >= 0
        if not 0 < (y[ok] == 1).sum() < ok.sum():
            resultat[cle] = {"nom": TROUBLES[cle][0], "yeux": int(ok.sum()), "atteints": int((y == 1).sum())}
            continue
        a, b = calibrage[cle]
        seuil = (seuils or {}).get(cle, SEUIL_POSSIBLE_MAX)
        p = 1 / (1 + np.exp(-(a * z[:, j] + b)))
        yk, zk, pk = y[ok], z[ok, j], p[ok]
        aurocs = []
        for _ in range(n):
            t = rng.integers(0, len(yk), len(yk))
            if 0 < yk[t].sum() < len(t):
                aurocs.append(roc_auc_score(yk[t], zk[t]))
        resultat[cle] = {
            "nom": TROUBLES[cle][0],
            "yeux": int(ok.sum()),
            "atteints": int((yk == 1).sum()),
            "auroc": float(roc_auc_score(yk, zk)),
            "ic_bas": float(np.quantile(aurocs, 0.025)),
            "ic_haut": float(np.quantile(aurocs, 0.975)),
            "sensibilite_possible": float((pk[yk == 1] >= seuil).mean()),
            "specificite_peu_probable": float((pk[yk == 0] < seuil).mean()),
        }
    return resultat


def _plis(jeu: pd.DataFrame) -> np.ndarray:
    premier = np.full(len(jeu), "autre", dtype=object)
    for cle in reversed(CLES):
        premier[jeu[cle].to_numpy() == 1] = cle
    strate = jeu["source"].to_numpy().astype(object) + "-" + premier
    plis = np.full(len(jeu), -1)
    decoupe = StratifiedGroupKFold(n_splits=PLIS, shuffle=True, random_state=GRAINE)
    for k, (_, test) in enumerate(decoupe.split(jeu, strate, groups=jeu["groupe"])):
        plis[test] = k
    return plis


def _auroc(y: np.ndarray, z: np.ndarray) -> float:
    ok = y >= 0
    return float(roc_auc_score(y[ok], z[ok])) if 0 < (y[ok] == 1).sum() < ok.sum() else float("nan")


def _verrouiller(chemin: Path):
    """Verrou exclusif : deux entraînements simultanés saturent la mémoire et le GPU. Le verrou est
    libéré automatiquement à la fin du processus, même en cas d'arrêt brutal."""
    import fcntl

    fichier = open(chemin, "w")  # noqa: SIM115 (gardé ouvert pendant tout l'entraînement)
    try:
        fcntl.flock(fichier, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fichier.close()
        return None
    fichier.write(str(os.getpid()))
    fichier.flush()
    return fichier


def lire_index(chemin: Path) -> pd.DataFrame:
    """Index de `vaisseaux.sources_troubles` ; les chemins d'images y sont relatifs à son dossier,
    ce qui permet de déplacer le tout (par exemple vers un jeu de données Kaggle)."""
    index = pd.read_csv(chemin, low_memory=False)
    base = Path(chemin).parent
    index["cle"] = index["chemin"]  # chemin tel qu'écrit dans l'index : identifiant stable de l'image
    index["chemin"] = [c if Path(c).is_absolute() else str(base / c) for c in index["chemin"]]
    return index


def exporter_depuis_poids(poids: Path, index: Path, onnx: Path, architecture: str, sortie: Path) -> float:
    """Export ONNX à partir des poids entraînés ailleurs (Kaggle) ; parité vérifiée sur des images
    réelles du test externe et reportée dans resultats.json."""
    import torch

    modele = _modele(architecture, pretrained=False)
    modele.load_state_dict(torch.load(poids, map_location="cpu", weights_only=True))
    externe = lire_index(index).query("source == 'jsiec'")
    exemples = np.stack([transformations(False)(Image.open(c).convert("RGB")).numpy() for c in externe["chemin"][:8]])
    ecart = exporter(modele, onnx, exemples)
    chemin_resultats = sortie / "resultats.json"
    if chemin_resultats.exists():
        resultats = json.loads(chemin_resultats.read_text())
        resultats["parite_onnx"] = ecart
        chemin_resultats.write_text(json.dumps(resultats, indent=1, ensure_ascii=False))
    return ecart


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--index", type=Path, default=Path("data/troubles/index.csv"))
    parser.add_argument("--sortie", type=Path, default=Path("resultats/troubles"))
    parser.add_argument("--modele", type=Path, default=Path("modeles/troubles.onnx"))
    parser.add_argument("--architecture", default=ARCHITECTURE)
    parser.add_argument("--epoques", type=int, default=12)
    parser.add_argument("--images-par-epoque", type=int, default=12000)
    parser.add_argument("--professeur", type=Path, help="avis du professeur RETFound (professeur.csv)")
    parser.add_argument("--poids-initiaux", type=Path, help="poids ImageNet (safetensors) si pas d'accès internet")
    parser.add_argument("--sans-onnx", action="store_true", help="n'exporte pas en ONNX (fait ensuite avec --exporter)")
    parser.add_argument("--exporter", type=Path, metavar="POIDS.pt", help="exporte seulement ces poids en ONNX")
    parser.add_argument(
        "--recalculer", action="store_true", help="refait calibrage, seuils et résultats depuis les prédictions"
    )
    parser.add_argument(
        "--gpu-max",
        type=float,
        default=0.55,
        help="part maximale du temps où le GPU calcule (0,55 = 55 %% ; 1 = sans limite)",
    )
    args = parser.parse_args(argv)
    global POIDS_INITIAUX
    POIDS_INITIAUX = args.poids_initiaux
    args.modele.parent.mkdir(parents=True, exist_ok=True)
    if args.recalculer:
        recalculer(args.sortie)
        return 0
    if args.exporter:
        ecart = exporter_depuis_poids(args.exporter, args.index, args.modele, args.architecture, args.sortie)
        print(f"exporté : {args.modele} ; parité ONNX (images réelles) : {ecart:.2e}")
        return 0
    verrou = _verrouiller(args.modele.parent / ".entrainement_troubles.verrou")
    if verrou is None:
        print("un entraînement tourne déjà (voir : pgrep -af vaisseaux.troubles) : arrêt", file=sys.stderr)
        return 1
    debut = time.perf_counter()
    args.sortie.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(GRAINE)

    index = lire_index(args.index)
    jeu = index[index["source"] != "jsiec"].reset_index(drop=True)
    externe = index[index["source"] == "jsiec"].reset_index(drop=True)
    jeu["pli"] = _plis(jeu)
    if args.professeur:
        avis = pd.read_csv(args.professeur).set_index("cle")["logit_professeur"]
        jeu["prof_glaucome"] = 1 / (1 + np.exp(-jeu["cle"].map(avis).to_numpy(dtype=np.float64)))
        print(f"avis du professeur : {int(jeu['prof_glaucome'].notna().sum())} images", flush=True)
    print(
        f"entraînement : {len(jeu)} images ({', '.join(f'{s} {n}' for s, n in jeu['source'].value_counts().items())}) ; "
        f"test externe JSIEC : {len(externe)} images\natteints : "
        + ", ".join(f"{c} {int((jeu[c] == 1).sum())}" for c in CLES),
        flush=True,
    )
    # une seule barre pour les 5 plis et le modèle final, avec le temps restant estimé
    progression = tqdm(
        total=(PLIS + 1) * args.epoques * (args.images_par_epoque // 32),
        unit="lot",
        mininterval=10,
        ncols=150,
        file=sys.stdout,
        bar_format="{desc} {percentage:3.0f}% |{bar:30}| {elapsed} écoulé, reste ~{remaining} {postfix}",
    )
    z = np.zeros((len(jeu), len(CLES)), dtype=np.float32)
    for k in range(PLIS):
        dans = jeu["pli"].to_numpy() == k
        modele = entrainer(
            jeu[~dans],
            args.epoques,
            args.images_par_epoque,
            args.architecture,
            progression,
            f"validation croisée, pli {k + 1}/{PLIS}",
            args.gpu_max,
        )
        z[dans] = logits(modele, jeu[dans])
        test = jeu[dans]
        progression.write(
            f"pli {k + 1}/{PLIS} terminé, AUROC : "
            + ", ".join(f"{c} {_auroc(test[c].to_numpy(), z[dans, j]):.2f}" for j, c in enumerate(CLES))
        )

    par_source = {
        s: {c: _auroc(jeu.loc[jeu["source"] == s, c].to_numpy(), z[jeu["source"] == s, j]) for j, c in enumerate(CLES)}
        for s in jeu["source"].unique()
    }
    final = entrainer(
        jeu,
        args.epoques,
        args.images_par_epoque,
        args.architecture,
        progression,
        "modèle final (toutes les images)",
        args.gpu_max,
    )
    progression.close()
    print("évaluation sur le test externe JSIEC et export ONNX…", flush=True)
    torch.save(final.state_dict(), args.modele.with_suffix(".pt"))
    z_externe = logits(final, externe)
    z_gros_plan = logits(final, externe, gros_plan=GROS_PLAN_TEST)
    exemples = np.stack([transformations(False)(Image.open(c).convert("RGB")).numpy() for c in externe["chemin"][:8]])
    ecart = None if args.sans_onnx else exporter(final, args.modele, exemples)
    infos = {
        "architecture": args.architecture,
        "auroc_par_source": par_source,
        "parite_onnx": ecart,
        "duree_s": round(time.perf_counter() - debut, 1),
    }
    ecrire_resultats(
        args.sortie, jeu.drop(columns=["prof_glaucome"], errors="ignore"), z, externe, z_externe, infos, z_gros_plan
    )
    return 0


def ecrire_resultats(
    sortie: Path,
    jeu: pd.DataFrame,
    z: np.ndarray,
    externe: pd.DataFrame,
    z_externe: np.ndarray,
    infos: dict,
    z_gros_plan: np.ndarray | None = None,
) -> None:
    """Calibrage, évaluations, seuils et fichiers de résultats à partir des prédictions hors pli
    (jeu) et du modèle final (externe). Utilisée après l'entraînement et par --recalculer."""
    calibrage = {c: calibrer(z[:, j], jeu[c].to_numpy()) for j, c in enumerate(CLES)}
    resultat = evaluer(jeu, z, calibrage)
    seuils = {c: resultat[c]["seuil_possible"] for c in CLES}
    externe_resultat = evaluer_externe(externe, z_externe, calibrage, seuils)
    gros_plan = None if z_gros_plan is None else evaluer_externe(externe, z_gros_plan, calibrage, seuils)
    resultat_complet = {
        "architecture": infos.get("architecture", ARCHITECTURE),
        "troubles": resultat,
        "externe_jsiec": externe_resultat,
        "externe_jsiec_gros_plan": gros_plan,
        "auroc_par_source": infos.get("auroc_par_source", {}),
        "calibrage": calibrage,
        "parite_onnx": infos.get("parite_onnx"),
        "effectifs": {
            "images": int(len(jeu)),
            "groupes": int(jeu["groupe"].nunique()),
            "par_source": {s: int(n) for s, n in jeu["source"].value_counts().items()},
            "externe": int(len(externe)),
        },
        "duree_s": infos.get("duree_s"),
    }
    (sortie / "resultats.json").write_text(json.dumps(resultat_complet, indent=1, ensure_ascii=False))
    # fichier lu par la démo : ordre des sorties, noms, calibrage, seuils, qualité de détection
    noms_sources = {"odir": "ODIR-5K", "rfmid": "RFMiD", "smdg": "SMDG-19", "sjchoi86": "sjchoi86"}
    effectif_sources = jeu["source"].value_counts()
    demo = {
        "ordre": CLES,
        "origine": f"{len(jeu):,} photos de {len(effectif_sources)} bases publiques (".replace(",", "\u202f")
        + ", ".join(noms_sources.get(s, s) for s in effectif_sources.index)
        + ")",
        "seuil_probable": SEUIL_PROBABLE,
        "troubles": {
            c: {
                "nom": resultat[c]["nom"],
                "a": calibrage[c][0],
                "b": calibrage[c][1],
                "seuil_possible": resultat[c]["seuil_possible"],
                "auroc": resultat[c]["auroc"],
                "ic_bas": resultat[c]["ic_bas"],
                "ic_haut": resultat[c]["ic_haut"],
                "auroc_externe": externe_resultat[c].get("auroc"),
                "atteints_externe": externe_resultat[c].get("atteints"),
                "fausses_alertes": resultat[c]["fausses_alertes"],
                "fausses_alertes_externe": (
                    1 - externe_resultat[c]["specificite_peu_probable"] if "auroc" in externe_resultat[c] else None
                ),
                "detectes_des_possible": resultat[c]["detectes_des_possible"],
                "niveaux": resultat[c]["niveaux"],
            }
            for c in CLES
        },
    }
    (sortie / "troubles_demo.json").write_text(json.dumps(demo, indent=1, ensure_ascii=False))
    jeu.assign(**{f"logit_{c}": z[:, j] for j, c in enumerate(CLES)}).to_csv(sortie / "predictions.csv", index=False)
    colonnes = {f"logit_{c}": z_externe[:, j] for j, c in enumerate(CLES)}
    if z_gros_plan is not None:
        colonnes |= {f"logit_gros_plan_{c}": z_gros_plan[:, j] for j, c in enumerate(CLES)}
    externe.assign(**colonnes).to_csv(sortie / "predictions_externe.csv", index=False)
    figure(resultat, sortie / "auroc_par_trouble.png", externe_resultat)
    print("trouble        seuil « possible »   interne (IC 95 %)        externe JSIEC (IC 95 %)")
    for c in CLES:
        r, e = resultat[c], externe_resultat[c]
        texte_externe = (
            f"{e['auroc']:.3f} [{e['ic_bas']:.2f} ; {e['ic_haut']:.2f}] ({e['atteints']} atteints)"
            if "auroc" in e
            else "—"
        )
        print(
            f"{c:13s}  {100 * r['seuil_possible']:5.1f} %   {r['auroc']:.3f} [{r['ic_bas']:.2f} ; {r['ic_haut']:.2f}]"
            f"   {texte_externe}"
        )
    if gros_plan and "auroc" in gros_plan["glaucome"]:
        g = gros_plan["glaucome"]
        print(
            f"glaucome sur gros plans de la papille (JSIEC) : AUROC {g['auroc']:.3f} [{g['ic_bas']:.2f} ; {g['ic_haut']:.2f}],"
            f" {100 * g['sensibilite_possible']:.0f} % des cas signalés"
        )
    ecart = infos.get("parite_onnx")
    parite = "export ONNX à faire avec --exporter" if ecart is None else f"parité ONNX (images réelles) : {ecart:.2e}"
    print(f"{parite} ; durée {infos.get('duree_s')} s")


def recalculer(sortie: Path) -> None:
    """Refait calibrage, seuils et évaluations à partir des prédictions déjà enregistrées, sans
    réentraîner (par exemple après un entraînement fait sur Kaggle)."""
    anciens = json.loads((sortie / "resultats.json").read_text())
    jeu = pd.read_csv(sortie / "predictions.csv", low_memory=False)
    externe = pd.read_csv(sortie / "predictions_externe.csv", low_memory=False)
    z = jeu[[f"logit_{c}" for c in CLES]].to_numpy(dtype=np.float32)
    z_externe = externe[[f"logit_{c}" for c in CLES]].to_numpy(dtype=np.float32)
    colonnes_gp = [f"logit_gros_plan_{c}" for c in CLES]
    z_gros_plan = externe[colonnes_gp].to_numpy(dtype=np.float32) if colonnes_gp[0] in externe else None
    jeu = jeu.drop(columns=[f"logit_{c}" for c in CLES])
    externe = externe.drop(columns=[f"logit_{c}" for c in CLES] + (colonnes_gp if z_gros_plan is not None else []))
    ecrire_resultats(sortie, jeu, z, externe, z_externe, anciens, z_gros_plan)


if __name__ == "__main__":
    raise SystemExit(main())
