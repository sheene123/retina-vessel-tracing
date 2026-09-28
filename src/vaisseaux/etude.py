"""Étude : les métriques de segmentation prédisent-elles l'erreur des marqueurs vasculaires ?

Pour chaque image de test DRIVE, on segmente les vaisseaux avec 20 méthodes : des chaînes
classiques (Frangi), un U-Net à trois seuils, et des perturbations contrôlées de l'annotation
experte qui isolent chacune un type d'erreur. Pour chaque segmentation, on calcule
- les métriques de segmentation habituelles (face à l'annotation) ;
- l'erreur relative de 8 marqueurs vasculaires (face aux marqueurs de l'annotation) ;
- la qualité du tracé entre deux points (F1, avec le traceur du projet).
On mesure ensuite, image par image, si la méthode la mieux classée par une métrique est aussi
celle qui mesure le mieux chaque marqueur (corrélation de rang de Spearman), avec un intervalle
de confiance par bootstrap sur les images.

    python -m vaisseaux.unet                      # une fois : entraîne le U-Net
    python -m vaisseaux.etude --sortie resultats/etude_metriques
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage.filters import gaussian
from skimage.measure import euler_number
from skimage.morphology import disk, skeletonize

from .benchmark import CONFIGURATIONS
from .biomarqueurs import MARQUEURS, mesurer
from .donnees import charger, lister
from .evaluation import cl_dice, echantillonner_paires, evaluer_trace, metriques_binaires
from .graphe import carte_de_cout, plus_court_chemin
from .pretraitement import pretraiter

# seuils de binarisation choisis sur les images d'entraînement (resultats/evaluation.json)
SEUILS_FRANGI = {
    "vert_brut": 0.06,
    "clahe": 0.08,
    "clahe_gaussien": 0.06,
    "clahe_median": 0.06,
    "clahe_bilateral": 0.02,
    "clahe_nl_means": 0.08,
}
SEUILS_UNET = (0.3, 0.5, 0.7)
METRIQUES = ("exactitude", "sensibilite", "specificite", "dice", "mcc", "cl_dice", "erreur_b0", "erreur_b1", "hd95")
PLUS_EST_MIEUX = {"exactitude", "sensibilite", "specificite", "dice", "mcc", "cl_dice"}
CIBLES = (*MARQUEURS, "trace")
FAMILLES = ("Frangi", "U-Net", "Humain", "Perturbation")

# ------------------------------------------------------------------ segmentations


def perturbations(verite: np.ndarray, masque: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Erreurs contrôlées appliquées à l'annotation experte, chacune d'un seul type."""
    squelette = skeletonize(verite)
    points = np.argwhere(squelette)
    sortie = {
        "amincissement": ndi.binary_erosion(verite) | squelette,
        "epaississement_1": ndi.binary_dilation(verite),
        "epaississement_2": ndi.binary_dilation(verite, iterations=2),
        "petits_supprimes_1": ndi.binary_opening(verite),
        "petits_supprimes_2": ndi.binary_opening(verite, structure=disk(2)),
    }
    for n in (20, 60):
        coupe = verite.copy()
        for li, co in points[rng.choice(len(points), n, replace=False)]:
            coupe[max(0, li - 2) : li + 3, max(0, co - 2) : co + 3] = False
        sortie[f"coupures_{n}"] = coupe
    fond = np.argwhere(masque & ~ndi.binary_dilation(verite, iterations=3))
    for pourcent in (5, 15):
        ajout = np.zeros_like(verite)
        cible = pourcent / 100 * verite.sum()
        while ajout.sum() < cible:
            li, co = fond[rng.integers(len(fond))]
            r = int(rng.integers(1, 3))
            ajout[max(0, li - r) : li + r + 1, max(0, co - r) : co + r + 1] = True
        sortie[f"faux_positifs_{pourcent}"] = verite | ajout
    interieur = verite & ~ndi.binary_erosion(verite)
    exterieur = ndi.binary_dilation(verite) & ~verite
    bruite = verite.copy()
    bruite[interieur & (rng.random(verite.shape) < 0.3)] = False
    bruite[exterieur & (rng.random(verite.shape) < 0.3)] = True
    sortie["contours_bruites"] = bruite
    return {nom: b & masque for nom, b in sortie.items()}


# ------------------------------------------------------------------ mesures


def _betti(binaire: np.ndarray) -> tuple[int, int]:
    b0 = ndi.label(binaire, structure=np.ones((3, 3)))[1]
    return b0, b0 - int(euler_number(binaire, connectivity=2))


def _hd95(a: np.ndarray, b: np.ndarray) -> float:
    bord_a, bord_b = a & ~ndi.binary_erosion(a), b & ~ndi.binary_erosion(b)
    if not bord_a.any() or not bord_b.any():
        return float("nan")
    d_ab = ndi.distance_transform_edt(~bord_b)[bord_a]
    d_ba = ndi.distance_transform_edt(~bord_a)[bord_b]
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))


def metriques(pred: np.ndarray, verite: np.ndarray, masque: np.ndarray) -> dict[str, float]:
    m = metriques_binaires(pred, verite, masque)
    (b0p, b1p), (b0v, b1v) = _betti(pred & masque), _betti(verite & masque)
    return {
        "exactitude": m["exactitude"],
        "sensibilite": m["sensibilite"],
        "specificite": m["specificite"],
        "dice": m["dice"],
        "mcc": m["mcc"],
        "cl_dice": cl_dice(pred & masque, verite & masque),
        "erreur_b0": float(abs(b0p - b0v)),
        "erreur_b1": float(abs(b1p - b1v)),
        "hd95": _hd95(pred & masque, verite & masque),
    }


def qualite_trace(pred: np.ndarray, verite: np.ndarray, masque: np.ndarray, paires) -> float:
    """F1 moyen du tracé quand le coût est tiré de cette segmentation."""
    v = gaussian(pred.astype(float), 1.0)
    cout = carte_de_cout(v / max(v.max(), 1e-9), masque, alpha=1.0)
    scores = []
    for p in paires:
        try:
            chemin = plus_court_chemin(cout, p.depart, p.arrivee, a_etoile=True)
            scores.append(evaluer_trace(chemin.pixels, p.reference, verite)["f1"])
        except ValueError:
            scores.append(0.0)
    return float(np.mean(scores))


# ------------------------------------------------------------------ une image


def _image(tache: tuple) -> list[dict]:
    racine, ident, probas_unet, n_paires, graine = tache
    im = charger(racine, "test", ident)
    rng = np.random.default_rng([graine, int(ident)])
    paires = echantillonner_paires(im.verite, im.masque, n_paires, np.random.default_rng([0, int(ident)]))
    reference = mesurer(im.verite, im.masque)
    methodes: list[tuple[str, str, np.ndarray]] = []
    for nom, seuil in SEUILS_FRANGI.items():
        carte = pretraiter(im.rgb, im.masque, CONFIGURATIONS[nom]).vaisseaux
        methodes.append((f"frangi_{nom}", "Frangi", (carte >= seuil) & im.masque))
    for seuil in SEUILS_UNET:
        methodes.append((f"unet_p{int(100 * seuil)}", "U-Net", (probas_unet >= seuil) & im.masque))
    if im.verite2 is not None:  # second expert : référence humaine (archive officielle DRIVE)
        methodes.append(("second_expert", "Humain", im.verite2 & im.masque))
    for nom, binaire in perturbations(im.verite, im.masque, rng).items():
        methodes.append((nom, "Perturbation", binaire))
    lignes = []
    for nom, famille, binaire in methodes:
        biomarqueurs = mesurer(binaire, im.masque)
        # erreur relative, sauf pour le nombre de fragments (souvent 1 à 3 dans la vérité) : différence absolue
        erreurs = {
            k: abs(biomarqueurs[k] - reference[k]) / max(abs(reference[k]), 1e-9) for k in MARQUEURS if k != "fragments"
        }
        erreurs["fragments"] = abs(biomarqueurs["fragments"] - reference["fragments"])
        erreurs["trace"] = 1.0 - qualite_trace(binaire, im.verite, im.masque, paires)
        lignes.append(
            {
                "image": ident,
                "methode": nom,
                "famille": famille,
                "metriques": metriques(binaire, im.verite, im.masque),
                "marqueurs": biomarqueurs,
                "reference": reference,
                "erreurs": erreurs,
            }
        )
    return lignes


# ------------------------------------------------------------------ analyse


def _qualite(ligne: dict, metrique: str) -> float:
    v = ligne["metriques"][metrique]
    return v if metrique in PLUS_EST_MIEUX else -v


def _pouvoir_intra_image(lignes: list[dict], metrique: str, cible: str) -> np.ndarray:
    """Pour chaque image : corrélation de rang entre la qualité selon la métrique et la
    justesse du marqueur (erreur changée de signe) sur l'ensemble des méthodes."""
    valeurs = []
    for ident in sorted({lg["image"] for lg in lignes}):
        sous = [lg for lg in lignes if lg["image"] == ident]
        q = np.array([_qualite(lg, metrique) for lg in sous])
        e = np.array([lg["erreurs"][cible] for lg in sous])
        ok = np.isfinite(q) & np.isfinite(e)
        if ok.sum() >= 4 and np.ptp(q[ok]) > 0 and np.ptp(e[ok]) > 0:
            valeurs.append(-spearmanr(q[ok], e[ok]).statistic)
    return np.array(valeurs)


def _bootstrap(valeurs: np.ndarray, n: int = 2000, graine: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(graine)
    moyennes = np.array([rng.choice(valeurs, len(valeurs)).mean() for _ in range(n)])
    return float(valeurs.mean()), float(np.quantile(moyennes, 0.025)), float(np.quantile(moyennes, 0.975))


def _pouvoir_selection(lignes: list[dict], metrique: str, cible: str, n: int = 1000, graine: int = 0) -> dict:
    """Sélection de modèle : les méthodes sont classées sur leur moyenne sur les images, selon la
    métrique puis selon la justesse du marqueur. Intervalle par bootstrap sur les images."""
    methodes = list(dict.fromkeys(lg["methode"] for lg in lignes))
    images = sorted({lg["image"] for lg in lignes})
    index = {(lg["image"], lg["methode"]): lg for lg in lignes}
    q = np.array([[_qualite(index[i, m], metrique) for m in methodes] for i in images])
    e = np.array([[index[i, m]["erreurs"][cible] for m in methodes] for i in images])

    def rho(selection: np.ndarray) -> float:
        qm, em = np.nanmean(q[selection], axis=0), np.nanmean(e[selection], axis=0)
        return float(-spearmanr(qm, em).statistic) if np.ptp(qm) > 0 and np.ptp(em) > 0 else float("nan")

    rng = np.random.default_rng(graine)
    tirages = np.array([rho(rng.choice(len(images), len(images))) for _ in range(n)])
    return {
        "moyenne": rho(np.arange(len(images))),
        "ic_bas": float(np.nanquantile(tirages, 0.025)),
        "ic_haut": float(np.nanquantile(tirages, 0.975)),
    }


def verdict(correlation: float) -> str:
    """Une mesure sert à comparer des patients si elle les classe comme l'expert."""
    if correlation >= 0.8:
        return "fiable"
    return "approximative" if correlation >= 0.5 else "peu fiable"


def fiabilite(lignes: list[dict], n: int = 2000, graine: int = 0) -> dict:
    """Pour chaque méthode et chaque marqueur, sur les patients :
    - corrélation de rang entre la mesure et la valeur de l'expert (la méthode classe-t-elle les
      patients comme l'expert ?), avec intervalle par bootstrap sur les patients ;
    - biais (erreur constante) et dispersion de l'erreur, rapportés à l'écart-type entre patients.
    Un biais n'empêche pas de comparer des patients entre eux ; une dispersion de l'ordre de
    l'écart entre patients, si."""
    rng = np.random.default_rng(graine)
    sortie = {}
    for methode in dict.fromkeys(lg["methode"] for lg in lignes):
        sous = [lg for lg in lignes if lg["methode"] == methode]
        sortie[methode] = {}
        for k in MARQUEURS:
            reference = np.array([lg["reference"][k] for lg in sous], dtype=float)
            mesure = np.array([lg["marqueurs"][k] for lg in sous], dtype=float)
            ok = np.isfinite(reference) & np.isfinite(mesure)
            reference, mesure = reference[ok], mesure[ok]
            ecart = float(reference.std(ddof=1))
            difference = mesure - reference

            def rho(index: np.ndarray, r=reference, m=mesure) -> float:
                if np.ptp(r[index]) == 0 or np.ptp(m[index]) == 0:
                    return float("nan")
                return float(spearmanr(r[index], m[index]).statistic)

            correlation = rho(np.arange(len(reference)))
            tirages = np.array([rho(rng.choice(len(reference), len(reference))) for _ in range(n)])
            sortie[methode][k] = {
                "correlation_patients": correlation,
                "ic_bas": float(np.nanquantile(tirages, 0.025)),
                "ic_haut": float(np.nanquantile(tirages, 0.975)),
                "biais_normalise": float(difference.mean() / ecart) if ecart > 0 else float("nan"),
                "dispersion_normalisee": float(difference.std(ddof=1) / ecart) if ecart > 0 else float("nan"),
                "ecart_type_patients": ecart,
                "verdict": verdict(correlation),
            }
    return sortie


def analyser(lignes: list[dict]) -> dict:
    sous_ensembles = {
        "toutes": lignes,
        "segmenteurs_reels": [lg for lg in lignes if lg["famille"] in ("Frangi", "U-Net")],
    }
    resultat: dict = {"pouvoir_predictif": {}, "pouvoir_selection": {}, "par_methode": {}}
    for nom, sous in sous_ensembles.items():
        resultat["pouvoir_selection"][nom] = {m: {c: _pouvoir_selection(sous, m, c) for c in CIBLES} for m in METRIQUES}
        tableau = {}
        for m in METRIQUES:
            tableau[m] = {}
            for c in CIBLES:
                v = _pouvoir_intra_image(sous, m, c)
                tableau[m][c] = (
                    dict(zip(("moyenne", "ic_bas", "ic_haut"), _bootstrap(v), strict=True)) if v.size else None
                )
        resultat["pouvoir_predictif"][nom] = tableau
    ordre = list(dict.fromkeys(lg["methode"] for lg in lignes))
    for methode in ordre:
        sous = [lg for lg in lignes if lg["methode"] == methode]
        resultat["par_methode"][methode] = {
            "famille": sous[0]["famille"],
            "metriques": {m: float(np.nanmean([lg["metriques"][m] for lg in sous])) for m in METRIQUES},
            "erreurs": {c: float(np.nanmean([lg["erreurs"][c] for lg in sous])) for c in CIBLES},
        }
    if all("reference" in lg for lg in lignes):
        resultat["fiabilite"] = fiabilite(lignes)
    return resultat


# ------------------------------------------------------------------ rapport et figures

NOMS_METRIQUES = {
    "exactitude": "Exactitude",
    "sensibilite": "Sensibilité",
    "specificite": "Spécificité",
    "dice": "Dice",
    "mcc": "MCC",
    "cl_dice": "clDice",
    "erreur_b0": "Erreur β0",
    "erreur_b1": "Erreur β1",
    "hd95": "HD95",
}
NOMS_CIBLES = {
    "densite": "Densité",
    "densite_longueur": "Densité de longueur",
    "dimension_fractale": "Dim. fractale",
    "calibre_moyen": "Calibre moyen",
    "calibre_principal": "Calibre gros vaiss.",
    "tortuosite": "Tortuosité",
    "bifurcations": "Bifurcations",
    "fragments": "Fragmentation",
    "trace": "Tracé",
    "tortuosite_ponderee": "Tortuosité pondérée",
    "densite_longueur_principale": "Dens. longueur princ.",
    "dimension_fractale_principale": "Dim. fractale princ.",
}


def rapport(resultat: dict, infos: dict) -> str:
    lignes = [
        "# Les métriques de segmentation prédisent-elles l'erreur des marqueurs vasculaires ?",
        "",
        f"{infos['images']} images de test DRIVE, {infos['methodes']} méthodes de segmentation, "
        f"{infos['segmentations']} segmentations. Pouvoir prédictif : corrélation de rang (Spearman), calculée image par "
        "image, entre la qualité selon la métrique et la justesse du marqueur ; moyenne sur les images [IC 95 % bootstrap]. "
        "1 = la métrique classe les méthodes exactement comme la justesse du marqueur, 0 = aucune information, "
        "valeur négative = la métrique induit en erreur.",
        "",
    ]
    for nom, titre in (
        ("toutes", "Toutes les méthodes (réelles et perturbations contrôlées)"),
        ("segmenteurs_reels", "Segmenteurs réels seulement (Frangi et U-Net)"),
    ):
        lignes += [
            f"## {titre}",
            "",
            "| Métrique | " + " | ".join(NOMS_CIBLES[c] for c in CIBLES) + " |",
            "|---" * (len(CIBLES) + 1) + "|",
        ]
        for m in METRIQUES:
            cellules = []
            for c in CIBLES:
                v = resultat["pouvoir_predictif"][nom][m][c]
                cellules.append(
                    "n/d" if v is None else f"{v['moyenne']:+.2f} [{v['ic_bas']:+.2f} ; {v['ic_haut']:+.2f}]"
                )
            lignes.append(f"| {NOMS_METRIQUES[m]} | " + " | ".join(cellules) + " |")
        lignes.append("")
    lignes += [
        "## Sélection de modèle (segmenteurs réels, classement sur les moyennes des 20 images)",
        "",
        "Question : le modèle qui a la meilleure métrique moyenne est-il aussi celui dont le marqueur est le "
        "plus juste en moyenne ? Corrélation de rang entre les 9 méthodes [IC 95 % bootstrap sur les images].",
        "",
        "| Métrique | " + " | ".join(NOMS_CIBLES[c] for c in CIBLES) + " |",
        "|---" * (len(CIBLES) + 1) + "|",
    ]
    for m in METRIQUES:
        v = resultat["pouvoir_selection"]["segmenteurs_reels"][m]
        lignes.append(
            f"| {NOMS_METRIQUES[m]} | "
            + " | ".join(f"{v[c]['moyenne']:+.2f} [{v[c]['ic_bas']:+.2f} ; {v[c]['ic_haut']:+.2f}]" for c in CIBLES)
            + " |"
        )
    lignes += [
        "",
        "## Moyennes par méthode",
        "",
        "| Méthode | Famille | Dice | clDice | "
        + " | ".join("écart fragments (nb)" if c == "fragments" else f"err. {NOMS_CIBLES[c]}" for c in CIBLES)
        + " |",
        "|---" * (len(CIBLES) + 4) + "|",
    ]
    for methode, v in resultat["par_methode"].items():
        lignes.append(
            f"| {methode} | {v['famille']} | {v['metriques']['dice']:.3f} | {v['metriques']['cl_dice']:.3f} | "
            + " | ".join(
                f"{v['erreurs'][c]:.1f}" if c == "fragments" else f"{100 * v['erreurs'][c]:.1f} %" for c in CIBLES
            )
            + " |"
        )
    if "fiabilite" in resultat:
        f = resultat["fiabilite"]
        reelles = [m for m, v in resultat["par_methode"].items() if v["famille"] != "Perturbation"]
        lignes += [
            "",
            "## Les mesures permettent-elles de comparer des patients ?",
            "",
            "Pour chaque marqueur, corrélation de rang entre la mesure d'une méthode et celle de l'expert sur les "
            "20 patients [IC 95 % bootstrap sur les patients] : la méthode classe-t-elle les patients comme l'expert ? "
            "Biais : erreur constante, en écarts-types entre patients (il fausse les valeurs absolues, pas le "
            "classement). Dispersion : part aléatoire de l'erreur, dans la même unité (au-dessus de 1, elle dépasse "
            "les différences entre patients).",
            "",
            "| Marqueur | Second expert : corrélation (plafond humain) | U-Net 0,5 : corrélation | biais | dispersion "
            "| Frangi NL-means : corrélation | Meilleure méthode réelle |",
            "|---|---|---|---|---|---|---|",
        ]
        for k in MARQUEURS:
            u, fr_ = f["unet_p50"][k], f["frangi_clahe_nl_means"][k]
            meilleure = max(reelles, key=lambda m: np.nan_to_num(f[m][k]["correlation_patients"], nan=-2))
            h = f.get("second_expert", {}).get(k)
            humain = (
                "n/d" if h is None else f"{h['correlation_patients']:+.2f} [{h['ic_bas']:+.2f} ; {h['ic_haut']:+.2f}]"
            )
            lignes.append(
                f"| {NOMS_CIBLES[k]} | {humain} | {u['correlation_patients']:+.2f} [{u['ic_bas']:+.2f} ; {u['ic_haut']:+.2f}] | "
                f"{u['biais_normalise']:+.2f} | {u['dispersion_normalisee']:.2f} | {fr_['correlation_patients']:+.2f} | "
                f"{meilleure} : {f[meilleure][k]['correlation_patients']:+.2f} |"
            )
    lignes += ["", f"Durée : {infos['duree_s']} s."]
    return "\n".join(lignes) + "\n"


def figures(resultat: dict, dossier: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for nom, titre in (("toutes", "toutes les méthodes"), ("segmenteurs_reels", "segmenteurs réels (Frangi, U-Net)")):
        table = resultat["pouvoir_predictif"][nom]
        valeurs = np.array(
            [[np.nan if table[m][c] is None else table[m][c]["moyenne"] for c in CIBLES] for m in METRIQUES]
        )
        fig, ax = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
        image = ax.imshow(valeurs, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(CIBLES)), [NOMS_CIBLES[c] for c in CIBLES], rotation=35, ha="right")
        ax.set_yticks(range(len(METRIQUES)), [NOMS_METRIQUES[m] for m in METRIQUES])
        for i in range(len(METRIQUES)):
            for j in range(len(CIBLES)):
                if np.isfinite(valeurs[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{valeurs[i, j]:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="white" if abs(valeurs[i, j]) > 0.55 else "#222222",
                    )
        ax.set_title(f"Pouvoir prédictif des métriques, {titre}", fontsize=11, color="#333333", loc="left")
        ax.set_xlabel("Marqueur mesuré sur la segmentation", color="#555555")
        ax.set_ylabel("Métrique de segmentation", color="#555555")
        for bord in ax.spines.values():
            bord.set_visible(False)
        barre = fig.colorbar(image, ax=ax, shrink=0.8)
        barre.set_label("corrélation de rang (1 = prédit parfaitement)")
        fig.savefig(dossier / f"pouvoir_predictif_{nom}.png", dpi=130)
        plt.close(fig)

    couleurs = {"Frangi": "#0072B2", "U-Net": "#E69F00", "Humain": "#CC79A7", "Perturbation": "#009E73"}
    formes = {"Frangi": "o", "U-Net": "s", "Humain": "D", "Perturbation": "^"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, cible in zip(axes, ("calibre_moyen", "tortuosite", "dimension_fractale"), strict=True):
        for famille in FAMILLES:
            points = [
                (v["metriques"]["dice"], 100 * v["erreurs"][cible], n)
                for n, v in resultat["par_methode"].items()
                if v["famille"] == famille
            ]
            if points:
                x, y, _ = zip(*points, strict=True)
                ax.scatter(
                    x,
                    y,
                    s=46,
                    c=couleurs[famille],
                    marker=formes[famille],
                    label=famille,
                    edgecolors="white",
                    linewidths=0.8,
                )
        ax.set_xlabel("Dice moyen", color="#555555")
        ax.set_ylabel(f"Erreur relative moyenne, {NOMS_CIBLES[cible].lower()} (%)", color="#555555")
        ax.grid(alpha=0.25)
        for bord in ("top", "right"):
            ax.spines[bord].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Un meilleur Dice ne garantit pas un marqueur plus juste (une marque par méthode)", fontsize=11, color="#333333"
    )
    fig.savefig(dossier / "dice_contre_erreurs.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ programme


def lancer(racine: Path, sortie: Path, modele_unet: Path, n_paires: int, travailleurs: int, graine: int = 0) -> dict:
    from .unet import charger_modele, predire

    debut = time.perf_counter()
    ids = lister(racine, "test")
    modele = charger_modele(modele_unet)
    taches = []
    for ident in ids:
        im = charger(racine, "test", ident)
        taches.append((racine, ident, predire(modele, im.rgb, im.masque), n_paires, graine))
    del modele
    with ProcessPoolExecutor(travailleurs) as pool:
        lignes = [lg for bloc in pool.map(_image, taches) for lg in bloc]
    resultat = analyser(lignes)
    infos = {
        "images": len(ids),
        "methodes": len({lg["methode"] for lg in lignes}),
        "segmentations": len(lignes),
        "duree_s": round(time.perf_counter() - debut, 1),
    }
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "etude.json").write_text(
        json.dumps({"infos": infos, **resultat, "lignes": lignes}, indent=1, ensure_ascii=False)
    )
    (sortie / "etude.md").write_text(rapport(resultat, infos))
    _ecrire_fiabilite(resultat, sortie)
    figures(resultat, sortie)
    return {"infos": infos, **resultat}


def _ecrire_fiabilite(resultat: dict, sortie: Path) -> None:
    """Fiabilité des mesures du U-Net (seuil 0,5), lue par la démo web."""
    if "fiabilite" in resultat:
        fiab = resultat["fiabilite"]
        donnees = {}
        for k, v in fiab["unet_p50"].items():
            donnees[k] = dict(v)
            if "second_expert" in fiab:
                h = fiab["second_expert"][k]
                donnees[k]["accord_experts"] = {c: h[c] for c in ("correlation_patients", "ic_bas", "ic_haut")}
        (sortie / "fiabilite_unet.json").write_text(json.dumps(donnees, indent=1))


def reanalyser(sortie: Path) -> dict:
    """Recalcule l'analyse, le rapport et les figures à partir des segmentations déjà mesurées."""
    donnees = json.loads((sortie / "etude.json").read_text())
    resultat = analyser(donnees["lignes"])
    complet = {"infos": donnees["infos"], **resultat, "lignes": donnees["lignes"]}
    (sortie / "etude.json").write_text(json.dumps(complet, indent=1, ensure_ascii=False))
    (sortie / "etude.md").write_text(rapport(resultat, donnees["infos"]))
    _ecrire_fiabilite(resultat, sortie)
    figures(resultat, sortie)
    return {"infos": donnees["infos"], **resultat}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--racine", type=Path, default=Path("data/DRIVE"))
    parser.add_argument("--sortie", type=Path, default=Path("resultats/etude_metriques"))
    parser.add_argument("--unet", type=Path, default=Path("modeles/unet_drive.pt"))
    parser.add_argument("--paires", type=int, default=10)
    parser.add_argument("--travailleurs", type=int, default=10)
    parser.add_argument("--reanalyser", action="store_true", help="refait seulement l'analyse depuis etude.json")
    args = parser.parse_args(argv)
    if args.reanalyser:
        resultat = reanalyser(args.sortie)
    else:
        resultat = lancer(args.racine, args.sortie, args.unet, args.paires, args.travailleurs)
    print((args.sortie / "etude.md").read_text())
    print(resultat["infos"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
