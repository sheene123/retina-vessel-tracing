"""Évaluation objective face à une vérité terrain annotée par des experts.

Un tracé est une ligne, l'annotation est une surface : on ne peut pas se contenter des
métriques de segmentation. Le module fournit trois familles de mesures.

1. Mesures de tracé, face à un chemin de référence tiré du squelette de l'annotation :
   précision et couverture avec tolérance, distances de Hausdorff et moyenne, distance
   de Fréchet (sensible à l'ordre), plus long écart hors vaisseau (raccourci).
2. Mesures pixel de la carte de rehaussement : AUC ROC, sensibilité, spécificité,
   Dice, MCC (dans le champ de vue uniquement) et clDice (topologie).
3. Outils statistiques : intervalle de confiance par bootstrap sur les images et test
   apparié de Wilcoxon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from skimage.morphology import skeletonize

from .graphe import RACINE2, VOISINS_8

# --------------------------------------------------------------------------- tracés


@dataclass
class PaireReference:
    depart: tuple[int, int]
    arrivee: tuple[int, int]
    reference: np.ndarray  # (n, 2) chemin géodésique sur le squelette de l'annotation
    longueur: float


def _graphe_squelette(squelette: np.ndarray) -> tuple[np.ndarray, coo_matrix]:
    points = np.argwhere(squelette)
    index = -np.ones(squelette.shape, dtype=np.int64)
    index[tuple(points.T)] = np.arange(len(points))
    lignes, colonnes, poids = [], [], []
    hauteur, largeur = squelette.shape
    for dl, dc, d in VOISINS_8:
        voisin = points + (dl, dc)
        ok = (voisin[:, 0] >= 0) & (voisin[:, 0] < hauteur) & (voisin[:, 1] >= 0) & (voisin[:, 1] < largeur)
        cible = np.full(len(points), -1)
        cible[ok] = index[voisin[ok, 0], voisin[ok, 1]]
        garde = cible >= 0
        lignes.append(np.flatnonzero(garde))
        colonnes.append(cible[garde])
        poids.append(np.full(garde.sum(), d))
    n = len(points)
    graphe = coo_matrix((np.concatenate(poids), (np.concatenate(lignes), np.concatenate(colonnes))), shape=(n, n))
    return points, graphe.tocsr()


def echantillonner_paires(
    verite: np.ndarray,
    masque: np.ndarray,
    n: int,
    rng: np.random.Generator,
    longueur_min: float = 40.0,
    longueur_max: float = 150.0,
    marge: int = 10,
) -> list[PaireReference]:
    """Tire n paires de points sur le squelette de l'annotation, reliées le long du squelette
    par un chemin de longueur géodésique dans [longueur_min, longueur_max]."""
    interieur = ndi.binary_erosion(masque, iterations=marge)
    points, graphe = _graphe_squelette(skeletonize(verite) & masque)
    candidats = np.flatnonzero(interieur[tuple(points.T)])
    paires: list[PaireReference] = []
    essais = 0
    while len(paires) < n and essais < 50 * n:
        essais += 1
        source = int(rng.choice(candidats))
        dist, pred = dijkstra(graphe, indices=source, limit=longueur_max, return_predecessors=True)
        possibles = candidats[(dist[candidats] >= longueur_min) & (dist[candidats] <= longueur_max)]
        if possibles.size == 0:
            continue
        cible = int(rng.choice(possibles))
        trajet = [cible]
        while trajet[-1] != source:
            trajet.append(int(pred[trajet[-1]]))
        reference = points[trajet[::-1]]
        paires.append(PaireReference(tuple(reference[0]), tuple(reference[-1]), reference, float(dist[cible])))
    return paires


def reference_entre(
    verite: np.ndarray, depart: tuple[int, int], arrivee: tuple[int, int], rayon: float = 6.0
) -> np.ndarray | None:
    """Chemin de référence entre deux points quelconques (par exemple cliqués) : chaque point
    est projeté sur le squelette de l'annotation s'il en est à moins de `rayon` pixels, puis on
    suit le squelette. Renvoie None si un point est loin de tout vaisseau annoté ou si les deux
    projections ne sont pas reliées."""
    points, graphe = _graphe_squelette(skeletonize(verite))
    if len(points) == 0:
        return None
    arbre = cKDTree(points)
    (d0, i0), (d1, i1) = arbre.query(depart), arbre.query(arrivee)
    if d0 > rayon or d1 > rayon:
        return None
    dist, pred = dijkstra(graphe, indices=int(i0), return_predecessors=True)
    if not np.isfinite(dist[i1]):
        return None
    trajet = [int(i1)]
    while trajet[-1] != i0:
        trajet.append(int(pred[trajet[-1]]))
    return points[trajet[::-1]]


def _longueur(chemin: np.ndarray) -> float:
    pas = np.abs(np.diff(chemin, axis=0)).sum(axis=1)
    return float(np.sum(np.where(pas == 2, RACINE2, pas)))


def frechet_discret(a: np.ndarray, b: np.ndarray) -> float:
    """Distance de Fréchet discrète (Eiter & Mannila, 1994) : la « laisse » la plus courte
    pour parcourir les deux courbes dans le même sens. Pénalise les allers-retours que la
    distance de Hausdorff ne voit pas."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    ca = np.empty_like(d)
    ca[0, 0] = d[0, 0]
    ca[0, 1:] = np.maximum.accumulate(d[0, 1:].clip(min=d[0, 0]))
    ca[1:, 0] = np.maximum.accumulate(d[1:, 0].clip(min=d[0, 0]))
    for i in range(1, len(a)):
        precedente, courante, di = ca[i - 1], ca[i], d[i]
        for j in range(1, len(b)):
            courante[j] = max(di[j], min(precedente[j], precedente[j - 1], courante[j - 1]))
    return float(ca[-1, -1])


def plus_long_ecart(chemin: np.ndarray, verite: np.ndarray, tolerance: float) -> float:
    """Plus longue portion continue du tracé à plus de `tolerance` pixels d'un vaisseau
    annoté : mesure directe des « raccourcis » à travers le fond."""
    distance = ndi.distance_transform_edt(~verite)[tuple(chemin.T)]
    dehors = distance > tolerance
    meilleur, debut = 0.0, None
    for i, hors in enumerate(np.append(dehors, False)):
        if hors and debut is None:
            debut = i
        elif not hors and debut is not None:
            meilleur = max(meilleur, _longueur(chemin[max(debut - 1, 0) : i + 1]))
            debut = None
    return meilleur


def evaluer_trace(
    trace: np.ndarray, reference: np.ndarray, verite: np.ndarray, tolerance: float = 2.0
) -> dict[str, float]:
    distance_vaisseau = ndi.distance_transform_edt(~verite)
    d_trace = cKDTree(reference).query(trace)[0]  # tracé -> référence
    d_ref = cKDTree(trace).query(reference)[0]  # référence -> tracé
    precision = float(np.mean(distance_vaisseau[tuple(trace.T)] <= tolerance))
    couverture = float(np.mean(d_ref <= tolerance))
    return {
        "precision": precision,
        "couverture": couverture,
        "f1": 2 * precision * couverture / (precision + couverture) if precision + couverture else 0.0,
        "distance_moyenne": float((d_trace.mean() + d_ref.mean()) / 2),
        "hausdorff": float(max(d_trace.max(), d_ref.max())),
        "hausdorff95": float(max(np.percentile(d_trace, 95), np.percentile(d_ref, 95))),
        "frechet": frechet_discret(trace, reference),
        "ratio_longueur": _longueur(trace) / max(_longueur(reference), 1e-9),
        "ecart_hors_vaisseau": plus_long_ecart(trace, verite, tolerance),
    }


# --------------------------------------------------------------------------- pixels


def auc_roc(scores: np.ndarray, verite: np.ndarray, masque: np.ndarray) -> float:
    """Aire sous la courbe ROC, calculée dans le champ de vue uniquement."""
    s, y = scores[masque].astype(float), verite[masque]
    rangs = np.empty(len(s))
    rangs[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return float((rangs[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metriques_binaires(prediction: np.ndarray, verite: np.ndarray, masque: np.ndarray) -> dict[str, float]:
    p, v = prediction[masque], verite[masque]
    vp, fp = float(np.sum(p & v)), float(np.sum(p & ~v))
    fn, vn = float(np.sum(~p & v)), float(np.sum(~p & ~v))
    denominateur = np.sqrt((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))
    return {
        "sensibilite": vp / (vp + fn) if vp + fn else 0.0,
        "specificite": vn / (vn + fp) if vn + fp else 0.0,
        "exactitude": (vp + vn) / len(p),
        "precision": vp / (vp + fp) if vp + fp else 0.0,
        "dice": 2 * vp / (2 * vp + fp + fn) if vp + fp + fn else 1.0,
        "mcc": (vp * vn - fp * fn) / denominateur if denominateur else 0.0,
    }


def cl_dice(prediction: np.ndarray, verite: np.ndarray) -> float:
    """clDice (Shit et al., CVPR 2021) : Dice entre squelettes et surfaces, sensible aux
    coupures et aux fusions de vaisseaux que le Dice classique ignore presque."""
    sp, sv = skeletonize(prediction), skeletonize(verite)
    t_prec = (sp & verite).sum() / max(sp.sum(), 1)
    t_sens = (sv & prediction).sum() / max(sv.sum(), 1)
    return float(2 * t_prec * t_sens / (t_prec + t_sens)) if t_prec + t_sens else 0.0


# --------------------------------------------------------------------------- statistiques


def intervalle_bootstrap(
    valeurs: np.ndarray, groupes: np.ndarray, n: int = 2000, niveau: float = 0.95, graine: int = 0
) -> tuple[float, float, float]:
    """Moyenne et intervalle de confiance par bootstrap *par image* : les paires d'une même
    image ne sont pas indépendantes, on rééchantillonne donc les images."""
    rng = np.random.default_rng(graine)
    ids = np.unique(groupes)
    par_image = {g: valeurs[groupes == g] for g in ids}
    moyennes = np.empty(n)
    for k in range(n):
        tirage = rng.choice(ids, size=len(ids), replace=True)
        moyennes[k] = np.mean(np.concatenate([par_image[g] for g in tirage]))
    alpha = (1 - niveau) / 2
    return float(np.mean(valeurs)), float(np.quantile(moyennes, alpha)), float(np.quantile(moyennes, 1 - alpha))


def comparer_apparie(a_par_image: np.ndarray, b_par_image: np.ndarray) -> dict[str, float]:
    """Test de Wilcoxon apparié sur les moyennes par image (l'image est l'unité statistique)."""
    difference = np.asarray(b_par_image) - np.asarray(a_par_image)
    if np.allclose(difference, 0):
        return {"difference_mediane": 0.0, "p_valeur": 1.0}
    return {"difference_mediane": float(np.median(difference)), "p_valeur": float(wilcoxon(difference).pvalue)}
