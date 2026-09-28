"""Marqueurs vasculaires calculés sur une carte binaire des vaisseaux.

Ce sont les mesures globales utilisées en oculomique (logiciels SIVA, VAMPIRE, AutoMorph) :
densité, complexité du réseau (dimension fractale), calibre, tortuosité, bifurcations,
fragmentation. Les longueurs sont en pixels : pour comparer des images de résolutions
différentes, il faut les normaliser (par exemple par le diamètre du disque optique).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

from .evaluation import _graphe_squelette

_HUIT = np.ones((3, 3), dtype=bool)
_VOISINS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
MARQUEURS = (
    "densite",
    "densite_longueur",
    "dimension_fractale",
    "calibre_moyen",
    "calibre_principal",
    "tortuosite",
    "bifurcations",
    "fragments",
    # variantes robustes, choisies parce que deux experts s'accordent mieux sur elles (docs/etude_metriques.md)
    "tortuosite_ponderee",
    "densite_longueur_principale",
    "dimension_fractale_principale",
)


# voisins dans l'ordre du tour : N, NE, E, SE, S, SO, O, NO
_TOUR = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _nb_voisins(squelette: np.ndarray) -> np.ndarray:
    return ndi.convolve(squelette.astype(np.uint8), _VOISINS, mode="constant") * squelette


def _transitions(squelette: np.ndarray) -> np.ndarray:
    """Nombre de passages 0 -> 1 en faisant le tour des 8 voisins (crossing number) :
    1 pour une extrémité, 2 pour un point courant (y compris un coin d'escalier), 3 ou plus
    pour une vraie bifurcation."""
    cadre = np.pad(squelette, 1)
    h, w = squelette.shape
    anneau = [cadre[1 + dl : 1 + dl + h, 1 + dc : 1 + dc + w] for dl, dc in _TOUR]
    total = sum((~anneau[k]) & anneau[(k + 1) % 8] for k in range(8))
    return total.astype(np.uint8) * squelette


def jonctions(squelette: np.ndarray) -> np.ndarray:
    """Pixels de bifurcation ou de croisement du squelette."""
    return squelette & (_transitions(squelette) >= 3)


def segments(squelette: np.ndarray) -> tuple[np.ndarray, int]:
    """Découpe le squelette en segments de vaisseau entre bifurcations et extrémités."""
    coupe = squelette & ~ndi.binary_dilation(jonctions(squelette), structure=_HUIT)
    return ndi.label(coupe, structure=_HUIT)


def dimension_fractale(binaire: np.ndarray) -> float:
    """Dimension de comptage de boîtes (box-counting), pente de log N(s) contre log(1/s)."""
    if binaire.sum() < 2:
        return float("nan")
    image = binaire
    tailles = 2 ** np.arange(1, max(3, int(np.log2(min(image.shape))) - 1))
    comptes = []
    for s in tailles:
        h, w = -(-image.shape[0] // s) * s, -(-image.shape[1] // s) * s
        cadre = np.zeros((h, w), dtype=bool)
        cadre[: image.shape[0], : image.shape[1]] = image
        comptes.append(cadre.reshape(h // s, s, w // s, s).any(axis=(1, 3)).sum())
    return float(np.polyfit(np.log(1.0 / tailles), np.log(comptes), 1)[0])


def _longueur_segment(masque: np.ndarray) -> tuple[float, float] | None:
    """(longueur de l'arc, corde) d'un segment à deux extrémités, None sinon (boucle, amas)."""
    extremites = np.argwhere(_transitions(masque) == 1)
    if len(extremites) != 2:
        return None
    points, graphe = _graphe_squelette(masque)
    index = {tuple(p): i for i, p in enumerate(points)}
    distance = dijkstra(graphe, indices=index[tuple(extremites[0])])[index[tuple(extremites[1])]]
    return float(distance), float(np.linalg.norm(extremites[0] - extremites[1]))


def tortuosite(squelette: np.ndarray, longueur_min: int = 20) -> float:
    """Moyenne, sur les segments d'au moins `longueur_min` pixels, du rapport arc / corde."""
    etiquettes, _ = segments(squelette)
    valeurs = []
    for k, fenetre in enumerate(ndi.find_objects(etiquettes), start=1):
        if fenetre is None:
            continue
        masque = np.pad(etiquettes[fenetre] == k, 1)
        if masque.sum() < longueur_min:
            continue
        mesure = _longueur_segment(masque)
        if mesure is not None and mesure[1] > 0:
            valeurs.append(mesure[0] / mesure[1])
    return float(np.mean(valeurs)) if valeurs else float("nan")


def tortuosite_ponderee(squelette: np.ndarray, longueur_min: int = 40) -> float:
    """Tortuosité des longs segments (au moins `longueur_min` pixels), pondérée par leur longueur :
    somme des arcs / somme des cordes. Les segments courts, où l'annotation varie le plus, sont
    ignorés et ne pèsent plus autant que les grands vaisseaux."""
    etiquettes, _ = segments(squelette)
    arcs, cordes = 0.0, 0.0
    for k, fenetre in enumerate(ndi.find_objects(etiquettes), start=1):
        if fenetre is None:
            continue
        masque = np.pad(etiquettes[fenetre] == k, 1)
        if masque.sum() < longueur_min:
            continue
        mesure = _longueur_segment(masque)
        if mesure is not None and mesure[1] > 0:
            arcs, cordes = arcs + mesure[0], cordes + mesure[1]
    return arcs / cordes if cordes > 0 else float("nan")


def reseau_principal(binaire: np.ndarray, taille_min: int = 50) -> np.ndarray:
    """Vaisseaux d'au moins 3 pixels de large, en morceaux d'au moins `taille_min` pixels :
    on retire les capillaires, sur lesquels les annotateurs divergent le plus."""
    principal = ndi.binary_opening(binaire, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool))
    etiquettes, _ = ndi.label(principal, structure=_HUIT)
    garder = np.bincount(etiquettes.ravel()) >= taille_min
    garder[0] = False
    return garder[etiquettes]


def calibres(binaire: np.ndarray, squelette: np.ndarray) -> np.ndarray:
    """Largeur locale (en pixels) en chaque point du squelette : 2 d - 1, d étant la distance au fond."""
    return 2 * ndi.distance_transform_edt(binaire)[squelette] - 1


def mesurer(binaire: np.ndarray, masque: np.ndarray) -> dict[str, float]:
    """Tous les marqueurs d'une carte binaire, restreinte au champ de vue."""
    binaire = binaire & masque
    squelette = skeletonize(binaire)
    surface = masque.sum()
    largeurs = calibres(binaire, squelette)
    etiquettes, _ = ndi.label(binaire, structure=_HUIT)
    tailles = np.bincount(etiquettes.ravel())[1:]
    _, n_jonctions = ndi.label(jonctions(squelette), structure=_HUIT)
    principal = reseau_principal(binaire)
    squelette_principal = skeletonize(principal)
    return {
        "densite": float(binaire.sum() / surface),
        "densite_longueur": float(squelette.sum() / surface),
        "dimension_fractale": dimension_fractale(binaire),
        "calibre_moyen": float(largeurs.mean()) if largeurs.size else float("nan"),
        "calibre_principal": float(np.mean(np.sort(largeurs)[-max(1, largeurs.size // 10) :]))
        if largeurs.size
        else float("nan"),
        "tortuosite": tortuosite(squelette),
        "bifurcations": float(1e4 * n_jonctions / surface),
        "fragments": float((tailles >= 10).sum()),
        "tortuosite_ponderee": tortuosite_ponderee(squelette),
        "densite_longueur_principale": float(squelette_principal.sum() / surface),
        "dimension_fractale_principale": dimension_fractale(principal),
    }
