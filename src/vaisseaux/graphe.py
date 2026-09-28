"""Graphe des pixels et plus court chemin.

Chaque pixel du champ de vue est un sommet relié à ses 8 voisins. L'arête (p, q) coûte
d(p, q) * (c(p) + c(q)) / 2, où d vaut 1 ou sqrt(2) et c est le coût de traversée d'un
pixel, faible dans les vaisseaux. Le plus court chemin entre deux points suit donc le
vaisseau qui les relie.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

RACINE2 = math.sqrt(2.0)
VOISINS_8 = (
    (-1, -1, RACINE2),
    (-1, 0, 1.0),
    (-1, 1, RACINE2),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, RACINE2),
    (1, 0, 1.0),
    (1, 1, RACINE2),
)


@dataclass
class Chemin:
    pixels: np.ndarray  # (n, 2) en (ligne, colonne), du départ à l'arrivée
    cout: float
    pixels_visites: int

    @property
    def longueur(self) -> float:
        """Longueur géométrique en pixels."""
        pas = np.abs(np.diff(self.pixels, axis=0))
        return float(np.sum(np.where(pas.sum(axis=1) == 2, RACINE2, 1.0)))


def carte_de_cout(
    vaisseaux: np.ndarray, masque: np.ndarray | None = None, alpha: float = 2.0, epsilon: float = 0.01
) -> np.ndarray:
    """c = 1 / (epsilon + v)^alpha : environ 1 dans un vaisseau, epsilon^-alpha dans le fond.
    Les pixels hors du champ de vue sont infranchissables (coût infini)."""
    cout = 1.0 / (epsilon + np.clip(vaisseaux, 0, 1)) ** alpha
    if masque is not None:
        cout = np.where(masque, cout, np.inf)
    return cout


def plus_court_chemin(
    cout: np.ndarray, depart: tuple[int, int], arrivee: tuple[int, int], a_etoile: bool = False
) -> Chemin:
    """Dijkstra avec arrêt anticipé, ou A* avec l'heuristique octile (admissible et
    cohérente car chaque pas coûte au moins d * min(c))."""
    hauteur, largeur = cout.shape
    for nom, (ligne, colonne) in (("départ", depart), ("arrivée", arrivee)):
        if not (0 <= ligne < hauteur and 0 <= colonne < largeur) or not np.isfinite(cout[ligne, colonne]):
            raise ValueError(f"le point de {nom} {(ligne, colonne)} est hors du champ de vue")

    c = cout.ravel().tolist()
    source, cible = depart[0] * largeur + depart[1], arrivee[0] * largeur + arrivee[1]
    cible_l, cible_c = arrivee
    c_min = float(np.min(cout[np.isfinite(cout)])) if a_etoile else 0.0

    def heuristique(ligne: int, colonne: int) -> float:
        dl, dc = abs(ligne - cible_l), abs(colonne - cible_c)
        return c_min * (dl + dc + (RACINE2 - 2.0) * min(dl, dc))

    distance = [math.inf] * len(c)
    parent = [-1] * len(c)
    fixe = bytearray(len(c))
    distance[source] = 0.0
    tas = [(heuristique(*depart), 0.0, source)]
    visites = 0
    while tas:
        _, g, u = heapq.heappop(tas)
        if fixe[u]:
            continue
        fixe[u] = 1
        visites += 1
        if u == cible:
            break
        ligne, colonne = divmod(u, largeur)
        cu = c[u]
        for dl, dc, d in VOISINS_8:
            vl, vc = ligne + dl, colonne + dc
            if 0 <= vl < hauteur and 0 <= vc < largeur:
                v = vl * largeur + vc
                cv = c[v]
                if fixe[v] or cv == math.inf:
                    continue
                nouveau = g + d * 0.5 * (cu + cv)
                if nouveau < distance[v]:
                    distance[v] = nouveau
                    parent[v] = u
                    heapq.heappush(tas, (nouveau + heuristique(vl, vc), nouveau, v))

    if distance[cible] == math.inf:
        raise ValueError("aucun chemin ne relie les deux points dans le champ de vue")
    trajet = [cible]
    while trajet[-1] != source:
        trajet.append(parent[trajet[-1]])
    pixels = np.array([divmod(p, largeur) for p in reversed(trajet)])
    return Chemin(pixels=pixels, cout=distance[cible], pixels_visites=visites)


def plus_court_chemin_scipy(cout: np.ndarray, depart: tuple[int, int], arrivee: tuple[int, int]) -> float:
    """Coût de référence calculé par scipy.sparse.csgraph (sert à valider l'implémentation)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    hauteur, largeur = cout.shape
    indices = np.arange(cout.size).reshape(cout.shape)
    lignes, colonnes, poids = [], [], []
    for dl, dc, d in VOISINS_8:
        l0, l1 = max(0, -dl), hauteur - max(0, dl)
        c0, c1 = max(0, -dc), largeur - max(0, dc)
        a = indices[l0:l1, c0:c1].ravel()
        b = indices[l0 + dl : l1 + dl, c0 + dc : c1 + dc].ravel()
        w = d * 0.5 * (cout.ravel()[a] + cout.ravel()[b])
        ok = np.isfinite(w)
        lignes.append(a[ok])
        colonnes.append(b[ok])
        poids.append(w[ok])
    graphe = coo_matrix(
        (np.concatenate(poids), (np.concatenate(lignes), np.concatenate(colonnes))), shape=(cout.size, cout.size)
    ).tocsr()
    distances = dijkstra(graphe, indices=depart[0] * largeur + depart[1])
    return float(distances[arrivee[0] * largeur + arrivee[1]])
