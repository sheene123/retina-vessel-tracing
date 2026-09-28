import numpy as np
import pytest

from vaisseaux.evaluation import evaluer_trace
from vaisseaux.graphe import carte_de_cout, plus_court_chemin, plus_court_chemin_scipy
from vaisseaux.pretraitement import pretraiter


@pytest.mark.parametrize("a_etoile", [False, True])
def test_cout_identique_a_scipy(a_etoile):
    rng = np.random.default_rng(1)
    cout = rng.uniform(1, 10, (25, 30))
    cout[5:20, 12] = np.inf  # mur partiel à contourner
    for _ in range(5):
        depart, arrivee = (
            (int(rng.integers(25)), int(rng.integers(12))),
            (int(rng.integers(25)), 13 + int(rng.integers(17))),
        )
        chemin = plus_court_chemin(cout, depart, arrivee, a_etoile=a_etoile)
        assert chemin.cout == pytest.approx(plus_court_chemin_scipy(cout, depart, arrivee))


def test_chemin_continu_et_bornes():
    cout = np.ones((20, 20))
    chemin = plus_court_chemin(cout, (0, 0), (19, 10))
    assert tuple(chemin.pixels[0]) == (0, 0) and tuple(chemin.pixels[-1]) == (19, 10)
    assert np.all(np.abs(np.diff(chemin.pixels, axis=0)).max(axis=1) == 1)
    assert chemin.longueur == pytest.approx(10 * np.sqrt(2) + 9)


def test_a_etoile_explore_moins():
    cout = np.ones((60, 60))
    dijkstra = plus_court_chemin(cout, (5, 5), (50, 50))
    a_etoile = plus_court_chemin(cout, (5, 5), (50, 50), a_etoile=True)
    assert a_etoile.cout == pytest.approx(dijkstra.cout)
    assert a_etoile.pixels_visites < dijkstra.pixels_visites / 2


def test_points_inaccessibles():
    cout = np.ones((10, 10))
    cout[:, 5] = np.inf
    with pytest.raises(ValueError, match="aucun chemin"):
        plus_court_chemin(cout, (0, 0), (0, 9))
    with pytest.raises(ValueError, match="hors du champ"):
        plus_court_chemin(cout, (0, 5), (0, 9))
    with pytest.raises(ValueError, match="hors du champ"):
        plus_court_chemin(cout, (0, 0), (0, 10))


def test_le_trace_suit_le_vaisseau(retine):
    rgb, masque, verite = retine
    cout = carte_de_cout(pretraiter(rgb, masque).vaisseaux, masque, alpha=1.0)
    lignes = np.flatnonzero(verite[:, 8])
    depart = (int(lignes[len(lignes) // 2]), 8)
    lignes = np.flatnonzero(verite[:, 88])
    arrivee = (int(lignes[len(lignes) // 2]), 88)
    chemin = plus_court_chemin(cout, depart, arrivee)
    assert evaluer_trace(chemin.pixels, chemin.pixels, verite)["precision"] >= 0.95
