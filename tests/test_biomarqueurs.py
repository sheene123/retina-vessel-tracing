import numpy as np
import pytest

from vaisseaux.biomarqueurs import dimension_fractale, mesurer, reseau_principal, tortuosite, tortuosite_ponderee


def _masque(n=200):
    return np.ones((n, n), dtype=bool)


def test_ligne_droite():
    image = np.zeros((200, 200), dtype=bool)
    image[98:103, 20:180] = True  # vaisseau droit de 5 px de large
    m = mesurer(image, _masque())
    assert m["tortuosite"] == pytest.approx(1.0, abs=0.02)
    assert m["calibre_moyen"] == pytest.approx(5.0, abs=0.6)
    assert m["densite"] == pytest.approx(5 * 160 / 200**2)
    assert m["fragments"] == 1 and m["bifurcations"] == 0


def _squelette(image):
    from skimage.morphology import skeletonize

    return skeletonize(image)


def test_demi_cercle():
    lignes, colonnes = np.mgrid[:200, :200]
    rayon = np.hypot(lignes - 120, colonnes - 100)
    arc = (np.abs(rayon - 60) <= 1.5) & (lignes <= 120)
    assert tortuosite(_squelette(arc)) == pytest.approx(np.pi / 2, abs=0.06)


def test_diagonale_non_biaisee():
    # vaisseau diagonal de 4 px de large : les pas en diagonale comptent sqrt(2)
    lignes, colonnes = np.mgrid[:200, :200]
    image = (np.abs(lignes - colonnes) / np.sqrt(2) <= 2) & (lignes > 20) & (lignes < 180)
    assert tortuosite(_squelette(image)) == pytest.approx(1.0, abs=0.05)


def test_dimension_fractale():
    plein = np.ones((256, 256), dtype=bool)
    ligne = np.zeros((256, 256), dtype=bool)
    ligne[128, :] = True
    assert dimension_fractale(plein) == pytest.approx(2.0, abs=0.05)
    assert dimension_fractale(ligne) == pytest.approx(1.0, abs=0.1)


def test_bifurcation_et_fragments():
    image = np.zeros((200, 200), dtype=bool)
    image[99:102, 20:180] = True
    image[20:100, 99:102] = True  # branche en T
    image[150:153, 30:60] = True  # fragment isolé
    m = mesurer(image, _masque())
    assert m["bifurcations"] > 0
    assert m["fragments"] == 2


def test_tortuosite_ponderee_ignore_les_courts_segments():
    lignes, colonnes = np.mgrid[:200, :200]
    rayon = np.hypot(lignes - 120, colonnes - 100)
    arc = (np.abs(rayon - 60) <= 1.5) & (lignes <= 120)
    assert tortuosite_ponderee(_squelette(arc)) == pytest.approx(np.pi / 2, abs=0.06)
    court = np.zeros((200, 200), dtype=bool)
    court[50:53, 20:45] = True  # 25 px : ignoré au seuil de 40 px
    assert np.isnan(tortuosite_ponderee(_squelette(court)))


def test_reseau_principal_retire_les_capillaires():
    image = np.zeros((200, 200), dtype=bool)
    image[98:103, 20:180] = True  # gros vaisseau, 5 px
    image[20:90, 60] = True  # capillaire, 1 px
    principal = reseau_principal(image)
    assert principal[100, 100] and not principal[50, 60]
