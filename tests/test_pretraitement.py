import numpy as np
import pytest

from vaisseaux.pretraitement import DEBRUITAGES, ParametresPretraitement, estimer_bruit, etendre_hors_fov, pretraiter


@pytest.mark.parametrize("debruitage", DEBRUITAGES)
def test_rehaussement_contraste_vaisseaux(retine, debruitage):
    rgb, masque, verite = retine
    cartes = pretraiter(rgb, masque, ParametresPretraitement(debruitage=debruitage))
    v = cartes.vaisseaux
    assert v.shape == verite.shape
    assert np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1
    assert v[verite].mean() > 3 * v[~verite].mean()


def test_rien_hors_du_champ_de_vue(retine):
    rgb, _, _ = retine
    masque = np.zeros(rgb.shape[:2], bool)
    masque[10:-10, 10:-10] = True
    assert np.all(pretraiter(rgb, masque).vaisseaux[~masque] == 0)


def test_estimation_du_bruit():
    rng = np.random.default_rng(0)
    lignes, colonnes = np.mgrid[:128, :128]
    image = 0.5 + 0.2 * np.sin(colonnes / 20) + rng.normal(0, 0.03, (128, 128))
    assert estimer_bruit(image) == pytest.approx(0.03, rel=0.1)


def test_extension_hors_fov():
    image = np.full((50, 50), 0.7)
    masque = np.zeros((50, 50), bool)
    masque[15:35, 15:35] = True
    image[~masque] = 0.0
    etendue = etendre_hors_fov(image, masque)
    assert np.allclose(etendue[masque], 0.7)
    assert np.allclose(etendue, 0.7, atol=1e-6)
