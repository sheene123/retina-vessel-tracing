import numpy as np
import pytest
from PIL import Image

from vaisseaux.donnees import preparer_fond_oeil


def _page_avec_oeil() -> np.ndarray:
    page = np.full((400, 700, 3), 255, dtype=np.uint8)
    page[40:60, 30:250] = 0  # « texte »
    page[20:380, 330:690] = 0
    lignes, colonnes = np.mgrid[:400, :700]
    page[(lignes - 200) ** 2 + (colonnes - 510) ** 2 < 170**2] = (220, 90, 30)
    return page


def test_preparation_identique_a_la_demo():
    carre = preparer_fond_oeil(_page_avec_oeil())
    assert carre.shape == (384, 384, 3) and carre.dtype == np.uint8
    assert carre[192, 192, 0] > 150  # l'œil occupe le centre
    assert carre[:8, :8].max() == 0  # le texte et la page blanche ont disparu


def test_augmentations():
    pytest.importorskip("torchvision")
    pytest.importorskip("sklearn")
    from vaisseaux.troubles import TAILLE, Annotations, BasseResolution, transformations

    image = Image.fromarray(preparer_fond_oeil(_page_avec_oeil(), 448))
    for entrainement in (True, False):
        x = transformations(entrainement)(image)
        assert tuple(x.shape) == (3, TAILLE, TAILLE) and bool(x.isfinite().all())
    assert np.any(np.asarray(Annotations(p=1.0)(image)) != np.asarray(image))
    assert BasseResolution(p=1.0)(image).size == image.size


def test_etiquettes_du_test_externe_coherentes():
    pytest.importorskip("sklearn")
    from vaisseaux.sources_troubles import JSIEC_INCERTAINS, JSIEC_POSITIFS
    from vaisseaux.troubles import CLES

    assert set(JSIEC_POSITIFS) <= set(CLES) and set(JSIEC_INCERTAINS) <= set(CLES)
    for cle, positifs in JSIEC_POSITIFS.items():
        assert not set(positifs) & set(JSIEC_INCERTAINS.get(cle, ()))
