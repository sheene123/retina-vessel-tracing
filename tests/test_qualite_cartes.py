import numpy as np
from scipy import ndimage as ndi

from vaisseaux.donnees import cadre_carre, carre_fond_oeil, carte_sur_image
from vaisseaux.qualite import evaluer


def _fond_oeil(flou: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    lignes, colonnes = np.mgrid[:400, :500]
    image = np.zeros((400, 500, 3), np.float32)
    disque = (lignes - 200) ** 2 + (colonnes - 250) ** 2 < 180**2
    image[disque] = (0.8, 0.35, 0.12)
    # « vaisseaux » : fines lignes sombres dans le vert
    for k in range(12):
        image[:, 30 + 38 * k : 32 + 38 * k, 1] *= 0.4
    image[..., 1] += rng.normal(0, 0.01, image.shape[:2])
    if flou:
        image = np.stack([ndi.gaussian_filter(image[..., c], flou) for c in range(3)], axis=-1)
    image *= disque[..., None]
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def test_qualite_repere_le_flou():
    assert evaluer(_fond_oeil())["niveau"] == "bonne"
    floue = evaluer(_fond_oeil(flou=6))
    assert floue["niveau"] != "bonne" and any("floue" in p for p in floue["problemes"])
    sombre = evaluer((_fond_oeil() * 0.15).astype(np.uint8))
    assert "image trop sombre" in sombre["problemes"]


def test_carte_revient_sur_l_image():
    image = _fond_oeil()
    cadre = cadre_carre(image)
    carre = carre_fond_oeil(image, 384)
    carte = np.zeros((12, 12), np.float32)
    carte[5:7, 5:7] = 1  # zone centrale du carré
    retour = carte_sur_image(carte, image.shape[:2], cadre)
    assert retour.shape == image.shape[:2] and carre.shape == (384, 384, 3)
    ligne, colonne = np.unravel_index(np.argmax(retour), retour.shape)
    assert abs(ligne - 200) < 25 and abs(colonne - 250) < 25  # le centre du carré revient au centre de l'œil
