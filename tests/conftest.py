import numpy as np
import pytest


def fabriquer_retine(graine: int = 0, taille: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fond d'œil synthétique : fond clair légèrement dégradé, un vaisseau sombre sinueux
    et une branche verticale, bruit gaussien. Renvoie (rgb, masque, vérité)."""
    rng = np.random.default_rng(graine)
    lignes, colonnes = np.mgrid[:taille, :taille].astype(float)
    centre = taille / 2 + 15 * np.sin(colonnes / 15)
    distance = np.minimum(np.abs(lignes - centre), np.abs(colonnes - 0.7 * taille) + 1000 * (lignes > centre))
    verite = distance <= 1.5
    vert = 0.55 + 0.1 * colonnes / taille - 0.25 * np.exp(-(distance**2) / (2 * 1.2**2))
    vert += rng.normal(0, 0.02, vert.shape)
    rgb = np.stack([np.full_like(vert, 0.8), vert, 0.3 * vert], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), np.ones(verite.shape, bool), verite


@pytest.fixture
def retine():
    return fabriquer_retine()
