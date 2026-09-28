"""Prétraitement : réduction du bruit et rehaussement des vaisseaux.

Chaîne : canal vert -> extension hors champ de vue -> CLAHE -> débruitage
-> filtre de Frangi multi-échelle (vaisseaux sombres sur fond clair).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, restoration

DEBRUITAGES = ("aucun", "gaussien", "median", "bilateral", "nl_means")


@dataclass(frozen=True)
class ParametresPretraitement:
    clahe: bool = True
    clahe_limite: float = 0.01
    debruitage: str = "nl_means"
    force_debruitage: float = 1.0
    echelles: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)  # sigmas du filtre de Frangi, en pixels


@dataclass
class CartesPretraitement:
    vert: np.ndarray  # canal vert normalisé [0, 1]
    rehaussee: np.ndarray  # après CLAHE et débruitage
    vaisseaux: np.ndarray  # probabilité « vaisseau » (vesselness) normalisée [0, 1], nulle hors FOV


def etendre_hors_fov(image: np.ndarray, masque: np.ndarray, sigmas: tuple[float, ...] = (4, 16, 64)) -> np.ndarray:
    """Remplit l'extérieur du champ de vue par convolution normalisée, pour que les filtres
    ne détectent pas le bord du disque comme un vaisseau."""
    resultat = np.where(masque, image, 0.0)
    connu = masque.astype(float)
    for sigma in sigmas:
        num = filters.gaussian(resultat * connu, sigma)
        den = filters.gaussian(connu, sigma)
        remplissage = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-3)
        nouveau = (connu == 0) & (den > 1e-3)
        resultat = np.where(nouveau, remplissage, resultat)
        connu = np.maximum(connu, nouveau)
    return np.where(connu > 0, resultat, float(np.median(image[masque])))


def estimer_bruit(image: np.ndarray) -> float:
    """Écart-type du bruit gaussien par la méthode d'Immerkær (1996) : un noyau qui annule
    les variations lentes de l'image et ne laisse passer que le bruit."""
    noyau = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)
    reponse = np.abs(ndi.convolve(image.astype(float), noyau))[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * reponse.mean() / 6)


def debruiter(image: np.ndarray, methode: str, force: float = 1.0) -> np.ndarray:
    if methode == "aucun":
        return image
    if methode == "gaussien":
        return filters.gaussian(image, sigma=1.0 * force)
    if methode == "median":
        return filters.median(image, footprint=np.ones((3, 3)))
    if methode == "bilateral":
        return restoration.denoise_bilateral(image, sigma_color=0.05 * force, sigma_spatial=2)
    if methode == "nl_means":
        sigma = estimer_bruit(image)
        return restoration.denoise_nl_means(
            image, h=0.8 * sigma * force, sigma=sigma, patch_size=5, patch_distance=6, fast_mode=True
        )
    raise ValueError(f"débruitage inconnu : {methode!r} (choix : {', '.join(DEBRUITAGES)})")


def pretraiter(
    rgb: np.ndarray, masque: np.ndarray, parametres: ParametresPretraitement | None = None
) -> CartesPretraitement:
    p = parametres or ParametresPretraitement()
    vert = rgb[..., 1].astype(float) / 255.0
    image = etendre_hors_fov(vert, masque)
    if p.clahe:
        image = exposure.equalize_adapthist(np.clip(image, 0, 1), clip_limit=p.clahe_limite)
    image = debruiter(image, p.debruitage, p.force_debruitage)
    reponse = filters.frangi(image, sigmas=p.echelles, black_ridges=True)
    echelle = np.percentile(reponse[masque], 99.5) if masque.any() else 1.0
    vaisseaux = np.clip(reponse / max(echelle, 1e-12), 0, 1) * masque
    return CartesPretraitement(vert=vert, rehaussee=image, vaisseaux=vaisseaux)


def normaliser_pour_reseau(rgb: np.ndarray, masque: np.ndarray, multiple: int = 16) -> np.ndarray:
    """Entrée du U-Net : image normalisée canal par canal dans le champ de vue, (3, H, W),
    complétée par des zéros jusqu'à un multiple de `multiple` (sous-échantillonnages du réseau)."""
    x = rgb.astype(np.float32) / 255.0
    moyenne, ecart = x[masque].mean(axis=0), x[masque].std(axis=0) + 1e-6
    x = (((x - moyenne) / ecart) * masque[..., None]).transpose(2, 0, 1)
    h, w = masque.shape
    return np.pad(x, ((0, 0), (0, -h % multiple), (0, -w % multiple))).astype(np.float32)
