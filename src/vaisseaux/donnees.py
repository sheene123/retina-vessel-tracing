"""Chargement du jeu de données DRIVE (arborescence officielle).

    DRIVE/
      training/images/21_training.tif   training/1st_manual/21_manual1.gif   training/mask/...
      test/images/01_test.tif           test/1st_manual/01_manual1.gif       test/2nd_manual/...

Les masques de champ de vue (FOV) sont estimés à partir de l'image s'ils sont absents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PARTIES = {"entrainement": "training", "test": "test"}


@dataclass
class ImageRetine:
    identifiant: str
    rgb: np.ndarray  # (H, W, 3) uint8
    masque: np.ndarray  # champ de vue, booléen
    verite: np.ndarray | None  # annotation du 1er expert, booléen
    verite2: np.ndarray | None = None  # 2e observateur (partie test officielle uniquement)


def lire_image(chemin: Path | str) -> np.ndarray:
    return np.asarray(Image.open(chemin))


def estimer_masque_fov(rgb: np.ndarray, seuil: int = 25, marge: int = 3) -> np.ndarray:
    """Le champ de vue est le disque éclairé : canal rouge au-dessus du fond noir."""
    masque = ndi.binary_opening(rgb[..., 0] > seuil, iterations=2)
    masque = ndi.binary_fill_holes(masque)
    etiquettes, n = ndi.label(masque)
    if n > 1:
        tailles = ndi.sum(masque, etiquettes, range(1, n + 1))
        masque = etiquettes == int(np.argmax(tailles)) + 1
    # iterations=0 signifierait « éroder jusqu'à stabilité » pour scipy : on l'évite.
    return ndi.binary_erosion(masque, iterations=marge) if marge > 0 else masque


def _premier(dossier: Path, motif: str) -> Path | None:
    trouves = sorted(dossier.glob(motif)) if dossier.is_dir() else []
    return trouves[0] if trouves else None


def lister(racine: Path | str, partie: str) -> list[str]:
    dossier = Path(racine) / PARTIES.get(partie, partie) / "images"
    return sorted(p.name.split("_")[0] for p in dossier.glob("*_*"))


def charger(racine: Path | str, partie: str, identifiant: str) -> ImageRetine:
    base = Path(racine) / PARTIES.get(partie, partie)
    chemin = _premier(base / "images", f"{identifiant}_*")
    if chemin is None:
        raise FileNotFoundError(f"image {identifiant} introuvable dans {base / 'images'}")
    rgb = lire_image(chemin)[..., :3]
    chemin_masque = _premier(base / "mask", f"{identifiant}_*")
    masque = lire_image(chemin_masque) > 0 if chemin_masque else estimer_masque_fov(rgb)
    annotations = [_premier(base / d, f"{identifiant}_*") for d in ("1st_manual", "2nd_manual")]
    v1, v2 = (lire_image(a) > 0 if a else None for a in annotations)
    return ImageRetine(identifiant, rgb, masque, v1, v2)


# normalisation ImageNet, utilisée par le réseau de détection des troubles
MOYENNE_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
ECART_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def carre_fond_oeil(rgb: np.ndarray, taille: int = 384) -> np.ndarray:
    """Recadre sur le disque du fond d'œil, complète en carré noir et redimensionne : même
    présentation que les images prétraitées d'ODIR-5K sur lesquelles le réseau est entraîné."""
    masque = estimer_masque_fov(rgb, marge=0)
    lignes, colonnes = np.nonzero(masque)
    if lignes.size:
        rgb = rgb[lignes.min() : lignes.max() + 1, colonnes.min() : colonnes.max() + 1]
    h, w = rgb.shape[:2]
    cote = max(h, w)
    carre = np.zeros((cote, cote, 3), dtype=np.uint8)
    carre[(cote - h) // 2 : (cote - h) // 2 + h, (cote - w) // 2 : (cote - w) // 2 + w] = rgb[..., :3]
    return np.asarray(Image.fromarray(carre).resize((taille, taille), Image.Resampling.BILINEAR))


def isoler_fond_oeil(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int] | None]:
    """Recadre sur le fond d'œil trouvé par `localiser_fond_oeil` et met en noir tout le reste.
    Renvoie l'image, le masque de l'œil et le cadre, ou l'image intacte si rien n'est trouvé."""
    trouve = localiser_fond_oeil(rgb)
    if trouve is None:
        return rgb, None, None
    (haut, bas, gauche, droite), oeil = trouve
    image = rgb[haut:bas, gauche:droite, :3].copy()
    image[~oeil] = 0
    return image, oeil, (haut, bas, gauche, droite)


def preparer_fond_oeil(rgb: np.ndarray, taille: int = 384) -> np.ndarray:
    """Chaîne complète appliquée avant le réseau des troubles, identique à l'entraînement et dans
    la démo : isolement de l'œil puis carré noir centré."""
    return carre_fond_oeil(isoler_fond_oeil(rgb[..., :3])[0], taille)


def entree_imagenet(carre: np.ndarray) -> np.ndarray:
    """(taille, taille, 3) uint8 -> (3, taille, taille) float32 normalisé."""
    return ((carre.astype(np.float32) / 255.0 - MOYENNE_IMAGENET) / ECART_IMAGENET).transpose(2, 0, 1).copy()


def localiser_fond_oeil(
    rgb: np.ndarray, surface_min: float = 0.03
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Trouve le fond d'œil dans une image quelconque (photo, capture d'écran, page de manuel).

    Les pixels de rétine sont orangés et saturés : on garde la plus grande zone de cette couleur,
    puis son enveloppe convexe (qui englobe le disque optique jaune et les annotations tracées
    dessus). Renvoie (haut, bas, gauche, droite) et le masque de l'œil dans ce cadre, ou None si
    aucune zone assez grande n'est trouvée.
    """
    from skimage.morphology import convex_hull_image

    h, w = rgb.shape[:2]
    echelle = max(1.0, max(h, w) / 400)
    petit = np.asarray(Image.fromarray(rgb[..., :3]).resize((max(1, round(w / echelle)), max(1, round(h / echelle)))))
    x = petit.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    maximum, minimum = x.max(axis=-1), x.min(axis=-1)
    saturation = (maximum - minimum) / (maximum + 1e-6)
    retine = (r >= g) & (g >= 0.8 * b) & (saturation > 0.35) & (maximum > 0.25)  # teintes rouge à jaune
    retine = ndi.binary_closing(ndi.binary_opening(retine, iterations=2), iterations=3)
    etiquettes, n = ndi.label(retine)
    if n == 0:
        return None
    tailles = ndi.sum(retine, etiquettes, range(1, n + 1))
    zone = etiquettes == int(np.argmax(tailles)) + 1
    if zone.sum() < surface_min * zone.size:
        return None
    enveloppe = convex_hull_image(ndi.binary_fill_holes(zone))
    lignes, colonnes = np.nonzero(enveloppe)
    haut, bas = int(lignes.min() * echelle), min(h, int((lignes.max() + 1) * echelle))
    gauche, droite = int(colonnes.min() * echelle), min(w, int((colonnes.max() + 1) * echelle))
    masque = np.asarray(
        Image.fromarray(enveloppe[lignes.min() : lignes.max() + 1, colonnes.min() : colonnes.max() + 1]).resize(
            (droite - gauche, bas - haut), Image.Resampling.NEAREST
        )
    )
    return (haut, bas, gauche, droite), ndi.binary_erosion(masque, iterations=3)
