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
