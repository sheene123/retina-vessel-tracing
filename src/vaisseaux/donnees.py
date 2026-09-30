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


def cadre_carre(rgb: np.ndarray) -> tuple[int, int, int, int, int, int, int]:
    """Cadre utilisé par `carre_fond_oeil` : (haut, gauche, hauteur, largeur) du disque dans
    l'image, côté du carré et décalages (vertical, horizontal) du disque dans le carré."""
    masque = estimer_masque_fov(rgb, marge=0)
    lignes, colonnes = np.nonzero(masque)
    haut, gauche, h, w = 0, 0, rgb.shape[0], rgb.shape[1]
    if lignes.size:
        haut, gauche = int(lignes.min()), int(colonnes.min())
        h, w = int(lignes.max()) + 1 - haut, int(colonnes.max()) + 1 - gauche
    cote = max(h, w)
    return haut, gauche, h, w, cote, (cote - h) // 2, (cote - w) // 2


def carre_fond_oeil(rgb: np.ndarray, taille: int = 384) -> np.ndarray:
    """Recadre sur le disque du fond d'œil, complète en carré noir et redimensionne : même
    présentation que les images prétraitées d'ODIR-5K sur lesquelles le réseau est entraîné."""
    haut, gauche, h, w, cote, dh, dw = cadre_carre(rgb)
    carre = np.zeros((cote, cote, 3), dtype=np.uint8)
    carre[dh : dh + h, dw : dw + w] = rgb[haut : haut + h, gauche : gauche + w, :3]
    return np.asarray(Image.fromarray(carre).resize((taille, taille), Image.Resampling.BILINEAR))


def carte_sur_image(carte: np.ndarray, forme: tuple[int, int], cadre: tuple[int, ...]) -> np.ndarray:
    """Replace une carte calculée sur le carré (par exemple 12 × 12) dans l'image d'origine :
    agrandissement lissé au côté du carré, retrait des bandes noires, placement dans le cadre."""
    haut, gauche, h, w, cote, dh, dw = cadre
    grande = np.asarray(Image.fromarray(carte.astype(np.float32)).resize((cote, cote), Image.Resampling.BICUBIC))
    sortie = np.zeros(forme, dtype=np.float32)
    sortie[haut : haut + h, gauche : gauche + w] = grande[dh : dh + h, dw : dw + w]
    return sortie


def localiser_papille(carre: np.ndarray) -> tuple[int, int]:
    """Centre approximatif de la papille (disque optique) sur un fond d'œil recadré : la zone la
    plus claire en rouge et en vert après un fort lissage, en évitant le bord du champ de vue.
    Sert à fabriquer des gros plans de la papille (augmentation et test du glaucome)."""
    h, w = carre.shape[:2]
    x = carre[..., :3].astype(np.float32)
    champ = ndi.binary_erosion(x[..., 0] > 20, iterations=max(2, h // 25))
    clarte = ndi.gaussian_filter(0.5 * x[..., 0] + x[..., 1], sigma=max(2.0, h / 40))
    clarte[~champ] = -1
    ligne, colonne = np.unravel_index(int(np.argmax(clarte)), clarte.shape)
    return int(ligne), int(colonne)


def gros_plan_papille(carre: np.ndarray, part: float = 0.4, taille: int | None = None) -> np.ndarray:
    """Gros plan carré centré sur la papille, de côté `part` × celui de l'image, comme une photo
    de papille prise de près ; redimensionné à `taille` (par défaut, la taille d'origine)."""
    h, w = carre.shape[:2]
    ligne, colonne = localiser_papille(carre)
    cote = max(8, int(part * min(h, w)))
    haut = min(max(0, ligne - cote // 2), h - cote)
    gauche = min(max(0, colonne - cote // 2), w - cote)
    zoom = carre[haut : haut + cote, gauche : gauche + cote]
    taille = taille or w
    return np.asarray(Image.fromarray(np.ascontiguousarray(zoom)).resize((taille, taille), Image.Resampling.BILINEAR))


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
