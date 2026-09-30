"""Contrôle de la qualité d'une photo du fond d'œil, avant d'en interpréter les résultats.

Mesures simples et explicables, calculées sur le fond d'œil recadré en carré de 448 pixels (même
échelle que les images d'entraînement) :

- netteté : écart-type du laplacien du canal vert lissé, dans le champ de vue ; une photo floue
  perd les fins détails (vaisseaux, contours) ;
- luminosité : moyenne du canal le plus clair ; surexposition et zones noires : part des pixels
  presque blancs ou presque noirs ; contraste : écart-type du canal vert.

Seuils calibrés sur deux jeux (voir docs/troubles.md) : les 159 photos que les ophtalmologistes de
JSIEC ont classées « fond d'œil flou », contre ses 841 autres photos, et 1 500 photos des bases
d'entraînement. La netteté sépare les photos floues des autres avec une AUROC de 0,996.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from vaisseaux.donnees import carre_fond_oeil

COTE = 448
# floue : 81 % des photos floues de JSIEC, 0,4 % des nettes ; un peu floue : 96 % et 4 % en cumulé
SEUIL_FLOUE, SEUIL_UN_PEU_FLOUE = 0.18, 0.25
SEUIL_SOMBRE = 0.25  # luminosité moyenne ; 1 % des photos d'entraînement sont sous 0,29
SEUIL_SUREXPOSEE = 0.5  # part des pixels presque blancs
SEUIL_CONTRASTE = 0.02  # 1 % des photos d'entraînement sont sous 0,023


def mesures(carre: np.ndarray) -> dict[str, float]:
    x = carre[..., :3].astype(np.float32) / 255
    champ = ndi.binary_erosion(x[..., 0] > 0.08, iterations=12)
    if champ.sum() < 1000:
        champ = np.ones(champ.shape, bool)
    vert = x[..., 1]
    laplacien = ndi.laplace(ndi.gaussian_filter(vert, 1.0))
    maximum = x.max(axis=-1)[champ]
    return {
        "nettete": float(laplacien[champ].std() * 100),
        "luminosite": float(maximum.mean()),
        "contraste": float(vert[champ].std()),
        "surexposee": float((maximum > 0.97).mean()),
    }


def evaluer(rgb: np.ndarray) -> dict:
    """Niveau (« bonne », « moyenne », « insuffisante »), problèmes détectés et mesures."""
    m = mesures(carre_fond_oeil(rgb, COTE))
    graves, legers = [], []
    if m["nettete"] < SEUIL_FLOUE:
        graves.append("image floue")
    elif m["nettete"] < SEUIL_UN_PEU_FLOUE:
        legers.append("image un peu floue")
    if m["luminosite"] < SEUIL_SOMBRE:
        graves.append("image trop sombre")
    if m["surexposee"] > SEUIL_SUREXPOSEE:
        graves.append("image surexposée")
    if m["contraste"] < SEUIL_CONTRASTE:
        legers.append("très peu contrastée")
    niveau = "insuffisante" if graves else "moyenne" if legers else "bonne"
    return {"niveau": niveau, "problemes": graves + legers, "mesures": {k: round(v, 4) for k, v in m.items()}}
