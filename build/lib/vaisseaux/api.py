"""API HTTP du traceur de vaisseaux.

    uvicorn vaisseaux.api:app --reload
    curl -F image=@data/DRIVE/test/images/01_test.tif -F depart_x=280 -F depart_y=90 \
         -F arrivee_x=330 -F arrivee_y=200 http://127.0.0.1:8000/tracer

Les coordonnées suivent la convention des images : x = colonne, y = ligne, origine en haut
à gauche. Le prétraitement d'une image est mis en cache : les tracés suivants sur la même
image ne coûtent que la recherche du plus court chemin.
"""

from __future__ import annotations

import hashlib
import io
import time
from collections import OrderedDict
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from scipy import ndimage as ndi

from . import __version__
from .donnees import estimer_masque_fov
from .graphe import Chemin, carte_de_cout, plus_court_chemin
from .pretraitement import DEBRUITAGES, CartesPretraitement, ParametresPretraitement, pretraiter

COTE_MAX = 2048
TAILLE_CACHE = 16
ALPHA_DEFAUT = 1.0  # choisi sur les images d'entraînement de DRIVE (voir resultats/evaluation.md)

app = FastAPI(
    title="Traceur de vaisseaux rétiniens",
    version=__version__,
    description="Plus court chemin (Dijkstra / A*) sur le graphe des pixels d'une image de fond d'œil.",
)
_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, CartesPretraitement]] = OrderedDict()


def _preparer(contenu: bytes, debruitage: str) -> tuple[np.ndarray, np.ndarray, CartesPretraitement]:
    cle = f"{hashlib.sha256(contenu).hexdigest()}:{debruitage}"
    if cle in _cache:
        _cache.move_to_end(cle)
        return _cache[cle]
    try:
        image = Image.open(io.BytesIO(contenu))
        image.load()
    except (UnidentifiedImageError, OSError) as erreur:
        raise HTTPException(400, "image illisible") from erreur
    if max(image.size) > COTE_MAX:
        raise HTTPException(413, f"image trop grande (côté maximal {COTE_MAX} px)")
    rgb = np.asarray(image.convert("RGB"))
    masque = estimer_masque_fov(rgb)
    cartes = pretraiter(rgb, masque, ParametresPretraitement(debruitage=debruitage))
    _cache[cle] = (rgb, masque, cartes)
    if len(_cache) > TAILLE_CACHE:
        _cache.popitem(last=False)
    return rgb, masque, cartes


def _tracer(
    contenu: bytes,
    depart: tuple[int, int],
    arrivee: tuple[int, int],
    debruitage: str,
    algorithme: str,
    alpha: float,
) -> tuple[np.ndarray, Chemin, dict[str, float]]:
    if debruitage not in DEBRUITAGES:
        raise HTTPException(422, f"débruitage inconnu, choix : {', '.join(DEBRUITAGES)}")
    t0 = time.perf_counter()
    rgb, masque, cartes = _preparer(contenu, debruitage)
    t1 = time.perf_counter()
    cout = carte_de_cout(cartes.vaisseaux, masque, alpha)
    try:
        # (x, y) côté API, (ligne, colonne) côté graphe
        chemin = plus_court_chemin(cout, depart[::-1], arrivee[::-1], a_etoile=algorithme == "a_etoile")
    except ValueError as erreur:
        raise HTTPException(422, str(erreur)) from erreur
    t2 = time.perf_counter()
    return rgb, chemin, {"pretraitement": round(1e3 * (t1 - t0), 1), "recherche": round(1e3 * (t2 - t1), 1)}


@app.get("/sante")
def sante() -> dict:
    return {"statut": "ok", "version": __version__}


@app.post("/tracer")
def tracer(
    image: UploadFile = File(..., description="image de fond d'œil (TIFF, PNG, JPEG…)"),
    depart_x: int = Form(...),
    depart_y: int = Form(...),
    arrivee_x: int = Form(...),
    arrivee_y: int = Form(...),
    debruitage: str = Form("nl_means"),
    algorithme: Literal["dijkstra", "a_etoile"] = Form("a_etoile"),
    alpha: float = Form(ALPHA_DEFAUT, gt=0, le=5),
) -> dict:
    _, chemin, durees = _tracer(
        image.file.read(), (depart_x, depart_y), (arrivee_x, arrivee_y), debruitage, algorithme, alpha
    )
    return {
        "chemin": chemin.pixels[:, ::-1].tolist(),
        "longueur_px": round(chemin.longueur, 2),
        "cout": round(chemin.cout, 3),
        "pixels_visites": chemin.pixels_visites,
        "durees_ms": durees,
    }


@app.post("/tracer/image", response_class=Response)
def tracer_image(
    image: UploadFile = File(...),
    depart_x: int = Form(...),
    depart_y: int = Form(...),
    arrivee_x: int = Form(...),
    arrivee_y: int = Form(...),
    debruitage: str = Form("nl_means"),
    algorithme: Literal["dijkstra", "a_etoile"] = Form("a_etoile"),
    alpha: float = Form(ALPHA_DEFAUT, gt=0, le=5),
) -> Response:
    """Même calcul que /tracer, renvoyé sous forme d'image PNG avec le tracé superposé."""
    rgb, chemin, _ = _tracer(
        image.file.read(), (depart_x, depart_y), (arrivee_x, arrivee_y), debruitage, algorithme, alpha
    )
    return Response(content=superposer(rgb, chemin.pixels), media_type="image/png")


def superposer(rgb: np.ndarray, pixels: np.ndarray, couleur: tuple[int, int, int] = (0, 255, 255)) -> bytes:
    trace = np.zeros(rgb.shape[:2], dtype=bool)
    trace[tuple(pixels.T)] = True
    trace = ndi.binary_dilation(trace)
    sortie = rgb.copy()
    sortie[trace] = couleur
    tampon = io.BytesIO()
    Image.fromarray(sortie).save(tampon, format="PNG")
    return tampon.getvalue()
