"""Démo interactive (Hugging Face Space) : cliquer deux points sur un fond d'œil,
le plus court chemin suit le vaisseau qui les relie.

    python demo/app.py
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from vaisseaux.api import ALPHA_DEFAUT
from vaisseaux.donnees import estimer_masque_fov
from vaisseaux.graphe import carte_de_cout, plus_court_chemin
from vaisseaux.pretraitement import DEBRUITAGES, ParametresPretraitement, pretraiter

MIROIR_DRIVE = "Zomba/DRIVE-digital-retinal-images-for-vessel-extraction"
EXEMPLES_DRIVE = ("val/input/01.tif", "val/input/02.tif", "val/input/19.tif")
COTE_MAX = 1024  # au-delà, l'image est réduite pour le calcul
COULEUR_TRACE = (0, 230, 255)
_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}


def _exemples() -> list[str]:
    """Télécharge quelques images de test DRIVE depuis le miroir public (non redistribuées)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return []
    dossier = Path(tempfile.gettempdir()) / "exemples_drive"
    dossier.mkdir(exist_ok=True)
    chemins = []
    for fichier in EXEMPLES_DRIVE:
        try:
            tif = hf_hub_download(MIROIR_DRIVE, fichier, repo_type="dataset")
        except Exception:  # pas de réseau : la démo reste utilisable par téléversement
            continue
        png = dossier / f"drive_{Path(fichier).stem}.png"
        Image.open(tif).convert("RGB").save(png)
        chemins.append(str(png))
    return chemins


def _rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return np.ascontiguousarray(image[..., :3].astype(np.uint8))


def _carte(rgb: np.ndarray, debruitage: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(masque, vesselness, échelle) à la résolution de calcul, mis en cache par image."""
    cle = hashlib.sha1(rgb.tobytes()).hexdigest() + debruitage
    if cle not in _cache:
        echelle = max(rgb.shape[:2]) / COTE_MAX if max(rgb.shape[:2]) > COTE_MAX else 1.0
        calcul = rgb
        if echelle > 1:
            taille = (round(rgb.shape[1] / echelle), round(rgb.shape[0] / echelle))
            calcul = np.asarray(Image.fromarray(rgb).resize(taille, Image.Resampling.LANCZOS))
        masque = estimer_masque_fov(calcul)
        cartes = pretraiter(calcul, masque, ParametresPretraitement(debruitage=debruitage))
        if len(_cache) > 8:
            _cache.clear()
        _cache[cle] = (masque, cartes.vaisseaux, echelle)
    return _cache[cle]


def _dessiner(rgb: np.ndarray, points: list[tuple[int, int]], trace: np.ndarray | None) -> np.ndarray:
    image = Image.fromarray(rgb)
    dessin = ImageDraw.Draw(image)
    epaisseur = max(2, round(max(rgb.shape[:2]) / 300))
    if trace is not None and len(trace) > 1:
        dessin.line([(float(x), float(y)) for x, y in trace], fill=COULEUR_TRACE, width=epaisseur, joint="curve")
    rayon = 2 * epaisseur + 2
    for x, y in points:
        dessin.ellipse((x - rayon, y - rayon, x + rayon, y + rayon), fill="white", outline=(40, 40, 40), width=2)
    return np.asarray(image)


def _tracer(image, points, debruitage, algorithme, alpha):
    rgb = _rgb(image)
    if len(points) < 2:
        return _dessiner(rgb, points, None), "Cliquez le point d'arrivée."
    debut = time.perf_counter()
    masque, vaisseaux, echelle = _carte(rgb, debruitage)
    milieu = time.perf_counter()
    cout = carte_de_cout(vaisseaux, masque, alpha)
    depart, arrivee = ((round(y / echelle), round(x / echelle)) for x, y in points)
    try:
        chemin = plus_court_chemin(cout, depart, arrivee, a_etoile=algorithme == "A*")
    except ValueError as erreur:
        return _dessiner(rgb, points, None), f"**Impossible de tracer** : {erreur}."
    fin = time.perf_counter()
    trace = chemin.pixels[:, ::-1] * echelle
    texte = (
        f"**Longueur** : {chemin.longueur * echelle:.0f} px · **pixels explorés** : {chemin.pixels_visites:,} · "
        f"**prétraitement** : {1e3 * (milieu - debut):.0f} ms · **recherche** : {1e3 * (fin - milieu):.0f} ms"
    ).replace(",", " ")
    return _dessiner(rgb, points, trace), texte


def cliquer(image, points, debruitage, algorithme, alpha, evt: gr.SelectData):
    if image is None:
        return points, None, "Chargez d'abord une image."
    points = [] if len(points) >= 2 else list(points)
    points.append((int(evt.index[0]), int(evt.index[1])))
    return points, *_tracer(image, points, debruitage, algorithme, alpha)


def retracer(image, points, debruitage, algorithme, alpha):
    if image is None or len(points) < 2:
        return gr.skip(), gr.skip()
    return _tracer(image, points, debruitage, algorithme, alpha)


with gr.Blocks(title="Tracé des vaisseaux rétiniens") as demo:
    gr.Markdown(
        "# Tracé des vaisseaux rétiniens par plus court chemin\n"
        "Cliquez **deux points** sur un vaisseau de l'image de gauche : l'image est vue comme un graphe "
        "dont chaque pixel est un sommet, et l'algorithme de Dijkstra (ou A\\*) suit le vaisseau de l'un à "
        "l'autre. Un troisième clic recommence. "
        "[Code et évaluation sur GitHub](https://github.com/sheene123/retina-vessel-tracing)."
    )
    points = gr.State([])
    with gr.Row():
        entree = gr.Image(label="Fond d'œil : cliquez deux points", type="numpy", sources=["upload"])
        sortie = gr.Image(label="Tracé", type="numpy", interactive=False)
    infos = gr.Markdown("Chargez une image ou choisissez un exemple DRIVE ci-dessous.")
    with gr.Row():
        debruitage = gr.Dropdown(list(DEBRUITAGES), value="nl_means", label="Débruitage")
        algorithme = gr.Radio(["A*", "Dijkstra"], value="A*", label="Algorithme")
        alpha = gr.Slider(0.5, 4.0, value=ALPHA_DEFAUT, step=0.5, label="α (contraste du coût)")
    exemples = _exemples()
    if exemples:
        gr.Examples(exemples, inputs=entree, label="Images de test DRIVE (Staal et al., 2004)")

    entree.select(cliquer, [entree, points, debruitage, algorithme, alpha], [points, sortie, infos])
    entree.change(lambda: ([], None, "Cliquez le point de départ."), None, [points, sortie, infos])
    for controle in (debruitage, algorithme, alpha):
        controle.change(retracer, [entree, points, debruitage, algorithme, alpha], [sortie, infos])

if __name__ == "__main__":
    demo.launch()
