"""Pont entre la page web et le paquet `vaisseaux`, exécuté dans le navigateur par Pyodide.

Toutes les images restent dans le navigateur : rien n'est envoyé à un serveur.
"""

import base64
import io
import json
import time

import numpy as np
from PIL import Image

from vaisseaux.benchmark import CONFIGURATIONS
from vaisseaux.donnees import estimer_masque_fov
from vaisseaux.evaluation import auc_roc, evaluer_trace, metriques_binaires, reference_entre
from vaisseaux.graphe import carte_de_cout, plus_court_chemin
from vaisseaux.pretraitement import pretraiter

COTE_MAX = 800  # le calcul dans le navigateur est plus lent : on réduit les grandes images
NOMS = {
    "vert_brut": "Canal vert brut",
    "clahe": "CLAHE",
    "clahe_gaussien": "CLAHE + gaussien",
    "clahe_median": "CLAHE + médian",
    "clahe_bilateral": "CLAHE + bilatéral",
    "clahe_nl_means": "CLAHE + NL-means",
}
# seuils de binarisation choisis sur les images d'entraînement DRIVE (resultats/evaluation.json)
SEUILS = {
    "vert_brut": 0.06,
    "clahe": 0.08,
    "clahe_gaussien": 0.06,
    "clahe_median": 0.06,
    "clahe_bilateral": 0.02,
    "clahe_nl_means": 0.08,
}
_etat: dict = {}


def _png(tableau: np.ndarray) -> bytes:
    if tableau.dtype != np.uint8:
        tableau = (np.clip(tableau, 0, 1) * 255).astype(np.uint8)
    tampon = io.BytesIO()
    Image.fromarray(tableau).save(tampon, format="PNG")
    return tampon.getvalue()


def _ouvrir(octets) -> Image.Image:
    return Image.open(io.BytesIO(bytes(octets)))


def charger_image(octets, octets_verite=None) -> str:
    """Décode l'image (TIFF compris), la réduit si besoin. Renvoie un JSON avec la taille et
    la présence d'une vérité terrain."""
    image = _ouvrir(octets).convert("RGB")
    echelle = max(image.size) / COTE_MAX
    taille = (round(image.width / echelle), round(image.height / echelle)) if echelle > 1 else image.size
    image = image.resize(taille, Image.Resampling.LANCZOS) if echelle > 1 else image
    _etat.clear()
    _etat["rgb"] = np.asarray(image)
    _etat["masque"] = estimer_masque_fov(_etat["rgb"])
    _etat["cartes"] = {}
    if octets_verite is not None:
        verite = _ouvrir(octets_verite).convert("L").resize(taille, Image.Resampling.NEAREST)
        _etat["verite"] = np.asarray(verite) > 127
    return json.dumps({"largeur": taille[0], "hauteur": taille[1], "verite": "verite" in _etat})


def _cartes(config: str):
    if config not in _etat["cartes"]:
        debut = time.perf_counter()
        _etat["cartes"][config] = pretraiter(_etat["rgb"], _etat["masque"], CONFIGURATIONS[config])
        _etat.setdefault("durees", {})[config] = 1e3 * (time.perf_counter() - debut)
    return _etat["cartes"][config]


def vue(nom: str, config: str) -> bytes:
    """PNG de la vue demandée : originale, vert, rehaussee, vaisseaux ou verite."""
    if nom == "originale":
        return _png(_etat["rgb"])
    if nom == "verite" and "verite" in _etat:
        return _png(_etat["verite"].astype(float))
    cartes = _cartes(config)
    if nom == "vert":
        return _png(cartes.vert)
    if nom == "rehaussee":
        return _png(cartes.rehaussee)
    return _png(np.sqrt(cartes.vaisseaux))  # racine : rend visibles les capillaires


def _metriques_pixels(config: str) -> dict:
    if "verite" not in _etat:
        return {}
    v, verite, masque = _cartes(config).vaisseaux, _etat["verite"], _etat["masque"]
    binaire = (v >= SEUILS[config]) & masque
    m = metriques_binaires(binaire, verite, masque)
    return {"auc": auc_roc(v, verite, masque), "mcc": m["mcc"], "dice": m["dice"]}


def vignette(config: str) -> str:
    """Carte de vaisseaux d'une configuration, en data URL, avec ses scores si vérité connue."""
    png = base64.b64encode(vue("vaisseaux", config)).decode()
    return json.dumps(
        {
            "config": config,
            "nom": NOMS[config],
            "image": f"data:image/png;base64,{png}",
            "duree_ms": round(_etat["durees"][config]),
            **_metriques_pixels(config),
        }
    )


def tracer(x0: int, y0: int, x1: int, y1: int, configs_json: str, algorithme: str, alpha: float) -> str:
    """Trace le chemin pour chaque configuration demandée ; si la vérité terrain est connue,
    ajoute le chemin de référence de l'expert et les scores du tracé."""
    depart, arrivee = (int(y0), int(x0)), (int(y1), int(x1))
    reponse: dict = {"traces": []}
    reference = None
    if "verite" in _etat:
        reference = reference_entre(_etat["verite"], depart, arrivee)
        reponse["reference"] = None if reference is None else reference[:, ::-1].tolist()
    for config in json.loads(configs_json):
        debut = time.perf_counter()
        cartes = _cartes(config)
        milieu = time.perf_counter()
        cout = carte_de_cout(cartes.vaisseaux, _etat["masque"], float(alpha))
        try:
            chemin = plus_court_chemin(cout, depart, arrivee, a_etoile=algorithme == "a_etoile")
        except ValueError as erreur:
            reponse["traces"].append({"config": config, "nom": NOMS[config], "erreur": str(erreur)})
            continue
        trace = {
            "config": config,
            "nom": NOMS[config],
            "chemin": chemin.pixels[:, ::-1].tolist(),
            "longueur": round(chemin.longueur, 1),
            "pixels_visites": chemin.pixels_visites,
            "pretraitement_ms": round(1e3 * (milieu - debut)),
            "recherche_ms": round(1e3 * (time.perf_counter() - milieu)),
        }
        if reference is not None:
            m = evaluer_trace(chemin.pixels, reference, _etat["verite"])
            trace.update({k: round(m[k], 3) for k in ("precision", "couverture", "f1", "frechet")})
        reponse["traces"].append(trace)
    return json.dumps(reponse)
