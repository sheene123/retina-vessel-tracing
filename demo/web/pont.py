"""Pont entre la page web et le paquet `vaisseaux`, exécuté dans le navigateur par Pyodide.

Toutes les images restent dans le navigateur : rien n'est envoyé à un serveur.
"""

import base64
import io
import json
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from vaisseaux.benchmark import CONFIGURATIONS
from vaisseaux.biomarqueurs import MARQUEURS, mesurer
from vaisseaux.donnees import cadre_carre, carre_fond_oeil, carte_sur_image, entree_imagenet, isoler_fond_oeil
from vaisseaux.evaluation import auc_roc, evaluer_trace, metriques_binaires, reference_entre
from vaisseaux.graphe import carte_de_cout, plus_court_chemin
from vaisseaux.pretraitement import normaliser_pour_reseau, pretraiter
from vaisseaux.qualite import evaluer as evaluer_qualite
from vaisseaux.zones import MESURES_ZONES, ZONE_B, ZONE_C, anneau, mesurer_zones, papille

MESURES_AV_AFFICHEES = ("crae_um", "crve_um")  # l'AVR n'est pas assez fiable (docs/mesures_zones.md)

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
    """Décode l'image (TIFF compris), repère le fond d'œil, recadre dessus et met en noir tout ce
    qui n'est pas l'œil (texte, schémas, fond de page), puis réduit si besoin. Renvoie un JSON
    avec la taille, la présence d'une vérité terrain et ce qui a été fait."""
    image = np.asarray(_ouvrir(octets).convert("RGB"))
    hauteur_origine, largeur_origine = image.shape[:2]
    image, oeil, cadre = isoler_fond_oeil(image)
    if cadre is None:
        couleur = np.abs(image.astype(np.int16) - image.mean(axis=-1, keepdims=True)).mean()
        if couleur < 6:  # canaux quasi identiques : angiographie, photo en noir et blanc
            return json.dumps(
                {
                    "erreur": "cette image est en noir et blanc (angiographie ou photo désaturée) : "
                    "l'outil attend une photo couleur du fond d'œil"
                }
            )
        return json.dumps({"erreur": "aucun fond d'œil détecté dans cette image"})
    haut, bas, gauche, droite = cadre
    recadre = (bas - haut) * (droite - gauche) < 0.9 * hauteur_origine * largeur_origine
    verite = None
    if octets_verite is not None:
        verite = np.asarray(_ouvrir(octets_verite).convert("L"))[haut:bas, gauche:droite] > 127

    echelle = max(image.shape[:2]) / COTE_MAX
    taille = (image.shape[1], image.shape[0])
    if echelle > 1:
        taille = (round(image.shape[1] / echelle), round(image.shape[0] / echelle))
        image = np.asarray(Image.fromarray(image).resize(taille, Image.Resampling.LANCZOS))
        oeil = np.asarray(Image.fromarray(oeil).resize(taille, Image.Resampling.NEAREST))
        if verite is not None:
            verite = np.asarray(Image.fromarray(verite).resize(taille, Image.Resampling.NEAREST))
    _etat.clear()
    _etat["rgb"] = image
    _etat["masque"] = oeil
    _etat["cartes"] = {}
    if verite is not None:
        _etat["verite"] = verite
    return json.dumps(
        {
            "qualite": evaluer_qualite(image),
            "largeur": taille[0],
            "hauteur": taille[1],
            "verite": "verite" in _etat,
            "recadre": bool(recadre),
            "part_image": round((bas - haut) * (droite - gauche) / (hauteur_origine * largeur_origine), 3),
        }
    )


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
    besoins = {
        "segmentation": "segmentation",
        "verite": "verite",
        "chaleur": "chaleur",
        "zones": "disque",
        "arteres": "arteres",
    }
    if nom in besoins and besoins[nom] not in _etat:
        return _png(_etat["rgb"])  # vue pas encore disponible : jamais une autre vue à sa place
    if nom == "segmentation" and "segmentation" in _etat:
        return _png(_etat["segmentation"].astype(float))
    if nom == "verite" and "verite" in _etat:
        return _png(_etat["verite"].astype(float))
    if nom == "chaleur" and "chaleur" in _etat:
        return _png(_etat["chaleur"])
    if nom == "zones" and "disque" in _etat:
        return _png(_vue_zones())
    if nom == "arteres" and "arteres" in _etat:
        return _png(_vue_arteres())
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


def _message(erreur: ValueError) -> str:
    """Traduit les erreurs du graphe en consignes compréhensibles."""
    texte = str(erreur)
    if "hors du champ" in texte:
        return "un des points est en dehors de la rétine visible : cliquez à l'intérieur du disque."
    if "aucun chemin" in texte:
        return "aucun chemin ne relie ces deux points dans la rétine visible."
    return texte


def preparer(config: str) -> int:
    """Calcule (ou reprend du cache) les cartes d'une configuration ; renvoie la durée en ms."""
    deja = config in _etat["cartes"]
    _cartes(config)
    return 0 if deja else round(_etat["durees"][config])


def accrocher(x: int, y: int, config: str, aimanter: bool = True, rayon: int = 6) -> str:
    """Point retenu pour un clic. Avec l'aimantation, on garde le pixel cliqué s'il est déjà sur un
    vaisseau ; sinon on prend le pixel le plus « vaisseau » tout proche, avec une forte préférence
    pour la distance afin de ne pas sauter sur un gros vaisseau voisin."""
    masque = _etat["masque"]
    hauteur, largeur = masque.shape
    x, y = int(np.clip(x, 0, largeur - 1)), int(np.clip(y, 0, hauteur - 1))
    if not aimanter:
        return json.dumps({"dans_fov": bool(masque[y, x]), "x": x, "y": y})
    vaisseaux = _cartes(config).vaisseaux
    l0, l1, c0, c1 = max(0, y - rayon), min(hauteur, y + rayon + 1), max(0, x - rayon), min(largeur, x + rayon + 1)
    fenetre = vaisseaux[l0:l1, c0:c1]
    lignes, colonnes = np.mgrid[l0:l1, c0:c1]
    d2 = (lignes - y) ** 2 + (colonnes - x) ** 2
    valides = masque[l0:l1, c0:c1] & (d2 <= rayon**2)
    if not valides.any():
        return json.dumps({"dans_fov": False, "x": x, "y": y})
    if masque[y, x] and vaisseaux[y, x] >= 0.6 * fenetre[valides].max():
        return json.dumps({"dans_fov": True, "x": x, "y": y})
    score = np.where(valides, fenetre * np.exp(-d2 / (2 * 2.0**2)), -1.0)
    i = np.unravel_index(int(np.argmax(score)), score.shape)
    return json.dumps({"dans_fov": True, "x": int(colonnes[i]), "y": int(lignes[i])})


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
            reponse["traces"].append({"config": config, "nom": NOMS[config], "erreur": _message(erreur)})
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


# ---------------------------------------------------------------- mesures vasculaires
# Le U-Net est exécuté par ONNX Runtime Web côté JavaScript : Python prépare l'entrée, reçoit la
# carte de probabilité et calcule les marqueurs.

SEUIL_UNET = 0.5
FIABILITE: dict = {}  # rempli par la page depuis fiabilite.json (erreur du U-Net / écart entre patients)


FIABILITE_ZONES: dict = {}  # accord U-Net / expert des mesures en zones (resultats/fiabilite_zones.json)


def definir_fiabilite_zones(texte_json: str) -> None:
    FIABILITE_ZONES.clear()
    FIABILITE_ZONES.update(json.loads(texte_json).get("mesures", {}))


REPERES: dict = {}  # repères yeux sains des mesures qui passent le contrôle du biais d'appareil


def definir_reperes(texte_json: str) -> None:
    """Repères « yeux sains » (resultats/reperes_zones.json). Une mesure n'est gardée que si les
    yeux des autres appareils (parties test de DRIVE_AV, HRF-AV, LES-AV) tombent, en médiane,
    entre les 25e et 75e percentiles des yeux sains : sinon le repère mesurerait l'appareil."""
    donnees = json.loads(texte_json)
    REPERES.clear()
    autres = donnees.get("autres_appareils", {})
    for k, valeurs in donnees.get("valeurs", {}).items():
        rangs = [b[k]["rang_centile_median"] for b in autres.values() if k in b]
        if len(rangs) == 3 and all(25 <= r <= 75 for r in rangs):
            REPERES[k] = np.sort(np.asarray(valeurs, dtype=float))


def definir_fiabilite(texte_json: str) -> None:
    FIABILITE.clear()
    FIABILITE.update(json.loads(texte_json))


def entree_unet() -> bytes:
    """Image normalisée et complétée à un multiple de 32, en float32 (3, H, W)."""
    x = normaliser_pour_reseau(_etat["rgb"], _etat["masque"], 32)
    _etat["forme_unet"] = x.shape[1:]
    return x.tobytes()


def forme_unet() -> str:
    return json.dumps(list(_etat["forme_unet"]))


def recevoir_unet(octets, octets_arteres=None) -> None:
    """Logits du U-Net (H, W complétés) -> probabilité, puis segmentation dans le champ de vue.
    `octets_arteres` : deuxième sortie du U-Net multi-appareils (artère plutôt que veine)."""
    hauteur, largeur = _etat["masque"].shape

    def proba_de(o):
        logits = np.frombuffer(bytes(o), dtype=np.float32).reshape(_etat["forme_unet"])[:hauteur, :largeur]
        return 1.0 / (1.0 + np.exp(-logits))

    proba = proba_de(octets)
    _etat["proba"] = proba
    _etat.pop("arteres", None)
    if octets_arteres is not None:
        _etat["arteres"] = proba_de(octets_arteres)
    _etat["segmentation"] = (proba >= SEUIL_UNET) & _etat["masque"]


def _nombre(valeur) -> float | None:
    valeur = float(valeur)
    return valeur if np.isfinite(valeur) else None


def mesures() -> str:
    """Marqueurs vasculaires de la segmentation U-Net ; avec une image annotée, valeurs de
    l'expert et écart relatif."""
    masque = _etat["masque"]
    mesure = mesurer(_etat["segmentation"], masque)
    expert = mesurer(_etat["verite"], masque) if "verite" in _etat else None
    reponse: dict = {"marqueurs": {}, "fiabilite_disponible": bool(FIABILITE)}
    for k in MARQUEURS:
        ligne = {"valeur": _nombre(mesure[k]), "fiabilite": FIABILITE.get(k)}
        if expert is not None:
            ligne["expert"] = _nombre(expert[k])
            ligne["ecart_relatif"] = _nombre(abs(mesure[k] - expert[k]) / abs(expert[k])) if expert[k] else None
        reponse["marqueurs"][k] = ligne
    if expert is not None:
        reponse["dice"] = metriques_binaires(_etat["segmentation"], _etat["verite"], masque)["dice"]
    reponse |= _mesures_zones()
    return json.dumps(reponse)


def _mesures_zones() -> dict:
    """Mesures dans les zones autour de la papille, comparables d'une photo à l'autre. La marge
    vient du seuil du U-Net (0,4 et 0,6 au lieu de 0,5) : une mesure qui bouge beaucoup avec ce
    seuil dépend fortement de la segmentation."""
    masque, segmentation = _etat["masque"], _etat["segmentation"]
    disque = papille(_etat["rgb"], masque, segmentation)
    _etat["disque"] = disque
    arteres = _etat.get("arteres")
    valeur = mesurer_zones(segmentation, disque, arteres)
    variantes = [mesurer_zones((_etat["proba"] >= s) & masque, disque, arteres) for s in (0.4, 0.6)]
    expert = mesurer_zones(_etat["verite"] & masque, disque) if "verite" in _etat else None
    zones = {}
    for k in MESURES_ZONES + (MESURES_AV_AFFICHEES if arteres is not None else ()):
        # la marge englobe la valeur elle-même et les deux variantes de seuil
        bornes = [v[k] for v in [*variantes, valeur] if np.isfinite(v[k])]
        zones[k] = {
            "valeur": _nombre(valeur[k]),
            "bas": _nombre(min(bornes)) if bornes else None,
            "haut": _nombre(max(bornes)) if bornes else None,
            "expert": _nombre(expert[k]) if expert is not None and k in expert else None,
            "fiabilite": FIABILITE_ZONES.get(k),
        }
        if k in REPERES and np.isfinite(valeur[k]):
            # rang parmi les yeux sains de référence (0 à 100)
            zones[k]["centile_sains"] = round(100 * np.searchsorted(REPERES[k], valeur[k]) / len(REPERES[k]))
    return {
        "zones": zones,
        "papille": {
            "diametre_px": round(disque["diametre"], 1),
            "estime": disque["diametre_estime"],
            "champ_dp": round(disque["champ_diametre"] / disque["diametre"], 2),
        },
    }


def _vue_zones() -> np.ndarray:
    """Papille (vert), zone B (jaune), limite de la zone C (cyan) et vaisseaux de la zone C colorés
    selon leur calibre, du bleu (fins) au rouge (gros), sur l'image assombrie."""
    from PIL import ImageDraw
    from skimage.morphology import skeletonize

    rgb, disque, segmentation = _etat["rgb"], _etat["disque"], _etat["segmentation"]
    fond = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
    zone_c = anneau(segmentation.shape, disque, ZONE_C)
    squelette = skeletonize(segmentation & zone_c)
    largeur = 2 * ndi.distance_transform_edt(segmentation)[squelette] - 1
    t = np.clip(largeur / max(np.percentile(largeur, 95), 1.0), 0, 1) if largeur.size else largeur
    couleurs = np.stack([255 * t, 80 + 60 * (1 - np.abs(2 * t - 1)), 255 * (1 - t)], axis=-1).astype(np.uint8)
    epais = ndi.binary_dilation(squelette, iterations=1)
    lignes, colonnes = np.nonzero(squelette)
    image = fond.copy()
    image[epais] = (200, 200, 200)
    image[lignes, colonnes] = couleurs
    dessin_image = Image.fromarray(image)
    dessin = ImageDraw.Draw(dessin_image)
    cx, cy, d = disque["colonne"], disque["ligne"], disque["diametre"]
    epaisseur = max(2, round(min(rgb.shape[:2]) / 250))
    for rayon, couleur in (
        (0.5, (80, 220, 80)),
        (ZONE_B[0], (240, 220, 60)),
        (ZONE_B[1], (240, 220, 60)),
        (ZONE_C[1], (80, 210, 230)),
    ):
        dessin.ellipse(
            [cx - rayon * d, cy - rayon * d, cx + rayon * d, cy + rayon * d], outline=couleur, width=epaisseur
        )
    return np.asarray(dessin_image)


def _vue_arteres() -> np.ndarray:
    """Vaisseaux du U-Net colorés : artères en rouge, veines en bleu (plus pâle si le réseau hésite),
    sur l'image assombrie."""
    seg, p = _etat["segmentation"], _etat["arteres"]
    image = (_etat["rgb"].astype(np.float32) * 0.35).astype(np.uint8)
    certitude = np.abs(2 * p - 1)  # 0 : hésite, 1 : sûr
    rouge = np.stack([np.full_like(p, 255), 60 * (1 - certitude), 60 * (1 - certitude)], axis=-1)
    bleu = np.stack([60 * (1 - certitude), 110 + 30 * (1 - certitude), np.full_like(p, 255)], axis=-1)
    couleur = np.where((p >= 0.5)[..., None], rouge, bleu).astype(np.uint8)
    image[seg] = couleur[seg]
    return image


# ---------------------------------------------------------------- troubles de l'œil
# Le réseau (ONNX) tourne côté JavaScript ; Python prépare l'image comme à l'entraînement.


def entree_troubles() -> bytes:
    """Fond d'œil recadré en carré 384 × 384, normalisé ImageNet, float32 (3, 384, 384)."""
    _etat["cadre_troubles"] = cadre_carre(_etat["rgb"])
    return entree_imagenet(carre_fond_oeil(_etat["rgb"])).tobytes()


def definir_carte(octets, cote: int, indice: int) -> None:
    """Carte de chaleur d'un trouble (sortie « cartes » du réseau, float32 (6, cote, cote)) :
    contribution de chaque zone au score. On garde la partie qui fait monter le score, normalisée,
    et on la superpose à l'image en rouge et jaune ; le reste de l'image est assombri."""
    cartes = np.frombuffer(bytes(octets), dtype=np.float32).reshape(-1, cote, cote)
    carte = carte_sur_image(cartes[indice], _etat["rgb"].shape[:2], _etat["cadre_troubles"])
    oeil = _etat["masque"]
    valeurs = carte[oeil]
    bas, haut = np.percentile(valeurs, 50), np.percentile(valeurs, 99.5)
    intensite = np.clip((carte - bas) / max(haut - bas, 1e-6), 0, 1) * oeil
    couleur = np.stack([np.ones_like(intensite), intensite, np.zeros_like(intensite)], axis=-1)  # rouge vers jaune
    alpha = (0.65 * intensite)[..., None]
    fond = _etat["rgb"].astype(np.float32) / 255 * 0.55
    _etat["chaleur"] = np.clip(fond * (1 - alpha) + couleur * alpha, 0, 1)
