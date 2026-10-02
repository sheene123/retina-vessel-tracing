"""Mesures vasculaires dans les zones standard autour de la papille, en unités comparables.

Les logiciels de recherche en oculomique (SIVA, VAMPIRE, AutoMorph) ne mesurent pas les vaisseaux
sur toute l'image mais dans des anneaux centrés sur la papille, et expriment les longueurs en
diamètres de papille (DP) : ainsi, deux photos du même œil prises avec des appareils, des cadrages
ou des résolutions différents donnent des mesures comparables.

- Zone B : de 0,5 à 1 DP du bord de la papille, soit de 1 à 1,5 DP de son centre ; c'est là qu'on
  mesure le calibre des artérioles et des veinules (CRAE, CRVE, AVR).
- Zone C : de 0,5 à 2 DP du bord, soit de 1 à 2,5 DP du centre ; on y mesure tortuosité et densité.
- Unités : 1 DP vaut en moyenne environ 1 800 µm chez l'adulte ; les valeurs en µm sont donc des
  estimations (la taille réelle de la papille varie d'une personne à l'autre).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

DIAMETRE_PAPILLE_UM = 1800.0
ZONE_B = (1.0, 1.5)  # en diamètres de papille depuis le centre
ZONE_C = (1.0, 2.5)


def _champ(rgb: np.ndarray) -> np.ndarray:
    champ = ndi.binary_fill_holes(ndi.binary_opening(rgb[..., 0] > 20, iterations=2))
    etiquettes, n = ndi.label(champ)
    if n > 1:
        champ = etiquettes == np.argmax(np.bincount(etiquettes.ravel())[1:]) + 1
    return champ


def papille(rgb: np.ndarray, masque: np.ndarray | None = None, vaisseaux: np.ndarray | None = None) -> dict:
    """Centre et diamètre de la papille, en pixels de l'image.

    Centre : zone claire (rouge et vert) où convergent les gros vaisseaux, si une segmentation est
    fournie ; la clarté seule se laisse tromper par un bord surexposé. Diamètre : profil radial de
    la clarté (médiane par anneau, pour ignorer les vaisseaux qui traversent la papille) ; le bord
    est l'endroit où la clarté redescend au quart de la hauteur entre le fond et le pic, ce qui
    englobe l'anneau plus pâle de la papille. Hors des proportions habituelles (entre 1/12 et 1/4
    du champ de vue), on retient 1/7 du champ de vue, valeur typique à 45°, et on le signale.
    """
    x = rgb[..., :3].astype(np.float32)
    masque = _champ(rgb) if masque is None else masque
    champ_diametre = 2 * np.sqrt(masque.sum() / np.pi)
    interieur = ndi.binary_erosion(masque, iterations=max(2, int(champ_diametre / 25)))
    clarte = 0.5 * x[..., 0] + x[..., 1]

    def centree(carte: np.ndarray) -> np.ndarray:
        valeurs = carte[interieur]
        return (carte - valeurs.mean()) / (valeurs.std() + 1e-6)

    score = centree(ndi.gaussian_filter(clarte, sigma=max(2.0, champ_diametre / 40)))
    if vaisseaux is not None and vaisseaux.any():
        # gros vaisseaux : ceux qui résistent à une ouverture morphologique
        gros = ndi.binary_opening(vaisseaux & masque, iterations=max(1, int(champ_diametre / 200)))
        score = score + centree(ndi.gaussian_filter(gros.astype(np.float32), sigma=max(2.0, champ_diametre / 25)))
    score[~interieur] = -np.inf
    ligne, colonne = np.unravel_index(int(np.argmax(score)), score.shape)
    # affinage : la convergence des vaisseaux est parfois un peu à côté du disque clair ; on prend
    # le point le plus clair dans un petit voisinage
    rayon = int(champ_diametre / 14)
    h0, g0 = max(0, ligne - rayon), max(0, colonne - rayon)
    voisinage = ndi.gaussian_filter(clarte, sigma=max(1.5, champ_diametre / 60))[
        h0 : ligne + rayon, g0 : colonne + rayon
    ]
    voisinage = np.where(interieur[h0 : ligne + rayon, g0 : colonne + rayon], voisinage, -np.inf)
    dl, dc = np.unravel_index(int(np.argmax(voisinage)), voisinage.shape)
    ligne, colonne = h0 + int(dl), g0 + int(dc)

    # diamètre : profil radial de la clarté (médiane par anneau d'épaisseur 2 px)
    lisse = ndi.gaussian_filter(clarte, sigma=max(1.0, champ_diametre / 200))
    lignes, colonnes = np.nonzero(masque)
    distances = np.hypot(lignes - ligne, colonnes - colonne)
    rayon_max = champ_diametre / 4
    proches = distances < rayon_max
    rangs = (distances[proches] // 2).astype(int)
    valeurs = lisse[lignes[proches], colonnes[proches]]
    ordre = np.argsort(rangs)
    rangs, valeurs = rangs[ordre], valeurs[ordre]
    coupures = np.flatnonzero(np.diff(rangs)) + 1
    profil = np.array([np.median(v) for v in np.split(valeurs, coupures)])
    rayons = 2.0 * np.unique(rangs) + 1
    fond = np.median(profil[rayons > rayon_max * 0.75]) if (rayons > rayon_max * 0.75).any() else profil[-1]
    pic = profil[rayons < champ_diametre / 40].max() if (rayons < champ_diametre / 40).any() else profil[0]
    sous = np.flatnonzero(profil < fond + 0.25 * (pic - fond))
    diametre = 2 * float(rayons[sous[0]]) if sous.size and pic > fond else 0.0
    estime = not (champ_diametre / 12 <= diametre <= champ_diametre / 4)
    if estime:
        diametre = champ_diametre / 7
    return {
        "ligne": int(ligne),
        "colonne": int(colonne),
        "diametre": float(diametre),
        "diametre_estime": bool(estime),
        "champ_diametre": float(champ_diametre),
    }


def anneau(forme: tuple[int, int], disque: dict, zone: tuple[float, float]) -> np.ndarray:
    """Masque de l'anneau `zone` (en diamètres de papille depuis le centre)."""
    lignes, colonnes = np.ogrid[: forme[0], : forme[1]]
    r = np.hypot(lignes - disque["ligne"], colonnes - disque["colonne"]) / disque["diametre"]
    return (r >= zone[0]) & (r < zone[1])


# ---------------------------------------------------------------- artérioles et veinules

KNUDTSON_ARTERES, KNUDTSON_VEINES = 0.88, 0.95  # formules révisées de Knudtson (2003)
NB_VAISSEAUX_KNUDTSON = 6


def largeurs_profil(
    carte: np.ndarray, lignes: np.ndarray, colonnes: np.ndarray, demi: float = 12.0, pas: float = 0.25
) -> np.ndarray:
    """Largeur au dixième de pixel en chaque point d'un segment : profil de la carte (probabilité
    « vaisseau » du U-Net, ou masque d'expert) en travers du segment, lu tous les quarts de pixel
    par interpolation, puis intégré sur la partie centrale (jusqu'à ce que le profil retombe sous
    0,05 de chaque côté). Une carte à bords nets (0 ou 1) donne exactement la largeur traversée ;
    une probabilité donne une largeur continue, sans le pas d'un pixel de la segmentation binaire."""
    points = np.stack([lignes, colonnes], axis=1).astype(np.float64)
    if len(points) < 2:
        return np.full(len(points), np.nan)
    centre = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    normale = np.array([-axes[0][1], axes[0][0]])  # perpendiculaire à l'axe principal du segment
    t = np.arange(-demi, demi + pas / 2, pas)
    coords = points[:, :, None] + normale[None, :, None] * t[None, None, :]  # (n, 2, k)
    profils = ndi.map_coordinates(
        carte.astype(np.float32), coords.transpose(1, 0, 2).reshape(2, -1), order=1, mode="constant"
    )
    profils = profils.reshape(len(points), -1)
    milieu = len(t) // 2
    sortie = np.empty(len(points))
    for i, v in enumerate(profils):
        bas = np.flatnonzero(v[:milieu] < 0.05)
        haut = np.flatnonzero(v[milieu:] < 0.05)
        debut = bas[-1] + 1 if bas.size else 0
        fin = milieu + haut[0] if haut.size else len(v)
        sortie[i] = np.clip(v[debut:fin], 0, 1).sum() * pas
    return sortie


def segments_zone(
    vaisseaux: np.ndarray, disque: dict, zone: tuple[float, float] = ZONE_B, carte: np.ndarray | None = None
) -> list[dict]:
    """Segments de vaisseau (entre bifurcations) qui traversent la zone : pixels du squelette dans
    la zone, longueur, largeur médiane et angle autour de la papille. Largeur : distance au fond de
    la segmentation binaire (au pixel près) ; avec `carte`, profil en travers du vaisseau (au
    dixième de pixel, voir `largeurs_profil`)."""
    from skimage.morphology import skeletonize

    from vaisseaux.biomarqueurs import calibres, segments

    squelette = skeletonize(vaisseaux)
    largeurs = np.zeros(vaisseaux.shape, np.float32)
    largeurs[squelette] = calibres(vaisseaux, squelette)
    dans_zone = anneau(vaisseaux.shape, disque, zone)
    etiquettes, _ = segments(squelette)
    etiquettes[~dans_zone] = 0
    longueur_min = max(4, int(0.08 * disque["diametre"]))
    resultat = []
    for k, fenetre in enumerate(ndi.find_objects(etiquettes), start=1):
        if fenetre is None:
            continue
        lignes, colonnes = np.nonzero(etiquettes[fenetre] == k)
        if lignes.size < longueur_min:
            continue
        lignes, colonnes = lignes + fenetre[0].start, colonnes + fenetre[1].start
        resultat.append(
            {
                "lignes": lignes,
                "colonnes": colonnes,
                "longueur": int(lignes.size),
                "largeur": float(
                    np.nanmedian(largeurs_profil(carte, lignes, colonnes))
                    if carte is not None
                    else np.median(largeurs[lignes, colonnes])
                ),
                "angle": float(np.arctan2(lignes.mean() - disque["ligne"], colonnes.mean() - disque["colonne"])),
            }
        )
    return resultat


def caracteristiques(rgb: np.ndarray, vaisseaux: np.ndarray, segs: list[dict]) -> np.ndarray:
    """Par segment : clarté du vaisseau rapportée au fond voisin, en rouge et en vert. Une artère,
    chargée de sang oxygéné, est plus claire (et souvent plus fine) qu'une veine."""
    x = rgb[..., :3].astype(np.float32) + 1.0
    sortie = np.zeros((len(segs), 3), np.float32)
    for i, s in enumerate(segs):
        r = int(np.ceil(max(2.0, s["largeur"]) * 2))
        h0, h1 = max(0, s["lignes"].min() - r), min(x.shape[0], s["lignes"].max() + r + 1)
        g0, g1 = max(0, s["colonnes"].min() - r), min(x.shape[1], s["colonnes"].max() + r + 1)
        axe = np.zeros((h1 - h0, g1 - g0), bool)
        axe[s["lignes"] - h0, s["colonnes"] - g0] = True
        coeur = ndi.binary_dilation(axe, iterations=max(1, int(s["largeur"] / 3))) & vaisseaux[h0:h1, g0:g1]
        fond = ndi.binary_dilation(axe, iterations=r) & ~ndi.binary_dilation(vaisseaux[h0:h1, g0:g1], iterations=2)
        if coeur.sum() == 0 or fond.sum() < 5:
            sortie[i] = np.nan
            continue
        v, f = x[h0:h1, g0:g1][coeur], x[h0:h1, g0:g1][fond]
        sortie[i] = (
            np.median(v[:, 0]) / np.median(f[:, 0]),
            np.median(v[:, 1]) / np.median(f[:, 1]),
            s["largeur"],
        )
    return sortie


def _deux_groupes(valeurs: np.ndarray) -> np.ndarray:
    """k-moyennes à deux groupes sur des caractéristiques centrées-réduites ; renvoie True pour le
    groupe le plus clair (artères)."""
    z = (valeurs - valeurs.mean(axis=0)) / (valeurs.std(axis=0) + 1e-6)
    clarte = z[:, :2].mean(axis=1)
    groupe = clarte > np.median(clarte)
    for _ in range(20):
        if groupe.all() or not groupe.any():
            break
        c1, c0 = z[groupe].mean(axis=0), z[~groupe].mean(axis=0)
        nouveau = np.linalg.norm(z - c1, axis=1) < np.linalg.norm(z - c0, axis=1)
        if (nouveau == groupe).all():
            break
        groupe = nouveau
    if groupe.any() and (~groupe).any() and clarte[groupe].mean() < clarte[~groupe].mean():
        groupe = ~groupe
    return groupe


def classer_arteres(carac: np.ndarray, angles: np.ndarray, par_quadrant: bool = True) -> np.ndarray:
    """True pour les segments classés artères. Par quadrant autour de la papille (l'éclairage varie
    d'un côté à l'autre de l'image), avec repli sur l'image entière si un quadrant a trop peu de
    segments ; seules la clarté rouge et verte servent (la largeur varie trop le long d'un vaisseau)."""
    valides = np.isfinite(carac).all(axis=1)
    arteres = np.zeros(len(carac), bool)
    if valides.sum() < 2:
        return arteres
    tout = np.zeros(len(carac), bool)
    tout[valides] = _deux_groupes(carac[valides][:, :2])
    if not par_quadrant:
        return tout
    quadrant = ((angles + np.pi / 4) % (2 * np.pi) // (np.pi / 2)).astype(int)
    for q in range(4):
        dans = valides & (quadrant == q)
        arteres[dans] = _deux_groupes(carac[dans][:, :2]) if dans.sum() >= 4 else tout[dans]
    return arteres


def knudtson(largeurs: list[float] | np.ndarray, coefficient: float) -> float:
    """Calibre équivalent central (CRAE ou CRVE) : les 6 plus gros vaisseaux, combinés deux à deux
    (le plus large avec le plus fin) par w = c √(w1² + w2²), jusqu'à une seule valeur."""
    w = sorted(largeurs, reverse=True)[:NB_VAISSEAUX_KNUDTSON]
    if len(w) < 2:
        return float("nan")
    while len(w) > 1:
        w = sorted(w, reverse=True)
        combines = [coefficient * np.hypot(w[i], w[-1 - i]) for i in range(len(w) // 2)]
        if len(w) % 2:
            combines.append(w[len(w) // 2])
        w = combines
    return float(w[0])


# ---------------------------------------------------------------- mesures dans les zones

MESURES_ZONES = ("calibre_zone_b", "calibre_zone_b_um", "tortuosite_zone_c", "densite_zone_c", "longueur_zone_c")
MESURES_AV = ("crae_um", "crve_um", "avr")
NB_MIN_KNUDTSON = 3  # en dessous de 3 artères (ou veines) mesurables, CRAE (ou CRVE) n'est pas calculé


def calibres_av(segs: list[dict], arteres: np.ndarray, diametre_papille: float) -> dict[str, float]:
    """CRAE, CRVE (µm estimés) et AVR à partir des segments de la zone B. `arteres` donne, pour
    chaque pixel, la probabilité « artère plutôt que veine » (NaN là où on ne sait pas) : un segment
    est une artère si la moyenne sur son squelette dépasse 0,5 ; il est ignoré si moins de la moitié
    de ses pixels sont renseignés."""
    largeurs = {True: [], False: []}
    for s in segs:
        v = arteres[s["lignes"], s["colonnes"]]
        connus = np.isfinite(v)
        if connus.mean() < 0.5:
            continue
        largeurs[bool(v[connus].mean() >= 0.5)].append(s["largeur"])
    crae = knudtson(largeurs[True], KNUDTSON_ARTERES) if len(largeurs[True]) >= NB_MIN_KNUDTSON else float("nan")
    crve = knudtson(largeurs[False], KNUDTSON_VEINES) if len(largeurs[False]) >= NB_MIN_KNUDTSON else float("nan")
    echelle = DIAMETRE_PAPILLE_UM / diametre_papille
    return {"crae_um": crae * echelle, "crve_um": crve * echelle, "avr": crae / crve}


def mesurer_zones(
    vaisseaux: np.ndarray, disque: dict, arteres: np.ndarray | None = None, carte: np.ndarray | None = None
) -> dict[str, float]:
    """Mesures comparables d'une photo à l'autre :

    - calibre_zone_b : largeur moyenne des 6 plus gros vaisseaux qui traversent la zone B, en
      diamètres de papille (et en µm estimés, 1 DP ≈ 1 800 µm) ;
    - tortuosite_zone_c : tortuosité pondérée (longs segments) dans la zone C ;
    - densite_zone_c : part de la zone C couverte par les vaisseaux ;
    - longueur_zone_c : longueur des vaisseaux par surface de la zone C, en DP⁻¹ (sans unité de
      pixel : même valeur quelle que soit la résolution) ;
    - avec `arteres` (probabilité « artère » par pixel, sortie du U-Net multi-appareils) : CRAE,
      CRVE et AVR (voir `calibres_av`). Le classement sans apprentissage n'est pas assez fiable
      pour cela (docs/mesures_zones.md).
    """
    from skimage.morphology import skeletonize

    from vaisseaux.biomarqueurs import tortuosite_ponderee

    dp = disque["diametre"]
    segs = segments_zone(vaisseaux, disque, ZONE_B, carte)
    largeurs = sorted((s["largeur"] for s in segs), reverse=True)[:NB_VAISSEAUX_KNUDTSON]
    calibre = float(np.mean(largeurs)) / dp if len(largeurs) >= 3 else float("nan")
    zone_c = anneau(vaisseaux.shape, disque, ZONE_C)
    dans_c = vaisseaux & zone_c
    squelette_c = skeletonize(dans_c)
    surface = zone_c.sum()
    av = calibres_av(segs, arteres, dp) if arteres is not None else {}
    return av | {
        "calibre_zone_b": calibre,
        "calibre_zone_b_um": calibre * DIAMETRE_PAPILLE_UM,
        "tortuosite_zone_c": tortuosite_ponderee(squelette_c, longueur_min=max(20, int(0.4 * dp))),
        "densite_zone_c": float(dans_c.sum() / surface) if surface else float("nan"),
        "longueur_zone_c": float(squelette_c.sum() / surface * dp) if surface else float("nan"),
    }
