import numpy as np

from vaisseaux.zones import ZONE_B, ZONE_C, anneau, knudtson, mesurer_zones, papille


def _oeil_avec_papille() -> tuple[np.ndarray, np.ndarray]:
    lignes, colonnes = np.mgrid[:500, :520]
    image = np.zeros((500, 520, 3), np.float32)
    champ = (lignes - 250) ** 2 + (colonnes - 260) ** 2 < 240**2
    image[champ] = (0.75, 0.32, 0.12)
    papille_vraie = (lignes - 250) ** 2 + (colonnes - 360) ** 2 < 32**2  # diamètre 64 px
    image[papille_vraie] = (0.98, 0.85, 0.55)
    vaisseaux = np.zeros((500, 520), bool)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):  # vaisseaux qui partent de la papille
        for r in range(20, 230):
            y, x = int(250 + r * np.sin(angle)), int(360 + r * np.cos(angle))
            if 0 <= y < 500 and 0 <= x < 520 and champ[y, x]:
                vaisseaux[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = True
    image[vaisseaux & champ] *= 0.55
    return (image * 255).astype(np.uint8), vaisseaux & champ


def test_knudtson():
    assert abs(knudtson([10, 10], 0.88) - 0.88 * np.hypot(10, 10)) < 1e-9
    assert np.isnan(knudtson([10], 0.88))
    # 6 vaisseaux : on combine le plus large avec le plus fin, jusqu'à une seule valeur
    assert knudtson([12, 11, 10, 9, 8, 7], 0.95) > 12


def test_papille_et_zones():
    image, vaisseaux = _oeil_avec_papille()
    disque = papille(image, vaisseaux=vaisseaux)
    assert abs(disque["ligne"] - 250) < 12 and abs(disque["colonne"] - 360) < 12
    assert 40 < disque["diametre"] < 95
    b, c = anneau(image.shape[:2], disque, ZONE_B), anneau(image.shape[:2], disque, ZONE_C)
    assert b.any() and c.sum() > b.sum() and not (b & ~c).any()  # la zone B est incluse dans la zone C


def test_mesures_sans_unite_de_pixel():
    image, vaisseaux = _oeil_avec_papille()
    disque = papille(image, vaisseaux=vaisseaux)
    m = mesurer_zones(vaisseaux, disque)
    assert 0 < m["densite_zone_c"] < 1 and m["tortuosite_zone_c"] >= 1
    # la même image deux fois plus grande donne le même calibre en diamètres de papille
    grande = np.kron(vaisseaux, np.ones((2, 2), bool))
    disque2 = {
        **disque,
        "ligne": 2 * disque["ligne"],
        "colonne": 2 * disque["colonne"],
        "diametre": 2 * disque["diametre"],
    }
    m2 = mesurer_zones(grande, disque2)
    assert abs(m2["calibre_zone_b"] - m["calibre_zone_b"]) / m["calibre_zone_b"] < 0.2
