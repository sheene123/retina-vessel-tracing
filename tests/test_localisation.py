import numpy as np

from vaisseaux.donnees import localiser_fond_oeil


def test_trouve_l_oeil_dans_une_page():
    # page blanche avec du texte noir et, à droite, un fond d'œil orangé sur fond noir
    page = np.full((400, 700, 3), 255, dtype=np.uint8)
    page[40:60, 30:250] = 0  # « texte »
    page[20:380, 330:690] = 0  # cadre noir de la photo
    lignes, colonnes = np.mgrid[:400, :700]
    disque = (lignes - 200) ** 2 + (colonnes - 510) ** 2 < 170**2
    page[disque] = (220, 90, 30)
    page[(lignes - 200) ** 2 + (colonnes - 600) ** 2 < 20**2] = (250, 240, 170)  # disque optique jaune
    (haut, bas, gauche, droite), masque = localiser_fond_oeil(page)
    assert abs(haut - 30) <= 12 and abs(bas - 370) <= 12
    assert abs(gauche - 340) <= 12 and abs(droite - 680) <= 12
    assert masque.mean() > 0.7


def test_aucun_oeil():
    page = np.full((300, 300, 3), 255, dtype=np.uint8)
    page[100:120, 20:280] = 0
    assert localiser_fond_oeil(page) is None
