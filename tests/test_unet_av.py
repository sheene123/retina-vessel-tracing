import numpy as np
import pytest
from test_zones import _oeil_avec_papille

from vaisseaux.zones import calibres_av, papille, segments_zone


def test_calibres_av_classe_les_segments():
    rgb, vaisseaux = _oeil_avec_papille()
    disque = papille(rgb, None, vaisseaux)
    segs = segments_zone(vaisseaux, disque)
    # moitié des vaisseaux (selon l'angle autour de la papille) en artères
    carte = np.full(vaisseaux.shape, np.nan)
    for s in segs:
        carte[s["lignes"], s["colonnes"]] = float(np.cos(s["angle"]) > 0)
    r = calibres_av(segs, carte, disque["diametre"])
    assert r["crae_um"] > 0 and r["crve_um"] > 0
    assert r["avr"] == pytest.approx(r["crae_um"] / r["crve_um"], rel=1e-6)
    # rien de renseigné : ni CRAE ni CRVE
    vide = calibres_av(segs, np.full(vaisseaux.shape, np.nan), disque["diametre"])
    assert np.isnan(vide["crae_um"]) and np.isnan(vide["avr"])


def test_unet_av_formes_et_augmentation():
    torch = pytest.importorskip("torch")
    from vaisseaux import unet_av

    modele = unet_av.UNetAV()
    assert modele(torch.zeros(1, 3, 64, 96)).shape == (1, 2, 64, 96)
    rng = np.random.default_rng(0)
    rgb, vaisseaux = _oeil_avec_papille()
    etiquette = np.where(vaisseaux, unet_av.ARTERE, unet_av.FOND).astype(np.uint8)
    masque = rgb[..., 0] > 0
    x, lab, msk = unet_av.exemple_augmente(rgb, etiquette, masque, rng, taille=128)
    assert x.shape == (3, 128, 128) and x.dtype == np.float32 and np.isfinite(x).all()
    assert lab.shape == msk.shape == (128, 128)
    assert set(np.unique(lab)) <= {unet_av.FOND, unet_av.ARTERE}
    assert (x[:, ~msk] == 0).all()  # hors champ de vue : zéro, comme dans la démo
    vaisseau, artere = unet_av.predire(modele, rgb, masque)
    assert vaisseau.shape == artere.shape == masque.shape


def test_largeurs_profil_vaisseau_oblique():
    from vaisseaux.zones import largeurs_profil

    lignes, colonnes = np.mgrid[:80, :80]
    distance = np.abs((lignes - 40) - 0.5 * (colonnes - 40)) / np.hypot(1, 0.5)
    carte = (distance <= 2.5).astype(float)  # vaisseau oblique de 5 pixels de large
    ls = np.arange(25, 56)
    cs = 40 + 2 * (ls - 40)
    garde = (cs >= 0) & (cs < 80)
    largeur = np.median(largeurs_profil(carte, ls[garde], cs[garde]))
    assert 4.5 < largeur < 6.0  # au demi-pixel près (bord pixelisé du masque)
