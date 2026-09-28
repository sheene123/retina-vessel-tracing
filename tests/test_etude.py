import numpy as np
import pytest
from conftest import fabriquer_retine

from vaisseaux.etude import CIBLES, METRIQUES, PLUS_EST_MIEUX, analyser, metriques, perturbations


@pytest.fixture(scope="module")
def retine():
    return fabriquer_retine(taille=128)


def test_perturbations_isolent_un_type_d_erreur(retine):
    _, masque, verite = retine
    p = perturbations(verite, masque, np.random.default_rng(0))
    assert p["epaississement_2"].sum() > p["epaississement_1"].sum() > verite.sum()
    assert p["amincissement"].sum() < verite.sum()
    assert p["faux_positifs_15"].sum() > p["faux_positifs_5"].sum() > verite.sum()
    for nom in ("coupures_20", "coupures_60", "petits_supprimes_1", "amincissement"):
        assert np.all(p[nom] <= verite), nom  # ne fait que retirer des pixels vaisseau
    assert p["coupures_60"].sum() < p["coupures_20"].sum()


def test_metriques_sur_l_annotation_elle_meme(retine):
    _, masque, verite = retine
    m = metriques(verite, verite, masque)
    assert m["dice"] == m["cl_dice"] == m["mcc"] == 1.0
    assert m["erreur_b0"] == m["erreur_b1"] == m["hd95"] == 0.0


def test_analyse_retrouve_une_metrique_parfaite():
    rng = np.random.default_rng(0)
    lignes = []
    for image in ("01", "02", "03", "04", "05"):
        for k in range(6):
            erreur = rng.random()
            lignes.append(
                {
                    "image": image,
                    "methode": f"m{k}",
                    "famille": "Frangi",
                    # toutes les métriques suivent exactement l'erreur de calibre
                    "metriques": {m: (1 - erreur) if m in PLUS_EST_MIEUX else erreur for m in METRIQUES},
                    "erreurs": {c: erreur if c == "calibre_moyen" else rng.random() for c in CIBLES},
                }
            )
    resultat = analyser(lignes)
    for m in METRIQUES:
        assert resultat["pouvoir_predictif"]["toutes"][m]["calibre_moyen"]["moyenne"] == pytest.approx(1.0)
    assert abs(resultat["pouvoir_predictif"]["toutes"]["dice"]["tortuosite"]["moyenne"]) < 0.6
