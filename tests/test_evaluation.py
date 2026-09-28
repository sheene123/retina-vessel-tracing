import numpy as np
import pytest

from vaisseaux.evaluation import (
    auc_roc,
    cl_dice,
    comparer_apparie,
    echantillonner_paires,
    evaluer_trace,
    frechet_discret,
    intervalle_bootstrap,
    metriques_binaires,
    plus_long_ecart,
    reference_entre,
)


def _ligne(n: int, ligne: int = 10) -> np.ndarray:
    return np.stack([np.full(n, ligne), np.arange(n)], axis=1)


def test_trace_parfait():
    verite = np.zeros((20, 40), bool)
    verite[9:12] = True
    m = evaluer_trace(_ligne(40), _ligne(40), verite)
    assert m["precision"] == m["couverture"] == m["f1"] == 1.0
    assert m["hausdorff"] == m["frechet"] == m["ecart_hors_vaisseau"] == 0.0
    assert m["ratio_longueur"] == pytest.approx(1.0)


def test_trace_decale():
    verite = np.zeros((20, 40), bool)
    verite[10] = True
    m = evaluer_trace(_ligne(40, 13), _ligne(40, 10), verite, tolerance=2)
    assert m["hausdorff"] == pytest.approx(3.0)
    assert m["precision"] == 0.0 and m["couverture"] == 0.0
    assert m["ecart_hors_vaisseau"] == pytest.approx(39.0)


def test_frechet_voit_l_ordre():
    a = _ligne(20)
    aller_retour = np.concatenate([a[:15], a[14::-1], a])
    assert evaluer_trace(aller_retour, a, np.ones((20, 20), bool))["hausdorff"] == 0.0
    assert frechet_discret(aller_retour, a) > 5
    assert frechet_discret(a, a[::-1]) == pytest.approx(19.0)


def test_plus_long_ecart():
    verite = np.zeros((20, 40), bool)
    verite[10, :10] = verite[10, 30:] = True
    assert plus_long_ecart(_ligne(40), verite, tolerance=0) == pytest.approx(21.0)


def test_metriques_pixels():
    rng = np.random.default_rng(0)
    verite = rng.random((50, 50)) < 0.2
    masque = np.ones_like(verite)
    assert auc_roc(verite.astype(float), verite, masque) == 1.0
    assert auc_roc(-verite.astype(float), verite, masque) == 0.0
    m = metriques_binaires(verite, verite, masque)
    assert m["dice"] == m["mcc"] == m["sensibilite"] == m["specificite"] == 1.0
    assert metriques_binaires(~verite, verite, masque)["mcc"] == pytest.approx(-1.0)


def test_cl_dice_penalise_les_coupures():
    verite = np.zeros((30, 60), bool)
    verite[14:17, 2:58] = True
    coupee = verite.copy()
    coupee[:, 20:40] = False
    assert cl_dice(verite, verite) == pytest.approx(1.0)
    assert cl_dice(coupee, verite) < 0.85


def test_echantillonnage_des_paires():
    verite = np.zeros((80, 120), bool)
    verite[40, 5:115] = True
    verite[10:40, 60] = True
    paires = echantillonner_paires(verite, np.ones_like(verite), 5, np.random.default_rng(0), 30, 80, marge=2)
    assert len(paires) == 5
    for p in paires:
        assert 30 <= p.longueur <= 80
        assert np.all(verite[tuple(p.reference.T)])
        assert np.all(np.abs(np.diff(p.reference, axis=0)).max(axis=1) == 1)


def test_statistiques():
    valeurs = np.array([0.8, 0.9, 0.85, 0.7, 0.95, 0.75])
    groupes = np.array([1, 1, 2, 2, 3, 3])
    moyenne, bas, haut = intervalle_bootstrap(valeurs, groupes)
    assert bas <= moyenne <= haut
    a = np.linspace(0.8, 0.9, 12)
    assert comparer_apparie(a, a + 0.05)["p_valeur"] < 0.01
    assert comparer_apparie(a, a)["p_valeur"] == 1.0


def test_reference_entre_deux_points_cliques():
    verite = np.zeros((60, 80), bool)
    verite[30, 5:75] = True
    verite[5:30, 40] = True
    reference = reference_entre(verite, (32, 10), (8, 41))
    assert reference is not None
    assert tuple(reference[0]) == (30, 10) and tuple(reference[-1]) == (8, 40)
    assert np.all(verite[tuple(reference.T)])
    assert reference_entre(verite, (50, 10), (8, 41)) is None  # départ trop loin d'un vaisseau
    verite[30, 38:43] = False  # coupure : plus de chemin
    assert reference_entre(verite, (30, 10), (30, 70)) is None
