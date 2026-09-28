"""Protocole d'évaluation sur DRIVE : quel prétraitement donne les meilleurs tracés ?

1. Sur les 20 images d'entraînement : choix, pour chaque configuration, du paramètre de
   coût alpha (meilleur F1 de tracé) et du seuil de binarisation (meilleur Dice).
2. Sur les 20 images de test, une seule fois : mesures de tracé et mesures pixel,
   moyennes avec intervalle de confiance bootstrap par image, et test de Wilcoxon apparié
   contre la configuration de référence (canal vert brut), corrigé par Holm.

    python -m vaisseaux.benchmark --racine data/DRIVE --sortie resultats
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .donnees import charger, lister
from .evaluation import (
    auc_roc,
    cl_dice,
    comparer_apparie,
    echantillonner_paires,
    evaluer_trace,
    intervalle_bootstrap,
    metriques_binaires,
)
from .graphe import carte_de_cout, plus_court_chemin
from .pretraitement import ParametresPretraitement, pretraiter

CONFIGURATIONS = {
    "vert_brut": ParametresPretraitement(clahe=False, debruitage="aucun"),
    "clahe": ParametresPretraitement(clahe=True, debruitage="aucun"),
    "clahe_gaussien": ParametresPretraitement(clahe=True, debruitage="gaussien"),
    "clahe_median": ParametresPretraitement(clahe=True, debruitage="median"),
    "clahe_bilateral": ParametresPretraitement(clahe=True, debruitage="bilateral"),
    "clahe_nl_means": ParametresPretraitement(clahe=True, debruitage="nl_means"),
}
REFERENCE = "vert_brut"
ALPHAS = (1.0, 2.0, 3.0)
SEUILS = np.round(np.arange(0.01, 0.61, 0.01), 2)
METRIQUES_TRACE = (
    "f1",
    "precision",
    "couverture",
    "distance_moyenne",
    "hausdorff95",
    "frechet",
    "ratio_longueur",
    "ecart_hors_vaisseau",
)


def _tracer(cartes, masque, paires, verite, alpha: float, tolerance: float) -> list[dict]:
    cout = carte_de_cout(cartes.vaisseaux, masque, alpha)
    resultats = []
    for paire in paires:
        debut = time.perf_counter()
        chemin = plus_court_chemin(cout, paire.depart, paire.arrivee, a_etoile=True)
        mesures = evaluer_trace(chemin.pixels, paire.reference, verite, tolerance)
        mesures["duree_ms"] = 1e3 * (time.perf_counter() - debut)
        resultats.append(mesures)
    return resultats


def _image_entrainement(tache: tuple) -> dict:
    racine, ident, n_paires, graine, tolerance, configs = tache
    im = charger(racine, "entrainement", ident)
    paires = echantillonner_paires(im.verite, im.masque, n_paires, np.random.default_rng([graine, int(ident)]))
    sortie = {}
    for nom in configs:
        cartes = pretraiter(im.rgb, im.masque, CONFIGURATIONS[nom])
        dice = [metriques_binaires(cartes.vaisseaux >= s, im.verite, im.masque)["dice"] for s in SEUILS]
        f1 = {
            a: float(np.mean([m["f1"] for m in _tracer(cartes, im.masque, paires, im.verite, a, tolerance)]))
            for a in ALPHAS
        }
        sortie[nom] = {"dice_par_seuil": dice, "f1_par_alpha": f1}
    return sortie


def _image_test(tache: tuple) -> dict:
    racine, ident, n_paires, graine, tolerance, choix = tache
    im = charger(racine, "test", ident)
    paires = echantillonner_paires(im.verite, im.masque, n_paires, np.random.default_rng([graine, int(ident)]))
    sortie: dict = {"identifiant": ident, "configs": {}}
    if im.verite2 is not None:
        sortie["second_observateur"] = {
            **metriques_binaires(im.verite2, im.verite, im.masque),
            "cl_dice": cl_dice(im.verite2 & im.masque, im.verite & im.masque),
        }
    for nom, (alpha, seuil) in choix.items():
        cartes = pretraiter(im.rgb, im.masque, CONFIGURATIONS[nom])
        binaire = (cartes.vaisseaux >= seuil) & im.masque
        sortie["configs"][nom] = {
            "pixels": {
                "auc": auc_roc(cartes.vaisseaux, im.verite, im.masque),
                **metriques_binaires(binaire, im.verite, im.masque),
                "cl_dice": cl_dice(binaire, im.verite & im.masque),
            },
            "traces": _tracer(cartes, im.masque, paires, im.verite, alpha, tolerance),
        }
    return sortie


def _holm(p_valeurs: dict[str, float]) -> dict[str, float]:
    ordre = sorted(p_valeurs, key=p_valeurs.get)
    m, ajustees, courant = len(ordre), {}, 0.0
    for rang, cle in enumerate(ordre):
        courant = max(courant, min(1.0, (m - rang) * p_valeurs[cle]))
        ajustees[cle] = courant
    return ajustees


def lancer(racine: Path, sortie: Path, n_paires: int, graine: int, tolerance: float, travailleurs: int, rapide: bool):
    configs = list(CONFIGURATIONS) if not rapide else [REFERENCE, "clahe_nl_means"]
    ids_train, ids_test = lister(racine, "entrainement"), lister(racine, "test")
    if rapide:
        ids_train, ids_test = ids_train[:3], ids_test[:3]
    debut = time.perf_counter()

    with ProcessPoolExecutor(travailleurs) as pool:
        train = list(
            pool.map(_image_entrainement, [(racine, i, n_paires, graine, tolerance, configs) for i in ids_train])
        )
        choix = {}
        for nom in configs:
            dice = np.mean([t[nom]["dice_par_seuil"] for t in train], axis=0)
            f1 = {a: np.mean([t[nom]["f1_par_alpha"][a] for t in train]) for a in ALPHAS}
            choix[nom] = (max(f1, key=f1.get), float(SEUILS[int(np.argmax(dice))]))
        test = list(pool.map(_image_test, [(racine, i, n_paires, graine, tolerance, choix) for i in ids_test]))

    resume: dict = {
        "protocole": {
            "images_entrainement": len(ids_train),
            "images_test": len(ids_test),
            "paires_par_image": n_paires,
            "tolerance_px": tolerance,
            "graine": graine,
            "parametres": {nom: asdict(CONFIGURATIONS[nom]) for nom in configs},
            "choix_sur_entrainement": {nom: {"alpha": a, "seuil": s} for nom, (a, s) in choix.items()},
        },
        "configs": {},
    }
    groupes_paires = np.concatenate([[r["identifiant"]] * len(r["configs"][configs[0]]["traces"]) for r in test])
    par_image: dict[str, dict[str, np.ndarray]] = {}
    for nom in configs:
        traces = [m for r in test for m in r["configs"][nom]["traces"]]
        bloc = {"traces": {}, "pixels": {}}
        for cle in (*METRIQUES_TRACE, "duree_ms"):
            valeurs = np.array([m[cle] for m in traces])
            bloc["traces"][cle] = dict(
                zip(("moyenne", "ic_bas", "ic_haut"), intervalle_bootstrap(valeurs, groupes_paires), strict=True)
            )
        ids = np.array([r["identifiant"] for r in test])
        for cle in test[0]["configs"][nom]["pixels"]:
            valeurs = np.array([r["configs"][nom]["pixels"][cle] for r in test])
            bloc["pixels"][cle] = dict(
                zip(("moyenne", "ic_bas", "ic_haut"), intervalle_bootstrap(valeurs, ids), strict=True)
            )
        par_image[nom] = {
            "f1": np.array([np.mean([m["f1"] for m in r["configs"][nom]["traces"]]) for r in test]),
            "auc": np.array([r["configs"][nom]["pixels"]["auc"] for r in test]),
        }
        resume["configs"][nom] = bloc

    for cle in ("f1", "auc"):
        tests = {
            nom: comparer_apparie(par_image[REFERENCE][cle], par_image[nom][cle]) for nom in configs if nom != REFERENCE
        }
        holm = _holm({nom: t["p_valeur"] for nom, t in tests.items()})
        for nom, t in tests.items():
            resume["configs"][nom].setdefault("comparaison_reference", {})[cle] = {**t, "p_holm": holm[nom]}

    if "second_observateur" in test[0]:
        humain = {}
        for cle in test[0]["second_observateur"]:
            valeurs = np.array([r["second_observateur"][cle] for r in test])
            humain[cle] = dict(zip(("moyenne", "ic_bas", "ic_haut"), intervalle_bootstrap(valeurs, ids), strict=True))
        resume["second_observateur"] = humain
    resume["duree_s"] = round(time.perf_counter() - debut, 1)

    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "evaluation.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False))
    (sortie / "evaluation.md").write_text(rapport_markdown(resume))
    return resume


def _ic(m: dict, chiffres: int = 3, pourcent: bool = False) -> str:
    f = 100 if pourcent else 1
    return f"{m['moyenne'] * f:.{chiffres}f} [{m['ic_bas'] * f:.{chiffres}f} ; {m['ic_haut'] * f:.{chiffres}f}]"


def rapport_markdown(r: dict) -> str:
    p = r["protocole"]
    lignes = [
        "# Évaluation sur DRIVE",
        "",
        f"{p['images_test']} images de test, {p['paires_par_image']} paires de points par image, "
        f"tolérance {p['tolerance_px']} px, graine {p['graine']}. Paramètres choisis sur les "
        f"{p['images_entrainement']} images d'entraînement. Moyenne [IC 95 % bootstrap par image].",
        "",
        "## Tracés (Dijkstra / A*, face au chemin de référence sur le squelette annoté)",
        "",
        "| configuration | α | F1 tracé ↑ | précision ↑ | couverture ↑ | Fréchet (px) ↓ | HD95 (px) ↓ | plus long écart (px) ↓ | p Holm (F1) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for nom, c in r["configs"].items():
        t = c["traces"]
        comparaison = c.get("comparaison_reference", {}).get("f1")
        p_holm = f"{comparaison['p_holm']:.3g}" if comparaison else "référence"
        lignes.append(
            f"| {nom} | {p['choix_sur_entrainement'][nom]['alpha']:g} | {_ic(t['f1'])} | {_ic(t['precision'])} | "
            f"{_ic(t['couverture'])} | {_ic(t['frechet'], 1)} | {_ic(t['hausdorff95'], 1)} | "
            f"{_ic(t['ecart_hors_vaisseau'], 1)} | {p_holm} |"
        )
    lignes += [
        "",
        "## Carte de rehaussement (pixels du champ de vue)",
        "",
        "| configuration | seuil | AUC ↑ | sensibilité | spécificité | Dice ↑ | MCC ↑ | clDice ↑ | p Holm (AUC) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for nom, c in r["configs"].items():
        x = c["pixels"]
        comparaison = c.get("comparaison_reference", {}).get("auc")
        p_holm = f"{comparaison['p_holm']:.3g}" if comparaison else "référence"
        lignes.append(
            f"| {nom} | {p['choix_sur_entrainement'][nom]['seuil']:g} | {_ic(x['auc'])} | {x['sensibilite']['moyenne']:.3f} | "
            f"{x['specificite']['moyenne']:.3f} | {_ic(x['dice'])} | {_ic(x['mcc'])} | {_ic(x['cl_dice'])} | {p_holm} |"
        )
    if "second_observateur" in r:
        h = r["second_observateur"]
        lignes += [
            "",
            f"Second observateur face au premier : Dice {_ic(h['dice'])}, MCC {_ic(h['mcc'])}, "
            f"clDice {_ic(h['cl_dice'])} (borne de l'accord humain).",
        ]
    lignes += ["", f"Durée totale : {r['duree_s']} s."]
    return "\n".join(lignes) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Évaluation du tracé de vaisseaux sur DRIVE")
    parser.add_argument("--racine", type=Path, default=Path("data/DRIVE"))
    parser.add_argument("--sortie", type=Path, default=Path("resultats"))
    parser.add_argument("--paires", type=int, default=10, help="paires de points par image")
    parser.add_argument("--graine", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=2.0, help="tolérance en pixels")
    parser.add_argument("--travailleurs", type=int, default=8)
    parser.add_argument("--rapide", action="store_true", help="3 images, 2 configurations (tests)")
    args = parser.parse_args(argv)
    resume = lancer(args.racine, args.sortie, args.paires, args.graine, args.tolerance, args.travailleurs, args.rapide)
    print((args.sortie / "evaluation.md").read_text())
    print(f"terminé en {resume['duree_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
