"""Figure du README : carte de rehaussement et tracés comparés au chemin de référence.

python scripts/figure_exemple.py --identifiant 02 --sortie resultats/exemple_trace.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vaisseaux.api import ALPHA_DEFAUT
from vaisseaux.donnees import charger
from vaisseaux.evaluation import echantillonner_paires, evaluer_trace
from vaisseaux.graphe import carte_de_cout, plus_court_chemin
from vaisseaux.pretraitement import pretraiter

REFERENCE = "#56B4E9"  # bleu ciel (palette Okabe-Ito, lisible par les daltoniens)
TRACE = "#E69F00"  # orange


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--racine", type=Path, default=Path("data/DRIVE"))
    parser.add_argument("--identifiant", default="02")
    parser.add_argument("--paires", type=int, default=5)
    parser.add_argument("--sortie", type=Path, default=Path("resultats/exemple_trace.png"))
    args = parser.parse_args()

    im = charger(args.racine, "test", args.identifiant)
    cartes = pretraiter(im.rgb, im.masque)
    cout = carte_de_cout(cartes.vaisseaux, im.masque, ALPHA_DEFAUT)
    paires = echantillonner_paires(im.verite, im.masque, args.paires, np.random.default_rng(3), 80, 200)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), constrained_layout=True)
    axes[0].imshow(np.where(im.masque, cartes.vaisseaux, np.nan), cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("Carte de rehaussement (Frangi après CLAHE + NL-means)", fontsize=10, color="#333333")
    axes[1].imshow(cartes.vert, cmap="gray")
    axes[1].set_title("Tracés entre deux points (A*) et référence experte", fontsize=10, color="#333333")
    for k, paire in enumerate(paires):
        chemin = plus_court_chemin(cout, paire.depart, paire.arrivee, a_etoile=True)
        f1 = evaluer_trace(chemin.pixels, paire.reference, im.verite)["f1"]
        axes[1].plot(
            *paire.reference[:, ::-1].T,
            color=REFERENCE,
            lw=4,
            alpha=0.8,
            label="référence (squelette annoté)" if k == 0 else None,
        )
        axes[1].plot(
            *chemin.pixels[:, ::-1].T, color=TRACE, lw=1.4, label="tracé (plus court chemin)" if k == 0 else None
        )
        for point in (paire.depart, paire.arrivee):
            axes[1].plot(point[1], point[0], "o", ms=5, mfc="white", mec="#333333", mew=1)
        milieu = chemin.pixels[len(chemin.pixels) // 2]
        axes[1].annotate(
            f"F1 {f1:.2f}",
            (milieu[1], milieu[0]),
            xytext=(6, -6),
            textcoords="offset points",
            fontsize=8,
            color="white",
        )
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.9)
    for ax in axes:
        ax.set_axis_off()
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.sortie, dpi=120)
    print(f"figure écrite dans {args.sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
