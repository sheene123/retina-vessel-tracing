"""Évalue un modèle ONNX des troubles sur le test externe JSIEC-1000 (jamais vu à l'entraînement).

Sert à comparer des versions sur exactement les mêmes images et le même protocole :

    python scripts/evaluer_externe.py --modele modeles/troubles.onnx --calibrage resultats/troubles/troubles_demo.json
    python scripts/evaluer_externe.py --version v0.1.1     # une version du registre
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from vaisseaux.troubles import CLES, evaluer_externe, lire_index, transformations  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modele", type=Path)
    parser.add_argument("--calibrage", type=Path)
    parser.add_argument("--version", help="version du registre (télécharge troubles.onnx et troubles.json)")
    parser.add_argument("--repo", default="sheenee261/retina-vessel-tracing")
    parser.add_argument("--index", type=Path, default=RACINE / "data" / "troubles" / "index.csv")
    parser.add_argument("--sortie", type=Path, help="écrit le résultat en JSON")
    args = parser.parse_args()
    if args.version:
        from huggingface_hub import hf_hub_download

        args.modele = Path(hf_hub_download(args.repo, "troubles.onnx", revision=args.version))
        args.calibrage = Path(hf_hub_download(args.repo, "troubles.json", revision=args.version))

    import onnxruntime as ort

    configuration = json.loads(args.calibrage.read_text())
    calibrage = {c: (configuration["troubles"][c]["a"], configuration["troubles"][c]["b"]) for c in CLES}
    index = lire_index(args.index)
    externe = index[index["source"] == "jsiec"].reset_index(drop=True)
    session = ort.InferenceSession(str(args.modele), providers=["CPUExecutionProvider"])
    preparer = transformations(entrainement=False)
    z = np.stack(
        [
            session.run(None, {"image": preparer(Image.open(c).convert("RGB")).numpy()[None]})[0][0]
            for c in externe["chemin"]
        ]
    )
    resultat = evaluer_externe(externe, z, calibrage)
    print(f"test externe JSIEC : {len(externe)} images — {args.modele}")
    print(
        f"{'trouble':13s} {'AUROC [IC 95 %]':>22s} {'atteints':>9s} {'signalés ≥ possible':>20s} {'sains peu probable':>19s}"
    )
    for c in CLES:
        r = resultat[c]
        if "auroc" not in r:
            print(f"{c:13s} {'— (aucun cas)':>22s}")
            continue
        print(
            f"{c:13s} {r['auroc']:.3f} [{r['ic_bas']:.2f} ; {r['ic_haut']:.2f}] {r['atteints']:9d}"
            f" {100 * r['sensibilite_possible']:19.0f}% {100 * r['specificite_peu_probable']:18.0f}%"
        )
    if args.sortie:
        args.sortie.write_text(json.dumps(resultat, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
