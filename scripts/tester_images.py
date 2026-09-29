"""Passe ses propres images dans un ou plusieurs modèles des troubles et compare les pourcentages.

Chaque image suit la même chaîne que la démo (isolement de l'œil, carré 384 × 384). Produit un
tableau dans le terminal et une planche PNG des images telles que le réseau les voit.

    python scripts/tester_images.py --images ~/mes_images \\
        --modele actuel=v0.1.1 --modele nouveau=modeles/troubles.onnx:resultats/troubles/troubles_demo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from vaisseaux.donnees import entree_imagenet, preparer_fond_oeil  # noqa: E402
from vaisseaux.troubles import CLES  # noqa: E402

EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")


def charger_modele(description: str, repo: str):
    """« nom=v0.1.1 » (registre) ou « nom=modele.onnx:calibrage.json »."""
    import onnxruntime as ort

    nom, source = description.split("=", 1)
    if source.startswith("v") and ":" not in source:
        from huggingface_hub import hf_hub_download

        onnx = hf_hub_download(repo, "troubles.onnx", revision=source)
        calibrage = hf_hub_download(repo, "troubles.json", revision=source)
    else:
        onnx, calibrage = source.split(":", 1)
    configuration = json.loads(Path(calibrage).read_text())
    return nom, ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"]), configuration


def probabilites(session, configuration: dict, carre: np.ndarray) -> dict[str, float]:
    z = session.run(None, {"image": entree_imagenet(carre)[None]})[0][0]
    t = configuration["troubles"]
    return {c: float(1 / (1 + np.exp(-(t[c]["a"] * z[j] + t[c]["b"])))) for j, c in enumerate(CLES)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--images", type=Path, required=True, help="dossier ou fichier image")
    parser.add_argument("--modele", action="append", required=True, help="nom=v0.1.1 ou nom=x.onnx:x.json")
    parser.add_argument("--repo", default="sheenee261/retina-vessel-tracing")
    parser.add_argument("--planche", type=Path, default=Path("planche_troubles.png"))
    args = parser.parse_args()

    fichiers = (
        sorted(p for p in args.images.iterdir() if p.suffix.lower() in EXTENSIONS)
        if args.images.is_dir()
        else [args.images]
    )
    modeles = [charger_modele(m, args.repo) for m in args.modele]
    vignettes = []
    print("pourcentages : " + " | ".join(nom for nom, _, _ in modeles))
    for i, chemin in enumerate(fichiers):
        rgb = np.asarray(Image.open(chemin).convert("RGB"))
        couleur = np.abs(rgb.astype(np.int16) - rgb.mean(axis=-1, keepdims=True)).mean()
        if couleur < 6:
            print(f"[{i}] {chemin.name[:40]} : noir et blanc (angiographie ?), ignorée")
            continue
        carre = preparer_fond_oeil(rgb)
        print(f"[{i}] {chemin.name[:40]}")
        for nom, session, configuration in modeles:
            p = probabilites(session, configuration, carre)
            print(f"    {nom:10s} " + "  ".join(f"{c[:6]} {100 * v:3.0f} %" for c, v in p.items()))
        vignette = Image.fromarray(carre).resize((256, 256))
        ImageDraw.Draw(vignette).text((6, 6), str(i), fill=(255, 255, 0))
        vignettes.append(vignette)
    if vignettes:
        colonnes = min(5, len(vignettes))
        lignes = (len(vignettes) + colonnes - 1) // colonnes
        planche = Image.new("RGB", (256 * colonnes, 256 * lignes))
        for k, v in enumerate(vignettes):
            planche.paste(v, ((k % colonnes) * 256, (k // colonnes) * 256))
        planche.save(args.planche)
        print(f"planche : {args.planche}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
