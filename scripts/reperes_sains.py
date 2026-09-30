"""Repères « yeux sains » des mesures en zones, avec la même chaîne que la démo.

Yeux sains : RFMiD (Disease_Risk = 0, trois appareils, Inde) et JSIEC (catégorie « Normal »). Chaque
photo suit exactement le chemin de la démo : isolement de l'œil, réduction à 800 pixels, U-Net
(seuil 0,5), papille, mesures en zones. Aucun modèle n'est entraîné. Écrit
resultats/reperes_zones.json : percentiles 5, 25, 50, 75 et 95 de chaque mesure, lus par la démo.

    python scripts/reperes_sains.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from vaisseaux.donnees import isoler_fond_oeil  # noqa: E402
from vaisseaux.zones import MESURES_ZONES, mesurer_zones, papille  # noqa: E402

KAGGLE = Path.home() / ".cache" / "kagglehub" / "datasets"
RFMID = KAGGLE / "andrewmvd/retinal-disease-classification/versions/1"
JSIEC = KAGGLE / "linchundan/fundusimage1000/versions/4/1000images/0.0.Normal"
COTE_MAX, SEUIL_UNET = 800, 0.5
_SESSION = None


def yeux_sains() -> list[str]:
    chemins = []
    for partie, csv, dossier in (
        ("Training_Set", "RFMiD_Training_Labels.csv", "Training"),
        ("Evaluation_Set", "RFMiD_Validation_Labels.csv", "Validation"),
        ("Test_Set", "RFMiD_Testing_Labels.csv", "Test"),
    ):
        base = RFMID / partie / partie
        table = pd.read_csv(base / csv)
        chemins += [str(base / dossier / f"{i}.png") for i in table.loc[table["Disease_Risk"] == 0, "ID"]]
    chemins += [str(p) for p in sorted(JSIEC.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return chemins


def mesurer(chemin: str) -> dict | None:
    import onnxruntime as ort

    from vaisseaux.pretraitement import normaliser_pour_reseau

    global _SESSION
    if _SESSION is None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        _SESSION = ort.InferenceSession(
            str(RACINE / "modeles" / "unet_drive.onnx"), options, providers=["CPUExecutionProvider"]
        )
    try:
        image = np.asarray(Image.open(chemin).convert("RGB"))
        image, oeil, cadre = isoler_fond_oeil(image)
        if cadre is None:
            return None
        echelle = max(image.shape[:2]) / COTE_MAX
        if echelle > 1:
            taille = (round(image.shape[1] / echelle), round(image.shape[0] / echelle))
            image = np.asarray(Image.fromarray(image).resize(taille, Image.Resampling.LANCZOS))
            oeil = np.asarray(Image.fromarray(oeil).resize(taille, Image.Resampling.NEAREST))
        x = normaliser_pour_reseau(image, oeil)[None]
        logits = _SESSION.run(None, {"image": x})[0][0][: oeil.shape[0], : oeil.shape[1]]
        vaisseaux = (1 / (1 + np.exp(-logits)) >= SEUIL_UNET) & oeil
        disque = papille(image, oeil, vaisseaux)
        return {"chemin": chemin, "diametre_estime": disque["diametre_estime"], **mesurer_zones(vaisseaux, disque)}
    except Exception as erreur:  # image illisible : ignorée, mais signalée
        print(f"ignorée : {chemin} ({erreur})", flush=True)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--processus", type=int, default=3)
    parser.add_argument("--sortie", type=Path, default=RACINE / "resultats" / "reperes_zones.json")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    chemins = yeux_sains()
    print(f"{len(chemins)} yeux sains", flush=True)
    with ProcessPoolExecutor(args.processus) as ex:
        lignes = [r for r in ex.map(mesurer, chemins, chunksize=4) if r is not None]
    table = pd.DataFrame(lignes)
    table.to_csv(args.sortie.with_suffix(".csv"), index=False)
    fiables = table[~table["diametre_estime"]]  # papille mesurée : repères plus propres
    reperes = {
        k: {
            "percentiles": {str(q): float(np.nanpercentile(fiables[k], q)) for q in (5, 25, 50, 75, 95)},
            "yeux": int(fiables[k].notna().sum()),
        }
        for k in MESURES_ZONES
    }
    sortie = {
        "source": "yeux sains de RFMiD (3 appareils) et de JSIEC, même chaîne que la démo",
        "yeux": int(len(table)),
        "papille_mesuree": int(len(fiables)),
        "mesures": reperes,
    }
    args.sortie.write_text(json.dumps(sortie, indent=1, ensure_ascii=False))
    print(json.dumps(sortie, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
