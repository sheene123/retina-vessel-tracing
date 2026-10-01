"""Repères « yeux sains » des mesures en zones, avec la même chaîne que la démo.

Yeux sains : RFMiD (Disease_Risk = 0, trois appareils, Inde) et JSIEC (catégorie « Normal »). Chaque
photo suit exactement le chemin de la démo : isolement de l'œil, réduction à 800 pixels, U-Net
(seuil 0,5), papille, mesures en zones. Aucun modèle n'est entraîné. Écrit
resultats/reperes_zones.json : percentiles 5, 25, 50, 75 et 95 de chaque mesure.

Contrôle du biais d'appareil (`--comparer`) : les parties « test » de DRIVE_AV, HRF-AV et LES-AV,
d'autres appareils et elles aussi surtout saines, passent dans la même chaîne. Si les repères
mesuraient la biologie, ces yeux tomberaient autour de la médiane ; s'ils tombent loin au-dessus
ou au-dessous, les repères mesurent surtout l'appareil et la démo ne doit pas les afficher.

    python scripts/reperes_sains.py [--modele modeles/unet_av.onnx] [--comparer <data de retina-av-dataset>]
"""

from __future__ import annotations

import argparse
import glob
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
from vaisseaux.zones import MESURES_AV, MESURES_ZONES, mesurer_zones, papille  # noqa: E402

KAGGLE = Path.home() / ".cache" / "kagglehub" / "datasets"
RFMID = KAGGLE / "andrewmvd/retinal-disease-classification/versions/1"
JSIEC = KAGGLE / "linchundan/fundusimage1000/versions/4/1000images/0.0.Normal"
COTE_MAX, SEUIL_UNET = 800, 0.5
_SESSION = None
_MODELE = RACINE / "modeles" / "unet_drive.onnx"


def _initialiser(modele: str) -> None:
    global _MODELE
    _MODELE = Path(modele)


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
        _SESSION = ort.InferenceSession(str(_MODELE), options, providers=["CPUExecutionProvider"])
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
        x = normaliser_pour_reseau(image, oeil, 32)[None]
        sorties = _SESSION.run(None, {"image": x})
        h, w = oeil.shape
        vaisseaux = (1 / (1 + np.exp(-sorties[0][0][:h, :w])) >= SEUIL_UNET) & oeil
        arteres = 1 / (1 + np.exp(-sorties[1][0][:h, :w])) if len(sorties) > 1 else None
        disque = papille(image, oeil, vaisseaux)
        mesures = mesurer_zones(vaisseaux, disque, arteres)
        return {"chemin": chemin, "diametre_estime": disque["diametre_estime"], **mesures}
    except Exception as erreur:  # image illisible : ignorée, mais signalée
        print(f"ignorée : {chemin} ({erreur})", flush=True)
        return None


def comparer(table: pd.DataFrame, autres: pd.DataFrame, mesures: list[str]) -> dict:
    """Pour chaque base d'un autre appareil : rang centile médian parmi les yeux sains (50 si les
    repères mesurent la biologie) et part des yeux sous le 5e ou au-dessus du 95e percentile
    (10 % attendus en tout)."""
    sortie = {}
    for base, groupe in autres.groupby("base"):
        sortie[base] = {"yeux": int(len(groupe))}
        for k in mesures:
            ref = table[k].dropna().to_numpy()
            v = groupe[k].dropna().to_numpy()
            if len(v) == 0:
                continue
            rangs = np.array([100 * (ref < x).mean() for x in v])
            sortie[base][k] = {
                "rang_centile_median": float(np.median(rangs)),
                "sous_p5": float((rangs < 5).mean()),
                "au_dessus_p95": float((rangs > 95).mean()),
            }
    return sortie


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--processus", type=int, default=3)
    parser.add_argument("--modele", type=Path, default=_MODELE)
    parser.add_argument("--comparer", type=Path, help="data de retina-av-dataset : contrôle du biais d'appareil")
    parser.add_argument("--sortie", type=Path, default=RACINE / "resultats" / "reperes_zones.json")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    chemins = yeux_sains()
    autres = []
    if args.comparer:
        for base in ("DRIVE_AV", "HRF-AV", "LES-AV"):
            autres += [(base, c) for c in sorted(glob.glob(str(args.comparer / base / "test" / "images" / "*")))]
    print(f"{len(chemins)} yeux sains, {len(autres)} yeux d'autres appareils ; modèle {args.modele.name}", flush=True)
    with ProcessPoolExecutor(args.processus, initializer=_initialiser, initargs=(str(args.modele),)) as ex:
        resultats = list(ex.map(mesurer, chemins + [c for _, c in autres], chunksize=4))
    lignes = [r for r in resultats[: len(chemins)] if r is not None]
    table = pd.DataFrame(lignes)
    table.to_csv(args.sortie.with_suffix(".csv"), index=False)
    fiables = table[~table["diametre_estime"]]  # papille mesurée : repères plus propres
    mesures = [k for k in (*MESURES_ZONES, *MESURES_AV) if k in table]
    reperes = {
        k: {
            "percentiles": {str(q): float(np.nanpercentile(fiables[k], q)) for q in (5, 25, 50, 75, 95)},
            "yeux": int(fiables[k].notna().sum()),
        }
        for k in mesures
    }
    sortie = {
        "source": "yeux sains de RFMiD (3 appareils) et de JSIEC, même chaîne que la démo",
        "modele": args.modele.name,
        "yeux": int(len(table)),
        "papille_mesuree": int(len(fiables)),
        "mesures": reperes,
    }
    if autres:
        autres_table = pd.DataFrame(
            [{"base": b} | r for (b, _), r in zip(autres, resultats[len(chemins) :], strict=True) if r is not None]
        )
        sortie["autres_appareils"] = comparer(fiables, autres_table[~autres_table["diametre_estime"]], mesures)
    args.sortie.write_text(json.dumps(sortie, indent=1, ensure_ascii=False))
    print(json.dumps(sortie, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
