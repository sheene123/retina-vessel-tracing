"""Fiabilité des mesures en zones : valeur du U-Net contre valeur de l'expert, image par image.

Sur les 106 images annotées de DRIVE_AV, HRF-AV et LES-AV (vaisseaux tracés par un expert), pour
chaque mesure : corrélation de rang entre U-Net et expert (le U-Net classe-t-il les yeux dans le
même ordre ?) et écart relatif médian. Écrit resultats/fiabilite_zones.json, lu par la démo.

    python scripts/fiabilite_zones.py --racine <dossier data de muflihsan/retina-av-dataset>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
sys.path.insert(0, str(RACINE / "scripts"))

from evaluer_arteres_veines import champ, charger, unet  # noqa: E402

from vaisseaux.zones import MESURES_ZONES, mesurer_zones, papille  # noqa: E402


def main() -> int:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--racine", type=Path, required=True)
    parser.add_argument("--sortie", type=Path, default=RACINE / "resultats" / "fiabilite_zones.json")
    args = parser.parse_args()
    session = ort.InferenceSession(str(RACINE / "modeles" / "unet_drive.onnx"), providers=["CPUExecutionProvider"])
    valeurs: dict[str, list] = {k: [] for k in MESURES_ZONES}
    estimes = 0
    images = []
    for base in ("DRIVE_AV", "HRF-AV", "LES-AV"):
        for partie in ("training", "test"):
            for image, etiquette in zip(
                sorted(glob.glob(str(args.racine / base / partie / "images" / "*"))),
                sorted(glob.glob(str(args.racine / base / partie / "1st_manual" / "*"))),
                strict=True,
            ):
                rgb, _, _, expert = charger(image, etiquette)
                masque = champ(rgb)
                auto = unet(session, rgb, masque)
                disque = papille(rgb, masque, auto)
                estimes += disque["diametre_estime"]
                m_auto, m_expert = mesurer_zones(auto, disque), mesurer_zones(expert & masque, disque)
                for k in MESURES_ZONES:
                    valeurs[k].append((m_auto[k], m_expert[k]))
                images.append(f"{base}/{partie}/{Path(image).name}")
    fiabilite = {}
    for k, paires in valeurs.items():
        a = np.array(paires, dtype=float)
        ok = np.isfinite(a).all(axis=1)
        fiabilite[k] = {
            "correlation_patients": float(spearmanr(a[ok, 0], a[ok, 1]).statistic),
            "ecart_relatif_median": float(np.median(np.abs(a[ok, 0] - a[ok, 1]) / np.abs(a[ok, 1]))),
            "images": int(ok.sum()),
        }
    sortie = {"mesures": fiabilite, "diametre_papille_par_defaut": estimes, "images": len(images)}
    args.sortie.write_text(json.dumps(sortie, indent=1, ensure_ascii=False))
    print(json.dumps(sortie, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
