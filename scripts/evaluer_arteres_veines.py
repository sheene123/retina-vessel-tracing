"""Évalue le classement artères / veines sans apprentissage et l'AVR face aux annotations d'experts.

Bases annotées (artères en rouge, veines en bleu) : DRIVE_AV (RITE), HRF-AV et LES-AV. Aucun modèle
n'est entraîné ; les réglages (par quadrant ou non) sont choisis sur les parties « training » et le
résultat est rapporté sur les parties « test ». Les images sont réduites à 800 pixels au plus,
comme dans la démo.

Deux conditions : vaisseaux de l'expert (mesure la qualité du seul classement) et vaisseaux du U-Net
(chaîne complète de la démo). La référence est calculée avec les vaisseaux et les étiquettes de
l'expert.

    python scripts/evaluer_arteres_veines.py --racine <dossier data de muflihsan/retina-av-dataset>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from vaisseaux.pretraitement import normaliser_pour_reseau  # noqa: E402
from vaisseaux.zones import (  # noqa: E402
    KNUDTSON_ARTERES,
    KNUDTSON_VEINES,
    caracteristiques,
    classer_arteres,
    knudtson,
    papille,
    segments_zone,
)

COTE_MAX, SEUIL_UNET = 800, 0.5


def charger(image: str, etiquette: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    im = Image.open(image).convert("RGB")
    im.thumbnail((COTE_MAX, COTE_MAX), Image.Resampling.LANCZOS)
    lab = np.asarray(Image.open(etiquette).convert("RGB").resize(im.size, Image.Resampling.NEAREST)).astype(int)
    rouge = (lab[..., 0] > 127) & (lab[..., 2] < 128) & (lab[..., 1] < 128)
    bleu = (lab[..., 2] > 127) & (lab[..., 0] < 128) & (lab[..., 1] < 128)
    vaisseaux = lab.max(axis=-1) > 0
    return np.asarray(im), rouge, bleu, vaisseaux


def champ(rgb: np.ndarray) -> np.ndarray:
    m = ndi.binary_fill_holes(ndi.binary_opening(rgb[..., 0] > 20, iterations=2))
    return ndi.binary_erosion(m, iterations=3)


def unet(session, rgb: np.ndarray, masque: np.ndarray) -> np.ndarray:
    x = normaliser_pour_reseau(rgb, masque)[None]
    logits = session.run(None, {"image": x})[0][0][: masque.shape[0], : masque.shape[1]]
    return (1 / (1 + np.exp(-logits)) >= SEUIL_UNET) & masque


def verite_segments(segs: list[dict], rouge: np.ndarray, bleu: np.ndarray) -> np.ndarray:
    """1 artère, 0 veine, -1 indéterminé : majorité des étiquettes de l'expert à 2 pixels près."""
    r, b = ndi.binary_dilation(rouge, iterations=2), ndi.binary_dilation(bleu, iterations=2)
    sortie = np.full(len(segs), -1)
    for i, s in enumerate(segs):
        nr, nb = r[s["lignes"], s["colonnes"]].sum(), b[s["lignes"], s["colonnes"]].sum()
        if nr + nb >= 0.5 * s["longueur"] and nr != nb:
            sortie[i] = int(nr > nb)
    return sortie


def avr(segs: list[dict], arteres: np.ndarray) -> float:
    largeurs = np.array([s["largeur"] for s in segs])
    crae, crve = knudtson(largeurs[arteres], KNUDTSON_ARTERES), knudtson(largeurs[~arteres], KNUDTSON_VEINES)
    return crae / crve if np.isfinite(crae) and np.isfinite(crve) and crve > 0 else float("nan")


def main() -> int:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--racine", type=Path, required=True)
    parser.add_argument("--sortie", type=Path, default=RACINE / "resultats" / "arteres_veines.json")
    args = parser.parse_args()
    session = ort.InferenceSession(str(RACINE / "modeles" / "unet_drive.onnx"), providers=["CPUExecutionProvider"])
    resultats: dict = {}
    for base in ("DRIVE_AV", "HRF-AV", "LES-AV"):
        for partie in ("training", "test"):
            images = sorted(glob.glob(str(args.racine / base / partie / "images" / "*")))
            etiquettes = sorted(glob.glob(str(args.racine / base / partie / "1st_manual" / "*")))
            lignes = []
            for image, etiquette in zip(images, etiquettes, strict=True):
                rgb, rouge, bleu, vaisseaux_expert = charger(image, etiquette)
                masque = champ(rgb)
                vaisseaux_unet = unet(session, rgb, masque)
                disque = papille(rgb, masque, vaisseaux_unet)
                # référence : vaisseaux et étiquettes de l'expert
                segs_ref = segments_zone(vaisseaux_expert & masque, disque)
                v_ref = verite_segments(segs_ref, rouge, bleu)
                connus = v_ref >= 0
                avr_ref = avr([s for s, c in zip(segs_ref, connus, strict=True) if c], v_ref[connus] == 1)
                ligne = {"image": Path(image).name, "avr_reference": avr_ref}
                for nom, vaisseaux in (("expert", vaisseaux_expert & masque), ("unet", vaisseaux_unet)):
                    segs = segments_zone(vaisseaux, disque)
                    if len(segs) < 4:
                        continue
                    carac = caracteristiques(rgb, vaisseaux, segs)
                    angles = np.array([s["angle"] for s in segs])
                    verite = verite_segments(segs, rouge, bleu)
                    poids = np.array([s["longueur"] for s in segs])
                    for variante, par_quadrant in (("quadrants", True), ("global", False)):
                        arteres = classer_arteres(carac, angles, par_quadrant)
                        ok = verite >= 0
                        ligne[f"{nom}_{variante}_exactitude"] = (
                            float(np.average((arteres[ok] == (verite[ok] == 1)), weights=poids[ok]))
                            if ok.any()
                            else float("nan")
                        )
                        ligne[f"{nom}_{variante}_avr"] = avr(segs, arteres)
                lignes.append(ligne)
            resultats[f"{base}/{partie}"] = lignes
            print(f"{base:9s} {partie:8s} {len(lignes)} images", flush=True)

    synthese = {}
    for cle, lignes in resultats.items():
        s = {}
        for nom in ("expert", "unet"):
            for variante in ("quadrants", "global"):
                ex = np.array([ligne.get(f"{nom}_{variante}_exactitude", np.nan) for ligne in lignes])
                a = np.array([ligne.get(f"{nom}_{variante}_avr", np.nan) for ligne in lignes])
                ref = np.array([ligne["avr_reference"] for ligne in lignes])
                ok = np.isfinite(a) & np.isfinite(ref)
                s[f"{nom}_{variante}"] = {
                    "exactitude": float(np.nanmean(ex)),
                    "avr_spearman": float(spearmanr(a[ok], ref[ok]).statistic) if ok.sum() > 3 else None,
                    "avr_ecart_moyen": float(np.mean(np.abs(a[ok] - ref[ok]))) if ok.any() else None,
                    "avr_reference_moyen": float(np.mean(ref[ok])) if ok.any() else None,
                }
        synthese[cle] = s
    args.sortie.write_text(json.dumps({"synthese": synthese, "images": resultats}, indent=1, ensure_ascii=False))
    print(
        f"\n{'base/partie':18s} {'vaisseaux':9s} {'variante':10s} {'exactitude':>10s} {'AVR ρ':>7s} {'écart AVR':>9s}"
    )
    for cle, s in synthese.items():
        for k, v in s.items():
            nom, variante = k.split("_")
            rho = "—" if v["avr_spearman"] is None else f"{v['avr_spearman']:.2f}"
            ecart = "—" if v["avr_ecart_moyen"] is None else f"{v['avr_ecart_moyen']:.3f}"
            print(f"{cle:18s} {nom:9s} {variante:10s} {v['exactitude']:10.1%} {rho:>7s} {ecart:>9s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
