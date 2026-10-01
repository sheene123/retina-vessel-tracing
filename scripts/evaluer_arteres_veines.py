"""Évalue le classement artères / veines et l'AVR face aux annotations d'experts.

Bases annotées (artères en rouge, veines en bleu) : DRIVE_AV (RITE), HRF-AV et LES-AV. Méthodes :

- sans apprentissage (« quadrants », « global ») : regroupement en deux classes selon la clarté ;
  réglages choisis sur les parties « training » ;
- « modele » : sortie « arteres » du U-Net multi-appareils (unet_av.onnx), si le modèle l'a. Ce
  modèle a appris sur les parties « training » : seules les parties « test » comptent pour lui.

Deux conditions de vaisseaux : ceux de l'expert (mesure la qualité du seul classement) et ceux du
U-Net (chaîne complète de la démo). La référence (CRAE, CRVE, AVR) est calculée avec les vaisseaux
et les étiquettes de l'expert. Les images sont réduites à 800 pixels au plus, comme dans la démo.

    python scripts/evaluer_arteres_veines.py --racine <dossier data de muflihsan/retina-av-dataset> \
        [--modele modeles/unet_av.onnx --parties test --sortie resultats/arteres_veines_unet_av.json]
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
    calibres_av,
    caracteristiques,
    classer_arteres,
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


def segmenter(session, rgb: np.ndarray, masque: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Probabilité « vaisseau » et, si le modèle a cette sortie, probabilité « artère »."""
    h, w = masque.shape
    sorties = session.run(None, {"image": normaliser_pour_reseau(rgb, masque, 32)[None]})
    proba = 1 / (1 + np.exp(-sorties[0][0][:h, :w]))
    arteres = 1 / (1 + np.exp(-sorties[1][0][:h, :w])) if len(sorties) > 1 else None
    return proba, arteres


def unet(session, rgb: np.ndarray, masque: np.ndarray) -> np.ndarray:
    return (segmenter(session, rgb, masque)[0] >= SEUIL_UNET) & masque


def carte_expert(rouge: np.ndarray, bleu: np.ndarray) -> np.ndarray:
    """1 artère, 0 veine, NaN ailleurs (à 2 pixels près des tracés de l'expert)."""
    r, b = ndi.binary_dilation(rouge, iterations=2), ndi.binary_dilation(bleu, iterations=2)
    return np.where(r & ~b, 1.0, np.where(b & ~r, 0.0, np.nan))


def verite_segments(segs: list[dict], rouge: np.ndarray, bleu: np.ndarray) -> np.ndarray:
    """1 artère, 0 veine, -1 indéterminé : majorité des étiquettes de l'expert à 2 pixels près."""
    r, b = ndi.binary_dilation(rouge, iterations=2), ndi.binary_dilation(bleu, iterations=2)
    sortie = np.full(len(segs), -1)
    for i, s in enumerate(segs):
        nr, nb = r[s["lignes"], s["colonnes"]].sum(), b[s["lignes"], s["colonnes"]].sum()
        if nr + nb >= 0.5 * s["longueur"] and nr != nb:
            sortie[i] = int(nr > nb)
    return sortie


def calibres_segments(segs: list[dict], arteres: np.ndarray, diametre: float) -> dict[str, float]:
    """CRAE, CRVE, AVR quand chaque segment a déjà sa classe (True = artère)."""
    carte = np.full((max(s["lignes"].max() for s in segs) + 1, max(s["colonnes"].max() for s in segs) + 1), np.nan)
    for s, a in zip(segs, arteres, strict=True):
        carte[s["lignes"], s["colonnes"]] = float(a)
    return calibres_av(segs, carte, diametre)


MESURES = ("crae_um", "crve_um", "avr")


def main() -> int:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--racine", type=Path, required=True)
    parser.add_argument("--modele", type=Path, default=RACINE / "modeles" / "unet_drive.onnx")
    parser.add_argument("--parties", default="training,test")
    parser.add_argument("--sortie", type=Path, default=RACINE / "resultats" / "arteres_veines.json")
    args = parser.parse_args()
    session = ort.InferenceSession(str(args.modele), providers=["CPUExecutionProvider"])
    resultats: dict = {}
    for base in ("DRIVE_AV", "HRF-AV", "LES-AV"):
        for partie in args.parties.split(","):
            images = sorted(glob.glob(str(args.racine / base / partie / "images" / "*")))
            etiquettes = sorted(glob.glob(str(args.racine / base / partie / "1st_manual" / "*")))
            lignes = []
            for image, etiquette in zip(images, etiquettes, strict=True):
                rgb, rouge, bleu, vaisseaux_expert = charger(image, etiquette)
                masque = champ(rgb)
                proba, p_arteres = segmenter(session, rgb, masque)
                vaisseaux_unet = (proba >= SEUIL_UNET) & masque
                disque = papille(rgb, masque, vaisseaux_unet)
                dp = disque["diametre"]
                # référence : vaisseaux et étiquettes de l'expert
                segs_ref = segments_zone(vaisseaux_expert & masque, disque)
                reference = calibres_av(segs_ref, carte_expert(rouge, bleu), dp)
                ligne = {"image": Path(image).name} | {f"reference_{k}": reference[k] for k in MESURES}
                for nom, vaisseaux in (("expert", vaisseaux_expert & masque), ("unet", vaisseaux_unet)):
                    segs = segments_zone(vaisseaux, disque)
                    if len(segs) < 4:
                        continue
                    verite = verite_segments(segs, rouge, bleu)
                    poids = np.array([s["longueur"] for s in segs])
                    angles = np.array([s["angle"] for s in segs])
                    carac = caracteristiques(rgb, vaisseaux, segs)
                    classements = {
                        "quadrants": classer_arteres(carac, angles, True),
                        "global": classer_arteres(carac, angles, False),
                    }
                    if p_arteres is not None:
                        classements["modele"] = np.array(
                            [np.mean(p_arteres[s["lignes"], s["colonnes"]]) >= 0.5 for s in segs]
                        )
                    for variante, arteres in classements.items():
                        ok = verite >= 0
                        ligne[f"{nom}_{variante}_exactitude"] = (
                            float(np.average((arteres[ok] == (verite[ok] == 1)), weights=poids[ok]))
                            if ok.any()
                            else float("nan")
                        )
                        for k, v in calibres_segments(segs, arteres, dp).items():
                            ligne[f"{nom}_{variante}_{k}"] = v
                lignes.append(ligne)
            resultats[f"{base}/{partie}"] = lignes
            print(f"{base:9s} {partie:8s} {len(lignes)} images", flush=True)

    def synthese_de(lignes: list[dict]) -> dict:
        s = {}
        variantes = sorted({k.split("_")[1] for li in lignes for k in li if k.endswith("_exactitude")})
        for nom in ("expert", "unet"):
            for variante in variantes:
                ex = np.array([li.get(f"{nom}_{variante}_exactitude", np.nan) for li in lignes])
                d = {"exactitude": float(np.nanmean(ex))}
                for k in MESURES:
                    a = np.array([li.get(f"{nom}_{variante}_{k}", np.nan) for li in lignes])
                    ref = np.array([li[f"reference_{k}"] for li in lignes])
                    ok = np.isfinite(a) & np.isfinite(ref)
                    d[f"{k}_spearman"] = float(spearmanr(a[ok], ref[ok]).statistic) if ok.sum() > 3 else None
                    d[f"{k}_ecart_relatif_median"] = (
                        float(np.median(np.abs(a[ok] - ref[ok]) / ref[ok])) if ok.any() else None
                    )
                    d[f"{k}_images"] = int(ok.sum())
                s[f"{nom}_{variante}"] = d
        return s

    synthese = {cle: synthese_de(lignes) for cle, lignes in resultats.items()}
    for partie in args.parties.split(","):
        synthese[f"toutes/{partie}"] = synthese_de([li for c, v in resultats.items() if c.endswith(partie) for li in v])
    args.sortie.write_text(
        json.dumps(
            {"modele": args.modele.name, "synthese": synthese, "images": resultats}, indent=1, ensure_ascii=False
        )
    )
    print(
        f"\n{'base/partie':18s} {'vaisseaux':9s} {'variante':10s} {'exactitude':>10s} {'AVR ρ':>6s} {'CRAE ρ':>7s} {'CRVE ρ':>7s}"
    )
    for cle, s in synthese.items():
        for k, v in s.items():
            nom, variante = k.split("_")
            rho = [
                "—" if v[f"{m}_spearman"] is None else f"{v[f'{m}_spearman']:.2f}"
                for m in ("avr", "crae_um", "crve_um")
            ]
            print(f"{cle:18s} {nom:9s} {variante:10s} {v['exactitude']:10.1%} {rho[0]:>6s} {rho[1]:>7s} {rho[2]:>7s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
