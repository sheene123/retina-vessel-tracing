"""Prépare le jeu d'entraînement du U-Net multi-appareils (vaisseaux + artères/veines).

Chaque photo suit le chemin de la démo : isolement de l'œil, réduction à 800 pixels au plus. Les
annotations sont réduites de la même façon (part de chaque classe par pixel, vaisseau si au moins
la moitié du pixel est couverte). Codes des étiquettes :

    0 fond, 1 vaisseau de classe inconnue, 2 artère, 3 veine, 4 croisement

Bases et découpage (les parties « test » ne servent jamais à l'entraînement ni au choix du modèle) :

- DRIVE_AV, HRF-AV, LES-AV (artères en rouge, veines en bleu) : parties officielles training/test ;
- FIVES (800 images, vaisseaux seulement) : parties officielles train/test ;
- CHASE_DB1 (28 images, 14 enfants) : patients 1 à 10 en entraînement, 11 à 14 en test.

Une petite validation (choix de l'itération) est prise dans les parties d'entraînement.

    python scripts/preparer_vaisseaux_multi.py --av <data de muflihsan/retina-av-dataset> \
        --fives <FIVES> --chase <CHASE_DB1> --sortie ~/kaggle_av/donnees
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from vaisseaux.donnees import isoler_fond_oeil  # noqa: E402

COTE_MAX = 800
FOND, INCONNU, ARTERE, VEINE, CROISEMENT = range(5)
VALIDATION = {"DRIVE_AV": 4, "HRF-AV": 3, "LES-AV": 2, "CHASE_DB1": 2, "FIVES": 24}  # images (patients pour CHASE)


def classes_av(rgb: np.ndarray) -> np.ndarray:
    """Annotation colorée -> codes : rouge artère, bleu veine, vert croisement, blanc incertain."""
    r, v, b = (rgb[..., k] > 127 for k in range(3))
    code = np.zeros(rgb.shape[:2], np.uint8)
    code[r | v | b] = INCONNU
    code[r & ~v & ~b] = ARTERE
    code[b & ~r & ~v] = VEINE
    code[v & ~r & ~b] = CROISEMENT
    return code


def reduire_etiquette(code: np.ndarray, taille: tuple[int, int]) -> np.ndarray:
    if code.shape[::-1] == taille:
        return code
    parts = [
        np.asarray(Image.fromarray((code == k).astype(np.float32)).resize(taille, Image.Resampling.BOX))
        for k in range(1, 5)
    ]
    parts = np.stack(parts)
    sortie = (np.argmax(parts, axis=0) + 1).astype(np.uint8)
    sortie[parts.sum(axis=0) < 0.5] = FOND
    return sortie


def preparer(chemin_image: str, code: np.ndarray, dossier: Path, nom: str) -> dict | None:
    rgb = np.asarray(Image.open(chemin_image).convert("RGB"))
    if code.shape != rgb.shape[:2]:
        raise ValueError(f"{chemin_image} : annotation {code.shape} contre image {rgb.shape[:2]}")
    image, oeil, cadre = isoler_fond_oeil(rgb)
    if cadre is None:
        print(f"ignorée (pas de fond d'œil trouvé) : {chemin_image}", flush=True)
        return None
    haut, bas, gauche, droite = cadre
    code = code[haut:bas, gauche:droite]
    echelle = max(image.shape[:2]) / COTE_MAX
    if echelle > 1:
        taille = (round(image.shape[1] / echelle), round(image.shape[0] / echelle))
        image = np.asarray(Image.fromarray(image).resize(taille, Image.Resampling.LANCZOS))
        oeil = np.asarray(Image.fromarray(oeil).resize(taille, Image.Resampling.NEAREST))
        code = reduire_etiquette(code, taille)
    code = np.where(oeil, code, FOND).astype(np.uint8)
    dossier.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(dossier / f"{nom}.png")
    Image.fromarray(code).save(dossier / f"{nom}_etiquette.png")
    Image.fromarray(oeil.astype(np.uint8) * 255).save(dossier / f"{nom}_masque.png")
    return {
        "hauteur": image.shape[0],
        "largeur": image.shape[1],
        "part_vaisseaux": float((code > 0).sum() / oeil.sum()),
    }


def lister(args) -> list[dict]:
    """(source, partie, groupe, image, annotation, av) pour toutes les photos."""
    lignes = []
    for base in ("DRIVE_AV", "HRF-AV", "LES-AV"):
        for partie_source, partie in (("training", "entrainement"), ("test", "test")):
            images = sorted(glob.glob(str(args.av / base / partie_source / "images" / "*")))
            etiquettes = sorted(glob.glob(str(args.av / base / partie_source / "1st_manual" / "*")))
            for image, etiquette in zip(images, etiquettes, strict=True):
                nom = Path(image).stem
                lignes.append(dict(source=base, partie=partie, groupe=nom, image=image, annotation=etiquette, av=True))
    for partie_source, partie in (("train", "entrainement"), ("test", "test")):
        for image in sorted(glob.glob(str(args.fives / partie_source / "Original" / "*.png"))):
            nom = Path(image).stem  # la numérotation repart de 1 dans chaque partie
            etiquette = args.fives / partie_source / "Ground truth" / f"{nom}.png"
            lignes.append(
                dict(
                    source="FIVES",
                    partie=partie,
                    groupe=f"{partie_source}_{nom}",
                    image=image,
                    annotation=str(etiquette),
                    av=False,
                )
            )
    for image in sorted(glob.glob(str(args.chase / "Images" / "*.jpg"))):
        nom = Path(image).stem  # Image_01L
        patient = int(nom.split("_")[1][:2])
        etiquette = args.chase / "Masks" / f"{nom}_1stHO.png"
        lignes.append(
            dict(
                source="CHASE_DB1",
                partie="entrainement" if patient <= 10 else "test",
                groupe=f"patient{patient:02d}",
                image=image,
                annotation=str(etiquette),
                av=False,
            )
        )
    return lignes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--av", type=Path, required=True)
    parser.add_argument("--fives", type=Path, required=True)
    parser.add_argument("--chase", type=Path, required=True)
    parser.add_argument("--sortie", type=Path, required=True)
    args = parser.parse_args()
    lignes = lister(args)
    # validation : quelques groupes (patients) des parties d'entraînement, tirés au hasard par source
    rng = np.random.default_rng(2026)
    for source, n in VALIDATION.items():
        groupes = sorted({li["groupe"] for li in lignes if li["source"] == source and li["partie"] == "entrainement"})
        for g in rng.choice(groupes, size=n, replace=False):
            for li in lignes:
                if li["source"] == source and li["groupe"] == g and li["partie"] == "entrainement":
                    li["partie"] = "validation"
    index = []
    for k, li in enumerate(lignes):
        annotation = np.asarray(Image.open(li["annotation"]).convert("RGB"))
        code = (
            classes_av(annotation)
            if li["av"]
            else np.where(annotation.max(axis=-1) > 127, INCONNU, FOND).astype(np.uint8)
        )
        nom = Path(li["image"]).stem
        dossier = args.sortie / "images" / li["source"] / li["partie"]
        infos = preparer(li["image"], code, dossier, nom)
        if infos is None:
            continue
        index.append(
            {
                "source": li["source"],
                "partie": li["partie"],
                "groupe": li["groupe"],
                "av": li["av"],
                "chemin": f"images/{li['source']}/{li['partie']}/{nom}",
                **infos,
            }
        )
        if k % 50 == 0:
            print(f"{k + 1}/{len(lignes)} {li['source']} {nom}", flush=True)
    table = pd.DataFrame(index)
    table.to_csv(args.sortie / "index.csv", index=False)
    print(table.groupby(["source", "partie"]).size().unstack(fill_value=0))
    print(table.groupby("source")[["hauteur", "largeur", "part_vaisseaux"]].median())
    return 0


if __name__ == "__main__":
    sys.exit(main())
