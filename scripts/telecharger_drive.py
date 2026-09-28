"""Télécharge DRIVE et le range selon l'arborescence officielle.

Deux sources possibles :
- l'archive officielle de https://drive.grand-challenge.org (inscription requise) :
  `python scripts/telecharger_drive.py --zip DRIVE.zip` ;
- par défaut, un miroir public Hugging Face (images + annotations du 1er expert,
  sans masques FOV ni 2e observateur ; les masques sont alors estimés à la volée).

Résultat : data/DRIVE/{training,test}/{images,1st_manual}/...
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

MIROIR = "https://huggingface.co/datasets/Zomba/DRIVE-digital-retinal-images-for-vessel-extraction/resolve/main"
# partie officielle -> (dossier du miroir, identifiants, suffixe officiel, nom d'annotation dans le miroir)
PARTIES = {
    "training": ("train", range(21, 41), "training", "{}.png"),
    "test": ("val", range(1, 21), "test", "{}_manual1.png"),
}


def depuis_miroir(destination: Path) -> None:
    for partie, (dossier_miroir, identifiants, suffixe, annotation) in PARTIES.items():
        for n in identifiants:
            ident = f"{n:02d}"
            fichiers = {
                f"{dossier_miroir}/input/{ident}.tif": destination / partie / "images" / f"{ident}_{suffixe}.tif",
                f"{dossier_miroir}/label/{annotation.format(ident)}": destination
                / partie
                / "1st_manual"
                / f"{ident}_manual1.png",
            }
            for source, cible in fichiers.items():
                if cible.exists():
                    continue
                cible.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(f"{MIROIR}/{source}", timeout=60) as reponse, cible.open("wb") as f:
                    shutil.copyfileobj(reponse, f)
            print(f"{partie} {ident}", end="\r", flush=True)
    print()


def depuis_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        for membre in z.namelist():
            parties = Path(membre).parts
            if "training" in parties or "test" in parties:
                debut = parties.index("training") if "training" in parties else parties.index("test")
                cible = destination.joinpath(*parties[debut:])
                if membre.endswith("/"):
                    continue
                cible.parent.mkdir(parents=True, exist_ok=True)
                cible.write_bytes(z.read(membre))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, help="archive officielle DRIVE.zip")
    parser.add_argument("--destination", type=Path, default=Path("data/DRIVE"))
    args = parser.parse_args()
    if args.zip:
        depuis_zip(args.zip, args.destination)
    else:
        depuis_miroir(args.destination)
    n = len(list(args.destination.glob("*/images/*")))
    print(f"{n} images dans {args.destination}")
    return 0 if n == 40 else 1


if __name__ == "__main__":
    sys.exit(main())
