"""Publie la démo web (Space statique Hugging Face, calcul dans le navigateur via Pyodide).

hf auth login                      # une fois, avec un token « write »
python scripts/deployer_space.py --space <utilisateur>/retina-vessel-tracing
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

RACINE = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="identifiant du Space, ex. utilisateur/retina-vessel-tracing")
    parser.add_argument("--message", default="Met à jour la démo web")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as construction, tempfile.TemporaryDirectory() as dossier:
        # uv dépose un .gitignore « * » dans son dossier de sortie, que l'envoi respecterait :
        # on construit à part et on ne copie que la wheel.
        subprocess.run(["uv", "build", "--wheel", "-q", "-o", construction, str(RACINE)], check=True)
        site = Path(dossier)
        for roue in Path(construction).glob("*.whl"):
            shutil.copy(roue, site / roue.name)
        for fichier in ("index.html", "pont.py", "README.md"):
            shutil.copy(RACINE / "demo" / "web" / fichier, site / fichier)
        # U-Net exporté (python -m vaisseaux.unet --onnx modeles/unet_drive.onnx) et fiabilité issue de l'étude
        for source, cible in (
            (RACINE / "modeles" / "unet_drive.onnx", "unet_drive.onnx"),
            (RACINE / "resultats" / "etude_metriques" / "fiabilite_unet.json", "fiabilite.json"),
        ):
            if not source.exists():
                raise SystemExit(f"{source} manquant : entraînez et exportez le U-Net, puis lancez l'étude")
            shutil.copy(source, site / cible)
        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(
            folder_path=str(site),
            repo_id=args.space,
            repo_type="space",
            commit_message=args.message,
            delete_patterns=["style.css"],  # fichier du modèle de Space statique
        )
    print(f"https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
