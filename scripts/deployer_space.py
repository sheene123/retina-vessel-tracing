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
    parser.add_argument("--space", help="identifiant du Space, ex. utilisateur/retina-vessel-tracing")
    parser.add_argument("--local", type=Path, help="assemble le site dans ce dossier, sans le publier (test)")
    parser.add_argument("--message", default="Met à jour la démo web")
    parser.add_argument("--depuis-registre", metavar="VERSION", help="prend les modèles de cette version du registre")
    parser.add_argument("--registre", default="sheenee261/retina-vessel-tracing")
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
        if args.depuis_registre:
            # modèles d'une version du registre (déploiement continu)
            from huggingface_hub import hf_hub_download

            for fichier in ("unet_drive.onnx", "fiabilite.json", "troubles.onnx", "troubles.json"):
                chemin = hf_hub_download(args.registre, fichier, repo_type="model", revision=args.depuis_registre)
                shutil.copy(chemin, site / fichier)
        else:
            # modèles locaux (poste d'entraînement)
            for source, cible in (
                (RACINE / "modeles" / "unet_drive.onnx", "unet_drive.onnx"),
                (RACINE / "resultats" / "etude_metriques" / "fiabilite_unet.json", "fiabilite.json"),
                (RACINE / "modeles" / "troubles.onnx", "troubles.onnx"),
                (RACINE / "resultats" / "troubles" / "troubles_demo.json", "troubles.json"),
            ):
                if not source.exists():
                    raise SystemExit(
                        f"{source} manquant : entraînez et exportez les modèles, ou utilisez --depuis-registre"
                    )
                shutil.copy(source, site / cible)
        if args.local:
            shutil.copytree(site, args.local, dirs_exist_ok=True)
            print(f"site assemblé dans {args.local}")
            return 0
        if not args.space:
            raise SystemExit("--space ou --local est requis")
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
