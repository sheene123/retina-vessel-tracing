"""Publie la démo web (Space statique Hugging Face, calcul dans le navigateur via Pyodide).

hf auth login                      # une fois, avec un token « write »
python scripts/deployer_space.py --space <utilisateur>/retina-vessel-tracing
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

RACINE = Path(__file__).resolve().parents[1]
# fichiers de la démo -> noms possibles dans le registre, du plus récent au plus ancien. Le U-Net
# est toujours servi sous « unet.onnx » : la page reconnaît le modèle multi-appareils à sa sortie
# « arteres », et un retour à une ancienne version (unet_drive.onnx) reste possible.
REGISTRE = {
    "unet.onnx": ("unet_av.onnx", "unet_drive.onnx"),
    "fiabilite.json": ("fiabilite.json",),
    "troubles.onnx": ("troubles.onnx",),
    "troubles.json": ("troubles.json",),
}
OPTIONNELS = ("fiabilite_zones.json", "reperes_zones.json")  # absents des versions avant v0.4.0
LOCAL = {
    "unet.onnx": ("modeles/unet_av.onnx", "modeles/unet_drive.onnx"),
    "fiabilite.json": (
        "resultats/etude_metriques_unet_av/fiabilite_unet.json",
        "resultats/etude_metriques/fiabilite_unet.json",
    ),
    "troubles.onnx": ("modeles/troubles.onnx",),
    "troubles.json": ("resultats/troubles/troubles_demo.json",),
    "fiabilite_zones.json": ("resultats/fiabilite_zones.json",),
    "reperes_zones.json": ("resultats/reperes_zones.json",),
}


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
        exemples = RACINE / "demo" / "web" / "exemples"  # yeux malades à diagnostic connu (JSIEC)
        if exemples.is_dir():
            shutil.copytree(exemples, site / "exemples")
            # liste écrite directement dans la page : les boutons s'affichent sans requête à part
            page = (site / "index.html").read_text()
            balise = '<script type="application/json" id="exemplesMaladesDonnees">null</script>'
            liste = json.dumps(json.loads((exemples / "exemples.json").read_text()), ensure_ascii=False)
            liste = liste.replace("</", "<\\/")  # jamais de « </script> » dans la liste
            if balise in page:
                page = page.replace(balise, balise.replace(">null<", ">" + liste + "<"))
                (site / "index.html").write_text(page)
        if args.depuis_registre:
            # modèles d'une version du registre (déploiement continu)
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            def telecharger(nom: str) -> Path | None:
                try:
                    return Path(hf_hub_download(args.registre, nom, repo_type="model", revision=args.depuis_registre))
                except EntryNotFoundError:
                    return None

            unet = None
            for cible, noms in REGISTRE.items():
                trouve = next(((n, c) for n in noms if (c := telecharger(n)) is not None), None)
                if trouve is None:
                    raise SystemExit(f"{cible} : aucun de {noms} dans la version {args.depuis_registre}")
                shutil.copy(trouve[1], site / cible)
                if cible == "unet.onnx":
                    unet = trouve[0]
            for nom in OPTIONNELS:
                chemin = telecharger(nom)
                if chemin is not None:
                    shutil.copy(chemin, site / nom)
                elif nom == "fiabilite_zones.json" and unet == "unet_drive.onnx":
                    # ancienne version : fiabilité des zones du premier U-Net, gardée dans le dépôt
                    shutil.copy(RACINE / "resultats" / "fiabilite_zones_unet_drive.json", site / nom)
        else:
            # modèles locaux (poste d'entraînement)
            for cible, sources in LOCAL.items():
                source = next((RACINE / s for s in sources if (RACINE / s).exists()), None)
                if source is None:
                    if cible in OPTIONNELS:
                        continue
                    raise SystemExit(
                        f"{sources[0]} manquant : entraînez et exportez les modèles, ou utilisez --depuis-registre"
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
