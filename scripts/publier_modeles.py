"""Registre de modèles : publie une version (U-Net + modèle des troubles) sur le Hugging Face Hub,
après validation indépendante et comparaison avec la version en production.

    python scripts/publier_modeles.py --version v0.2.0

À lancer sur le poste d'entraînement (GPU), après `python -m vaisseaux.unet`, l'export ONNX,
`python -m vaisseaux.etude` et `python -m vaisseaux.troubles`. Le challenger n'est publié que :
- s'il passe la validation indépendante (Dice du U-Net recalculé sur DRIVE, modèle des troubles
  fonctionnel) ;
- s'il ne régresse pas face au champion (dernière version publiée) : Dice du U-Net et AUROC de
  chaque trouble, avec une petite tolérance pour le bruit d'entraînement.
Code de sortie : 0 si publié, 2 si refusé, 1 en cas d'erreur.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
FICHIERS = {
    "modeles/unet_drive.onnx": "unet_drive.onnx",
    "resultats/etude_metriques/fiabilite_unet.json": "fiabilite.json",
    "modeles/troubles.onnx": "troubles.onnx",
    "resultats/troubles/troubles_demo.json": "troubles.json",
    "resultats/troubles/resultats.json": "troubles_resultats.json",
}
TOLERANCE_DICE, TOLERANCE_AUROC, AUROC_MIN = 0.01, 0.02, 0.75


def indicateurs(dossier: Path) -> dict[str, float]:
    metriques = json.loads((dossier / "metriques.json").read_text())
    troubles = json.loads((dossier / "troubles_resultats.json").read_text())["troubles"]
    return {"dice_unet": metriques["unet"]["dice"], **{f"auroc_{c}": t["auroc"] for c, t in troubles.items()}}


def main() -> int:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--repo", default="sheenee261/retina-vessel-tracing")
    parser.add_argument("--forcer", action="store_true")
    args = parser.parse_args()

    manquants = [f for f in FICHIERS if not (RACINE / f).exists()]
    if manquants:
        print(f"fichiers absents : {', '.join(manquants)}", file=sys.stderr)
        return 1
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    with tempfile.TemporaryDirectory() as tmp:
        dossier = Path(tmp)
        for source, cible in FICHIERS.items():
            (dossier / cible).write_bytes((RACINE / source).read_bytes())
        # 1. validation indépendante du challenger
        validation = subprocess.run(
            [
                sys.executable,
                str(RACINE / "scripts" / "valider_modeles.py"),
                "--dossier",
                str(dossier),
                "--sortie",
                str(dossier / "validation.json"),
            ],
            capture_output=True,
            text=True,
        )
        if validation.returncode != 0:
            print(f"validation indépendante en échec :\n{validation.stdout}{validation.stderr}", file=sys.stderr)
            return 2
        mesure = json.loads((dossier / "validation.json").read_text())
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=RACINE, capture_output=True, text=True
        ).stdout.strip()
        (dossier / "metriques.json").write_text(
            json.dumps({"version": args.version, "commit": commit, "unet": mesure["unet"]}, indent=1)
        )
        challenger = indicateurs(dossier)
        faibles = [k for k, v in challenger.items() if k.startswith("auroc_") and v < AUROC_MIN]
        if faibles:
            print(f"AUROC sous {AUROC_MIN} : {faibles}", file=sys.stderr)
            return 2

        # 2. comparaison avec le champion
        try:
            champion = indicateurs(Path(snapshot_download(args.repo, repo_type="model", token=api.token)))
        except (RepositoryNotFoundError, RevisionNotFoundError, FileNotFoundError):
            champion = None
        lignes, accepte = [], True
        if champion is None:
            lignes.append("aucune version en production : première publication")
        else:
            for cle, valeur in challenger.items():
                tolerance = TOLERANCE_DICE if cle == "dice_unet" else TOLERANCE_AUROC
                ecart = valeur - champion.get(cle, valeur)
                ok = ecart >= -tolerance
                accepte &= ok
                lignes.append(
                    f"{cle} : champion {champion.get(cle, float('nan')):.3f}, challenger {valeur:.3f} ({ecart:+.3f}) {'OK' if ok else 'RÉGRESSION'}"
                )
        print("Comparaison champion / challenger :\n" + "\n".join(f"- {ligne}" for ligne in lignes))
        if not accepte and not args.forcer:
            print("régression face à la version en production : publication refusée", file=sys.stderr)
            return 2

        # 3. publication et étiquette de version
        troubles = json.loads((dossier / "troubles_resultats.json").read_text())["troubles"]
        tableau = "\n".join(
            f"| {t['nom']} | {t['auroc']:.2f} [{t['ic_bas']:.2f} ; {t['ic_haut']:.2f}] |" for t in troubles.values()
        )
        (dossier / "README.md").write_text(f"""---
license: mit
library_name: onnx
tags: [medical-imaging, retina, vessel-segmentation, fundus, onnx]
---

# Tracé des vaisseaux rétiniens, modèles {args.version}

Modèles de la démo [retina-vessel-tracing](https://huggingface.co/spaces/sheenee261/retina-vessel-tracing)
(code : [sheene123/retina-vessel-tracing](https://github.com/sheene123/retina-vessel-tracing), commit `{commit}`).

- `unet_drive.onnx` : U-Net de segmentation des vaisseaux, entraîné sur DRIVE. Dice {mesure["unet"]["dice"]:.3f}, AUC {mesure["unet"]["auc"]:.3f}
  sur les 20 images de test (recalculés indépendamment avant publication).
- `troubles.onnx` : EfficientNet-B0, 6 troubles de l'œil, entraîné sur ODIR-5K ; sorties calibrées par `troubles.json`.

| Trouble | AUROC hors pli [IC 95 %] |
|---|---|
{tableau}

Démonstration de recherche, pas un dispositif médical.
""")
        (dossier / "validation.json").unlink()
        info = api.create_repo(args.repo, repo_type="model", exist_ok=True)
        del info
        commit_hub = api.upload_folder(
            folder_path=str(dossier), repo_id=args.repo, repo_type="model", commit_message=f"Publie {args.version}"
        )
        api.create_tag(args.repo, tag=args.version, revision=commit_hub.oid, repo_type="model", exist_ok=True)
    print(f"publié : https://huggingface.co/{args.repo}/tree/{args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
