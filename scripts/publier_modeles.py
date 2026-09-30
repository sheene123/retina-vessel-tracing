"""Registre de modèles : publie une version (U-Net + modèle des troubles) sur le Hugging Face Hub,
après validation indépendante et comparaison avec la version en production.

    python scripts/publier_modeles.py --version v0.2.0

À lancer sur le poste d'entraînement (GPU), après `python -m vaisseaux.unet`, l'export ONNX,
`python -m vaisseaux.etude` et `python -m vaisseaux.troubles`. Le challenger n'est publié que :
- s'il passe la validation indépendante (Dice du U-Net recalculé sur DRIVE, modèle des troubles
  fonctionnel) ;
- s'il ne régresse pas face au champion (dernière version publiée) : Dice du U-Net, et AUROC de
  chaque trouble sur le test externe JSIEC-1000, recalculée ici pour les deux modèles sur les
  mêmes images (la validation croisée ne compare pas des modèles entraînés sur des jeux
  différents). Petite tolérance pour le bruit d'entraînement.
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
INDEX_EXTERNE = RACINE / "data" / "troubles" / "index.csv"


def auroc_externe(dossier: Path) -> dict[str, float]:
    """AUROC par trouble du modèle d'un dossier du registre sur le test externe JSIEC."""
    sortie = dossier / "externe.json"
    subprocess.run(
        [
            sys.executable,
            str(RACINE / "scripts" / "evaluer_externe.py"),
            "--modele",
            str(dossier / "troubles.onnx"),
            "--calibrage",
            str(dossier / "troubles.json"),
            "--index",
            str(INDEX_EXTERNE),
            "--sortie",
            str(sortie),
        ],
        check=True,
        capture_output=True,
    )
    resultat = json.loads(sortie.read_text())
    sortie.unlink()
    return {f"auroc_externe_{c}": r["auroc"] for c, r in resultat.items() if "auroc" in r}


def indicateurs(dossier: Path) -> dict[str, float]:
    metriques = json.loads((dossier / "metriques.json").read_text())
    return {"dice_unet": metriques["unet"]["dice"], **auroc_externe(dossier)}


def main() -> int:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--repo", default="sheenee261/retina-vessel-tracing")
    parser.add_argument("--forcer", action="store_true", help="publie malgré une régression (décision humaine)")
    parser.add_argument(
        "--note", help="décision et justification, écrites dans la fiche du modèle (obligatoire avec --forcer)"
    )
    args = parser.parse_args()

    manquants = [f for f in FICHIERS if not (RACINE / f).exists()]
    if not INDEX_EXTERNE.exists():
        manquants.append(f"{INDEX_EXTERNE} (python -m vaisseaux.sources_troubles)")
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
        interne = json.loads((dossier / "troubles_resultats.json").read_text())["troubles"]
        faibles = [c for c, r in interne.items() if r["auroc"] < AUROC_MIN]
        if faibles:
            print(f"AUROC sous {AUROC_MIN} : {faibles}", file=sys.stderr)
            return 2

        # 2. comparaison avec le champion
        try:
            depot = Path(snapshot_download(args.repo, repo_type="model", token=api.token))
            with tempfile.TemporaryDirectory() as tmp_champion:  # hors du dossier publié
                copie = Path(tmp_champion)
                for fichier in ("metriques.json", "troubles.onnx", "troubles.json"):
                    (copie / fichier).write_bytes((depot / fichier).read_bytes())
                champion = indicateurs(copie)
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
        if not accepte and args.forcer and not args.note:
            print("--forcer sans --note : la décision doit être justifiée par écrit", file=sys.stderr)
            return 2
        if not accepte and not args.forcer:
            print("régression face à la version en production : publication refusée", file=sys.stderr)
            return 2

        # 3. publication et étiquette de version
        decision = ""
        if not accepte:
            decision = (
                "## Décision de publication\n\nLa comparaison automatique avec la version en production a relevé une "
                "régression :\n\n"
                + "\n".join(f"- {ligne}" for ligne in lignes if "RÉGRESSION" in ligne)
                + f"\n\nPublication décidée malgré tout : {args.note}\n\n"
            )
        resultats = json.loads((dossier / "troubles_resultats.json").read_text())
        externe = resultats.get("externe_jsiec", {})
        noms_bases = {"odir": "ODIR-5K", "rfmid": "RFMiD", "smdg": "SMDG-19", "sjchoi86": "sjchoi86"}

        def fr(v: float, n: int = 2) -> str:
            return f"{v:.{n}f}".replace(".", ",")

        lignes_tableau = []
        for c, t in resultats["troubles"].items():
            e = externe.get(c, {})
            externe_texte = (
                f"{fr(e['auroc'], 3)} [{fr(e['ic_bas'])} ; {fr(e['ic_haut'])}], {e['atteints']} cas"
                if "auroc" in e
                else "pas de cas"
            )
            alertes = f"{round(100 * t['fausses_alertes'])} %" if "fausses_alertes" in t else "n/d"
            lignes_tableau.append(
                f"| {t['nom']} | {fr(t['auroc'])} [{fr(t['ic_bas'])} ; {fr(t['ic_haut'])}] | "
                f"{round(100 * t['detectes_des_possible'])} % | {alertes} | {externe_texte} |"
            )
        tableau = "\n".join(lignes_tableau)
        sources = ", ".join(
            f"{noms_bases.get(s, s)} ({n:,})".replace(",", "\u202f")
            for s, n in resultats.get("effectifs", {}).get("par_source", {}).items()
        )
        (dossier / "README.md").write_text(f"""---
license: mit
library_name: onnx
tags: [medical-imaging, retina, vessel-segmentation, fundus, onnx]
---

# Tracé des vaisseaux rétiniens, modèles {args.version}

Modèles de la démo [retina-vessel-tracing](https://huggingface.co/spaces/sheenee261/retina-vessel-tracing)
(code : [sheene123/retina-vessel-tracing](https://github.com/sheene123/retina-vessel-tracing), commit `{commit}`).

- `unet_drive.onnx` : U-Net de segmentation des vaisseaux, entraîné sur DRIVE. Dice {fr(mesure["unet"]["dice"], 3)}, AUC {fr(mesure["unet"]["auc"], 3)}
  sur les 20 images de test (recalculés indépendamment avant publication).
- `troubles.onnx` : {resultats.get("architecture", "efficientnet_b0").split(".")[0]}, 6 troubles de l'œil, entraîné sur {sources or "ODIR-5K"} ;
  sorties calibrées et seuils « possible » par trouble dans `troubles.json`.

| Trouble | AUROC validation croisée [IC 95 %] | Atteints repérés dès « possible » | Fausses alertes (yeux sans le trouble signalés) | AUROC hôpital jamais vu (JSIEC) |
|---|---|---|---|---|
{tableau}

**Comment lire ces chiffres.** La validation croisée porte sur des milliers d'yeux que le réseau n'avait pas vus, de
gravité et de qualité variables : c'est la mesure la plus représentative. L'AUROC mesure le classement (1 si chaque œil
atteint a un score plus haut que chaque œil sain) ; elle ne dit pas qu'il n'y a pas d'erreur : au seuil « possible »,
il reste des cas manqués et des fausses alertes. Le test externe JSIEC-1000 (un hôpital jamais utilisé pour
l'entraînement) est optimiste : ses photos montrent surtout des cas typiques, souvent avancés, et certains troubles y ont
peu de cas, d'où des valeurs proches de 1 et des intervalles de confiance trop étroits.

{decision}Démonstration de recherche, pas un dispositif médical.
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
