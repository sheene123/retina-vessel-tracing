"""Validation indépendante des modèles ONNX (sans PyTorch), utilisée avant publication et en CI.

- U-Net : Dice et AUC recalculés sur les 20 images de test DRIVE, jamais vues à l'entraînement.
- Modèle des troubles : il doit charger, renvoyer 6 valeurs finies par image et être cohérent
  avec son fichier de calibrage.

    python scripts/valider_modeles.py --dossier modeles/            # modèles locaux
    python scripts/valider_modeles.py --version v0.1.0 --repo ...    # une version du registre
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from vaisseaux.donnees import charger, entree_imagenet, lister, preparer_fond_oeil  # noqa: E402
from vaisseaux.evaluation import auc_roc, metriques_binaires  # noqa: E402
from vaisseaux.pretraitement import normaliser_pour_reseau  # noqa: E402

SEUIL_DICE = 0.78  # quality gate absolue du U-Net sur DRIVE


def valider_unet(chemin: Path, drive: Path) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"])
    dices, aucs = [], []
    for ident in lister(drive, "test"):
        im = charger(drive, "test", ident)
        x = normaliser_pour_reseau(im.rgb, im.masque)[None]
        logits = session.run(None, {"image": x})[0][0][: im.masque.shape[0], : im.masque.shape[1]]
        proba = 1 / (1 + np.exp(-logits)) * im.masque
        dices.append(metriques_binaires(proba >= 0.5, im.verite, im.masque)["dice"])
        aucs.append(auc_roc(proba, im.verite, im.masque))
    return {"dice": float(np.mean(dices)), "auc": float(np.mean(aucs)), "images": len(dices)}


def valider_troubles(chemin: Path, configuration: dict, drive: Path) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"])
    ordre = configuration["ordre"]
    probabilites = []
    for ident in lister(drive, "test")[:3]:
        x = entree_imagenet(preparer_fond_oeil(charger(drive, "test", ident).rgb))[None]
        z = session.run(None, {"image": x})[0][0]
        if z.shape != (len(ordre),) or not np.all(np.isfinite(z)):
            raise ValueError(f"sortie invalide du modèle des troubles : {z.shape}")
        probabilites.append(
            {
                c: float(
                    1 / (1 + np.exp(-(configuration["troubles"][c]["a"] * z[j] + configuration["troubles"][c]["b"])))
                )
                for j, c in enumerate(ordre)
            }
        )
    return {"troubles": ordre, "exemple": probabilites[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dossier", type=Path, help="dossier contenant les fichiers du registre")
    parser.add_argument("--version", help="version du registre à valider (téléchargée)")
    parser.add_argument("--repo", default="sheenee261/retina-vessel-tracing")
    parser.add_argument("--drive", type=Path, default=RACINE / "data" / "DRIVE")
    parser.add_argument("--sortie", type=Path, help="écrit le résultat en JSON")
    args = parser.parse_args()

    if args.version:
        from huggingface_hub import snapshot_download

        dossier = Path(snapshot_download(args.repo, repo_type="model", revision=args.version))
    else:
        dossier = args.dossier or RACINE / "modeles"
    unet = valider_unet(dossier / "unet_drive.onnx", args.drive)
    resultat = {"unet": unet}
    if (dossier / "troubles.onnx").exists():
        chemin_configuration = dossier / "troubles.json"
        if not chemin_configuration.exists() and dossier.resolve() == (RACINE / "modeles").resolve():
            chemin_configuration = RACINE / "resultats" / "troubles" / "troubles_demo.json"
        configuration = json.loads(chemin_configuration.read_text())
        resultat["troubles"] = valider_troubles(dossier / "troubles.onnx", configuration, args.drive)
    annonces = dossier / "metriques.json"
    if annonces.exists():  # le registre annonce des métriques : elles doivent être retrouvées
        attendu = json.loads(annonces.read_text())["unet"]["dice"]
        resultat["unet"]["ecart_avec_registre"] = unet["dice"] - attendu
    ok = unet["dice"] >= SEUIL_DICE and abs(resultat["unet"].get("ecart_avec_registre", 0)) < 0.005
    resultat["valide"] = bool(ok)
    texte = json.dumps(resultat, indent=1, ensure_ascii=False)
    print(texte)
    if args.sortie:
        args.sortie.write_text(texte)
    resume = os.environ.get("GITHUB_STEP_SUMMARY")
    if resume:
        with open(resume, "a") as f:
            f.write(
                f"## Validation indépendante\n\nU-Net sur DRIVE (20 images de test) : Dice {unet['dice']:.3f}, AUC {unet['auc']:.3f}. "
                f"{'Validé' if ok else 'ÉCHEC'}.\n"
            )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
