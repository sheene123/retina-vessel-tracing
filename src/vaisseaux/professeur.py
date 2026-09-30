"""Professeur RETFound pour le glaucome, dont l'avis est ensuite transmis au petit réseau de la démo.

RETFound (Zhou et al., Nature 2023, version DINOv2 de Moorfields) est un ViT-L pré-entraîné sur
environ 1,6 million de photos du fond d'œil : il « connaît » la rétine, mais ses 300 millions de
paramètres sont trop lourds pour un navigateur. On l'affine donc sur le glaucome, puis ses avis
servent de cibles douces au réseau de la démo (distillation, `troubles --professeur`).

- Validation croisée en deux moitiés (groupées par patient) : chaque image reçoit l'avis d'un
  professeur qui ne l'a pas vue, sinon l'avis recopierait simplement l'étiquette.
- Gros plans de la papille dans 35 % des images d'entraînement, pour les photos prises de près.
- Test externe JSIEC-1000 : image entière et gros plan serré de la papille (moyenne des deux
  professeurs).

Les poids sont lus depuis un fichier safetensors (converti une fois depuis le dépôt officiel, en
mode « poids seulement ») : aucun code n'est exécuté au chargement.

    python -m vaisseaux.professeur --poids retfound_dinov2_meh_224.safetensors
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from vaisseaux.troubles import GRAINE, GROS_PLAN_TEST, _chargeur, _plis, calibrer, lire_index

ARCHITECTURE = "vit_large_patch14_dinov2.lvd142m"
TAILLE = 224
PART_GROS_PLANS = 0.35
DECROISSANCE_COUCHES = 0.75  # les premières couches, les plus génériques, bougent moins


def modele_retfound(poids: Path):
    import timm
    from safetensors.torch import load_file

    modele = timm.create_model(ARCHITECTURE, pretrained=False, img_size=TAILLE, num_classes=1)
    manquants, inattendus = modele.load_state_dict(load_file(str(poids)), strict=False)
    if set(manquants) != {"head.weight", "head.bias"} or inattendus:
        raise ValueError(f"poids RETFound inattendus : manquants {manquants}, inattendus {inattendus}")
    return modele


def transformations(entrainement: bool):
    import torch
    from torchvision.transforms import v2 as T

    fin = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    if not entrainement:
        return T.Compose([T.Resize(TAILLE), *fin])
    return T.Compose(
        [
            T.RandomResizedCrop(TAILLE, scale=(0.4, 1.0), ratio=(0.85, 1.18)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
            T.RandomApply([T.GaussianBlur(7, sigma=(0.1, 2.0))], p=0.2),
            T.RandomApply([T.JPEG((30, 90))], p=0.3),
            *fin,
        ]
    )


def _groupes_parametres(modele, lr: float):
    """Taux d'apprentissage décroissant avec la profondeur (layer-wise decay, comme RETFound)."""
    n = len(modele.blocks) + 1
    groupes: dict[int, list] = {}
    for nom, param in modele.named_parameters():
        if nom.startswith("blocks."):
            niveau = int(nom.split(".")[1]) + 1
        elif nom.startswith(("head", "norm", "fc_norm")):
            niveau = n
        else:
            niveau = 0
        groupes.setdefault(niveau, []).append(param)
    return [{"params": p, "lr": lr * DECROISSANCE_COUCHES ** (n - niveau)} for niveau, p in groupes.items()]


def entrainer(lignes: pd.DataFrame, poids: Path, epoques: int, progression, etape: str):
    import torch

    appareil = "cuda"
    modele = modele_retfound(poids).to(appareil)
    modele.set_grad_checkpointing(True)  # ViT-L sur 16 Go : on recalcule les activations
    y = lignes["glaucome"].to_numpy()
    poids_positif = torch.tensor([min(10.0, (y == 0).sum() / max((y == 1).sum(), 1))], device=appareil)
    charge = _chargeur(
        lignes,
        entrainement=True,
        colonnes=["glaucome"],
        transformation=transformations(True),
        taille_lot=32,
        part_gros_plans=PART_GROS_PLANS,
    )
    optimiseur = torch.optim.AdamW(_groupes_parametres(modele, 2e-4), weight_decay=0.05)
    total = epoques * len(charge)
    chauffe = len(charge) // 2
    planning = torch.optim.lr_scheduler.LambdaLR(
        optimiseur,
        lambda pas: (
            (pas + 1) / chauffe
            if pas < chauffe
            else 0.5 * (1 + math.cos(math.pi * (pas - chauffe) / (total - chauffe)))
        ),
    )
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    echelle = torch.amp.GradScaler("cuda", enabled=not bf16)
    for epoque in range(epoques):
        modele.train()
        progression.set_description(f"{etape} · époque {epoque + 1}/{epoques}")
        for n, (x, cible) in enumerate(charge):
            x, cible = x.to(appareil, non_blocking=True), cible.to(appareil, non_blocking=True)[:, 0]
            with torch.autocast(appareil, dtype=torch.bfloat16 if bf16 else torch.float16):
                sortie = modele(x)[:, 0]
            connu = cible >= 0
            perte = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    sortie.float()[connu], cible[connu], pos_weight=poids_positif
                )
                if connu.any()
                else sortie.sum() * 0
            )
            optimiseur.zero_grad()
            echelle.scale(perte).backward()
            echelle.unscale_(optimiseur)
            torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            echelle.step(optimiseur)
            echelle.update()
            planning.step()
            progression.update(1)
            if n % 50 == 0:
                progression.set_postfix_str(f"perte {perte.item():.3f}", refresh=False)
    return modele.eval()


def predire(modele, lignes: pd.DataFrame, gros_plan: float | None = None) -> np.ndarray:
    import torch

    sorties = []
    charge = _chargeur(
        lignes, entrainement=False, gros_plan=gros_plan, colonnes=["glaucome"], transformation=transformations(False)
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, _ in charge:
            sorties.append(modele(x.to("cuda"))[:, 0].float().cpu().numpy())
    return np.concatenate(sorties)


def _auroc(y: np.ndarray, z: np.ndarray) -> float | None:
    ok = y >= 0
    return float(roc_auc_score(y[ok], z[ok])) if 0 < (y[ok] == 1).sum() < ok.sum() else None


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--index", type=Path, default=Path("data/troubles/index.csv"))
    parser.add_argument("--poids", type=Path, required=True, help="RETFound converti en safetensors (224 × 224)")
    parser.add_argument("--sortie", type=Path, default=Path("resultats/professeur"))
    parser.add_argument("--epoques", type=int, default=5)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        print("le professeur RETFound (ViT-L) demande un GPU", file=sys.stderr)
        return 1
    debut = time.perf_counter()
    torch.manual_seed(GRAINE)
    args.sortie.mkdir(parents=True, exist_ok=True)
    index = lire_index(args.index)
    jeu = index[index["source"] != "jsiec"].reset_index(drop=True)
    externe = index[index["source"] == "jsiec"].reset_index(drop=True)
    moitie = _plis(jeu) % 2
    etiquetes = jeu["glaucome"].to_numpy() >= 0
    lots = sum(int(((moitie != k) & etiquetes).sum()) // 32 for k in (0, 1))
    progression = tqdm(
        total=args.epoques * lots,
        unit="lot",
        mininterval=10,
        ncols=150,
        file=sys.stdout,
        bar_format="{desc} {percentage:3.0f}% |{bar:30}| {elapsed} écoulé, reste ~{remaining} {postfix}",
    )
    print(f"{len(jeu)} images d'entraînement dont {int(etiquetes.sum())} étiquetées pour le glaucome", flush=True)

    avis = np.full(len(jeu), np.nan, dtype=np.float32)
    externe_entier, externe_gros_plan = [], []
    for k in (0, 1):
        apprentissage = jeu[(moitie != k) & etiquetes]
        modele = entrainer(apprentissage, args.poids, args.epoques, progression, f"professeur {k + 1}/2")
        avis[moitie == k] = predire(modele, jeu[moitie == k])
        externe_entier.append(predire(modele, externe))
        externe_gros_plan.append(predire(modele, externe, gros_plan=GROS_PLAN_TEST))
        auroc = _auroc(jeu["glaucome"].to_numpy()[moitie == k], avis[moitie == k])
        progression.write(f"professeur {k + 1}/2 : AUROC glaucome sur l'autre moitié {auroc}")
        del modele
        torch.cuda.empty_cache()
    progression.close()

    # avis calibrés (Platt, sur les avis hors moitié) : la pondération des cas positifs pendant
    # l'entraînement pousse les probabilités vers le haut ; l'élève doit apprendre de vraies probabilités
    a, b = calibrer(avis, jeu["glaucome"].to_numpy())
    avis = a * avis + b
    externe_entier = [a * z + b for z in externe_entier]
    externe_gros_plan = [a * z + b for z in externe_gros_plan]
    y_ext = externe["glaucome"].to_numpy()
    resultat = {
        "architecture": ARCHITECTURE,
        "taille": TAILLE,
        "auroc_hors_moitie": _auroc(jeu["glaucome"].to_numpy(), avis),
        "auroc_par_source": {
            s: _auroc(jeu.loc[jeu["source"] == s, "glaucome"].to_numpy(), avis[(jeu["source"] == s).to_numpy()])
            for s in jeu["source"].unique()
        },
        "auroc_externe_jsiec": _auroc(y_ext, np.mean(externe_entier, axis=0)),
        "auroc_externe_jsiec_gros_plan": _auroc(y_ext, np.mean(externe_gros_plan, axis=0)),
        "calibrage": [a, b],
        "duree_s": round(time.perf_counter() - debut, 1),
    }
    (args.sortie / "resultats_professeur.json").write_text(json.dumps(resultat, indent=1, ensure_ascii=False))
    pd.DataFrame({"cle": jeu["cle"], "logit_professeur": avis}).to_csv(args.sortie / "professeur.csv", index=False)
    externe.assign(
        logit_professeur=np.mean(externe_entier, axis=0), logit_professeur_gros_plan=np.mean(externe_gros_plan, axis=0)
    )[["cle", "glaucome", "logit_professeur", "logit_professeur_gros_plan"]].to_csv(
        args.sortie / "professeur_externe.csv", index=False
    )
    print(json.dumps(resultat, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
