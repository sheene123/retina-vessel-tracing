"""Petit U-Net de segmentation des vaisseaux (dépendance optionnelle : PyTorch).

    python -m vaisseaux.unet --racine data/DRIVE --sortie modeles/unet_drive.pt

Entraîné sur 18 images d'entraînement DRIVE, 2 gardées pour choisir la meilleure itération.
Les images de test ne sont jamais vues.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .donnees import charger, lister
from .evaluation import metriques_binaires
from .pretraitement import normaliser_pour_reseau


def _bloc(entree: int, sortie: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(entree, sortie, 3, padding=1),
        nn.BatchNorm2d(sortie),
        nn.ReLU(inplace=True),
        nn.Conv2d(sortie, sortie, 3, padding=1),
        nn.BatchNorm2d(sortie),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        c = [base, 2 * base, 4 * base, 8 * base]
        self.descente = nn.ModuleList([_bloc(3, c[0]), _bloc(c[0], c[1]), _bloc(c[1], c[2])])
        self.fond = _bloc(c[2], c[3])
        self.montee = nn.ModuleList([nn.ConvTranspose2d(c[k + 1], c[k], 2, stride=2) for k in (2, 1, 0)])
        self.fusion = nn.ModuleList([_bloc(2 * c[k], c[k]) for k in (2, 1, 0)])
        self.tete = nn.Conv2d(c[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sauts = []
        for bloc in self.descente:
            x = bloc(x)
            sauts.append(x)
            x = F.max_pool2d(x, 2)
        x = self.fond(x)
        for monte, fusion, saut in zip(self.montee, self.fusion, reversed(sauts), strict=True):
            x = fusion(torch.cat([monte(x), saut], dim=1))
        return self.tete(x)[:, 0]


def entree(rgb: np.ndarray, masque: np.ndarray) -> np.ndarray:
    """Image normalisée canal par canal dans le champ de vue, (3, H, W), sans complément."""
    h, w = masque.shape
    return normaliser_pour_reseau(rgb, masque)[:, :h, :w]


def _patchs(donnees, n: int, taille: int, rng: np.random.Generator):
    xs, ys = [], []
    for _ in range(n):
        x, y, m = donnees[rng.integers(len(donnees))]
        points = np.argwhere(m[taille // 2 : -taille // 2, taille // 2 : -taille // 2])
        li, co = points[rng.integers(len(points))]
        px, py = x[:, li : li + taille, co : co + taille], y[li : li + taille, co : co + taille]
        k = int(rng.integers(4))
        px, py = np.rot90(px, k, axes=(1, 2)), np.rot90(py, k)
        if rng.random() < 0.5:
            px, py = px[:, :, ::-1], py[:, ::-1]
        xs.append(px.copy())
        ys.append(py.copy())
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys).astype(np.float32))


@torch.no_grad()
def predire(modele: UNet, rgb: np.ndarray, masque: np.ndarray, appareil: str | None = None) -> np.ndarray:
    """Probabilité « vaisseau » de chaque pixel, nulle hors du champ de vue."""
    appareil = appareil or next(modele.parameters()).device
    modele.eval()
    x = entree(rgb, masque)
    h, w = masque.shape
    ph, pw = -h % 16, -w % 16
    x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
    proba = torch.sigmoid(modele(torch.from_numpy(x)[None].to(appareil)))[0, :h, :w].cpu().numpy()
    return proba * masque


def entrainer(
    racine: Path, iterations: int = 3000, lot: int = 24, taille: int = 96, graine: int = 0
) -> tuple[UNet, dict]:
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    images = [charger(racine, "entrainement", i) for i in lister(racine, "entrainement")]
    donnees = [(entree(im.rgb, im.masque), im.verite.astype(np.float32), im.masque) for im in images]
    apprentissage, validation = donnees[:-2], images[-2:]
    modele = UNet().to(appareil)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=2e-3, weight_decay=1e-4)
    planning = torch.optim.lr_scheduler.OneCycleLR(optimiseur, max_lr=2e-3, total_steps=iterations)
    meilleur, etat_meilleur, historique = -1.0, None, []
    debut = time.perf_counter()
    for it in range(1, iterations + 1):
        modele.train()
        x, y = _patchs(apprentissage, lot, taille, rng)
        x, y = x.to(appareil), y.to(appareil)
        logits = modele(x)
        p = torch.sigmoid(logits)
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
        perte = F.binary_cross_entropy_with_logits(logits, y) + dice
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        planning.step()
        if it % 250 == 0 or it == iterations:
            scores = [
                metriques_binaires(predire(modele, im.rgb, im.masque) >= 0.5, im.verite, im.masque)["dice"]
                for im in validation
            ]
            historique.append({"iteration": it, "perte": float(perte), "dice_validation": float(np.mean(scores))})
            print(f"itération {it:5d}  perte {float(perte):.4f}  Dice validation {np.mean(scores):.4f}")
            if np.mean(scores) > meilleur:
                meilleur = float(np.mean(scores))
                etat_meilleur = {k: v.detach().clone() for k, v in modele.state_dict().items()}
    modele.load_state_dict(etat_meilleur)
    return modele, {
        "dice_validation": meilleur,
        "iterations": iterations,
        "duree_s": round(time.perf_counter() - debut, 1),
        "appareil": torch.cuda.get_device_name(0) if appareil == "cuda" else "cpu",
        "historique": historique,
    }


def charger_modele(chemin: Path, appareil: str | None = None) -> UNet:
    appareil = appareil or ("cuda" if torch.cuda.is_available() else "cpu")
    modele = UNet()
    modele.load_state_dict(torch.load(chemin, map_location=appareil, weights_only=True))
    return modele.to(appareil).eval()


def exporter_onnx(poids: Path, sortie: Path, racine: Path | None = None) -> float:
    """Exporte le U-Net en ONNX (hauteur et largeur libres, multiples de 8) pour le navigateur.
    Renvoie l'écart maximal entre PyTorch et ONNX Runtime sur une image DRIVE (test de parité)."""
    import onnxruntime as ort

    modele = charger_modele(poids, "cpu")
    h, w = torch.export.Dim("h8", min=4, max=512), torch.export.Dim("w8", min=4, max=512)
    torch.onnx.export(
        modele,
        (torch.zeros(1, 3, 592, 576),),
        str(sortie),
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes=({2: 8 * h, 3: 8 * w},),
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    racine = racine or Path("data/DRIVE")
    im = charger(racine, "test", lister(racine, "test")[0])
    x = entree(im.rgb, im.masque)
    x = np.pad(x, ((0, 0), (0, -x.shape[1] % 16), (0, -x.shape[2] % 16)))[None]
    with torch.no_grad():
        attendu = modele(torch.from_numpy(x)).numpy()
    obtenu = ort.InferenceSession(str(sortie), providers=["CPUExecutionProvider"]).run(None, {"image": x})[0]
    return float(np.abs(obtenu - attendu).max())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entraîne le U-Net de segmentation des vaisseaux sur DRIVE")
    parser.add_argument("--racine", type=Path, default=Path("data/DRIVE"))
    parser.add_argument("--sortie", type=Path, default=Path("modeles/unet_drive.pt"))
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--onnx", type=Path, help="exporte les poids existants (--sortie) en ONNX à ce chemin")
    args = parser.parse_args(argv)
    if args.onnx:
        ecart = exporter_onnx(args.sortie, args.onnx, args.racine)
        print(f"ONNX écrit dans {args.onnx}, écart maximal avec PyTorch : {ecart:.2e}")
        return 0 if ecart < 1e-3 else 1
    modele, infos = entrainer(args.racine, args.iterations)
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modele.state_dict(), args.sortie)
    print({k: v for k, v in infos.items() if k != "historique"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
