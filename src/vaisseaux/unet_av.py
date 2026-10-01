"""U-Net multi-appareils : vaisseaux et artères/veines (dépendance optionnelle : PyTorch).

Le premier U-Net (unet.py) n'a vu que les 18 images d'entraînement de DRIVE : sur les photos
d'autres appareils, il dessine les vaisseaux plus larges, ce qui fausse les mesures et empêche
toute comparaison à des yeux sains. Celui-ci est entraîné sur cinq bases (DRIVE_AV, HRF-AV,
LES-AV, FIVES, CHASE_DB1 ; voir scripts/preparer_vaisseaux_multi.py) avec des augmentations
d'appareil (couleur, gamma, éclairage, flou, compression JPEG, échelle), et il a deux sorties :

- « logits » : vaisseau ou non, pour chaque pixel (toutes les bases) ;
- « arteres » : artère (1) ou veine (0), pour les pixels de vaisseau (bases annotées en artères
  et veines seulement ; les autres pixels sont ignorés).

Plus profond que le premier (5 niveaux, champ récepteur plus large) : distinguer une artère d'une
veine demande de voir le vaisseau sur une bonne longueur. L'entrée doit être un multiple de 32.

    python -m vaisseaux.unet_av --donnees <dossier de preparer_vaisseaux_multi> --sortie modeles/unet_av.pt
    python -m vaisseaux.unet_av --exporter modeles/unet_av.pt      # ONNX pour le navigateur
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.nn import functional as F

from .pretraitement import BORNE_ENTREE, ECART_MIN, normaliser_pour_reseau
from .unet import _bloc

MULTIPLE = 32
FOND, INCONNU, ARTERE, VEINE, CROISEMENT = range(5)
# tirage des bases pendant l'entraînement : les bases annotées en artères/veines sont petites
# mais seules à enseigner la deuxième sortie ; FIVES apporte le volume
POIDS_SOURCES = {"FIVES": 0.35, "CHASE_DB1": 0.1, "DRIVE_AV": 0.15, "HRF-AV": 0.2, "LES-AV": 0.2}
TAILLE_PATCH = 384
ECHELLES = (0.7, 1.35)


class UNetAV(nn.Module):
    def __init__(self, base: int = 16, profondeur: int = 5, plafond: int = 256):
        super().__init__()
        c = [min(base * 2**k, plafond) for k in range(profondeur + 1)]
        self.descente = nn.ModuleList([_bloc(3 if k == 0 else c[k - 1], c[k]) for k in range(profondeur)])
        self.fond = _bloc(c[profondeur - 1], c[profondeur])
        self.montee = nn.ModuleList(
            [nn.ConvTranspose2d(c[k + 1], c[k], 2, stride=2) for k in reversed(range(profondeur))]
        )
        self.fusion = nn.ModuleList([_bloc(2 * c[k], c[k]) for k in reversed(range(profondeur))])
        self.tete = nn.Conv2d(c[0], 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, 2, H, W) : logit « vaisseau », logit « artère plutôt que veine »."""
        sauts = []
        for bloc in self.descente:
            x = bloc(x)
            sauts.append(x)
            x = F.max_pool2d(x, 2)
        x = self.fond(x)
        for monte, fusion, saut in zip(self.montee, self.fusion, reversed(sauts), strict=True):
            x = fusion(torch.cat([monte(x), saut], dim=1))
        return self.tete(x)


class _DeuxSorties(nn.Module):
    """Pour l'export : deux sorties nommées, de même forme que celle du premier U-Net."""

    def __init__(self, modele: UNetAV):
        super().__init__()
        self.modele = modele

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.modele(x)
        return y[:, 0], y[:, 1]


# ---------------------------------------------------------------- données


def lire_index(donnees: Path, images: Path | None = None) -> list[dict]:
    import pandas as pd

    images = images or donnees / "images"
    table = pd.read_csv(donnees / "index.csv")
    lignes = table.to_dict("records")
    for li in lignes:
        li["base"] = images / li["chemin"].removeprefix("images/")
    return lignes


def charger_exemple(ligne: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = str(ligne["base"])
    rgb = np.asarray(Image.open(base + ".png").convert("RGB"))
    etiquette = np.asarray(Image.open(base + "_etiquette.png"))
    masque = np.asarray(Image.open(base + "_masque.png")) > 127
    return rgb, etiquette, masque


def _couleur(rgb: np.ndarray, p: dict, origine: tuple[float, float] = (0, 0), pas: float = 1.0) -> np.ndarray:
    """Variations d'appareil : gain et gamma par canal, contraste, éclairage inégal (dégradé).
    `origine` et `pas` situent les pixels dans l'image entière (patch recadré ou image réduite),
    pour que le dégradé et donc les statistiques de normalisation restent cohérents."""
    x = rgb.astype(np.float32) / 255.0
    x = np.clip(x * p["gain"], 0, 1) ** p["gamma"]
    x = (x - 0.35) * p["contraste"] + 0.35
    if p["degrade"] is not None:
        a, angle, h, w = p["degrade"]
        lignes, colonnes = np.mgrid[0 : x.shape[0], 0 : x.shape[1]].astype(np.float32)
        lignes, colonnes = origine[0] + pas * lignes, origine[1] + pas * colonnes
        t = (np.cos(angle) * colonnes / w + np.sin(angle) * lignes / h) - 0.5
        x = x * (1 + a * t)[..., None]
    return np.clip(x, 0, 1)


def _tirer_couleur(rng: np.random.Generator, forme: tuple[int, int]) -> dict:
    return {
        "gain": rng.uniform(0.75, 1.25, 3).astype(np.float32),
        "gamma": (rng.uniform(0.7, 1.4) * rng.uniform(0.9, 1.1, 3)).astype(np.float32),
        "contraste": float(rng.uniform(0.7, 1.3)),
        "degrade": (float(rng.uniform(-0.5, 0.5)), float(rng.uniform(0, 2 * np.pi)), *forme)
        if rng.random() < 0.5
        else None,
    }


def exemple_augmente(
    rgb: np.ndarray, etiquette: np.ndarray, masque: np.ndarray, rng: np.random.Generator, taille: int = TAILLE_PATCH
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patch augmenté : (3, T, T) normalisé comme dans la démo, étiquettes (T, T), masque (T, T)."""
    echelle = math.exp(rng.uniform(*np.log(ECHELLES)))
    cote = min(int(round(taille / echelle)), *rgb.shape[:2])
    points = np.argwhere(masque)
    li, co = points[rng.integers(len(points))]
    h0 = int(np.clip(li - cote // 2, 0, rgb.shape[0] - cote))
    g0 = int(np.clip(co - cote // 2, 0, rgb.shape[1] - cote))
    fenetre = (slice(h0, h0 + cote), slice(g0, g0 + cote))
    couleur = _tirer_couleur(rng, rgb.shape[:2])
    # statistiques de normalisation sur toute l'image (comme la démo), estimées sur une image réduite
    reduite = _couleur(rgb[::4, ::4], couleur, pas=4)
    m = masque[::4, ::4]
    moyenne, ecart = reduite[m].mean(axis=0), np.maximum(reduite[m].std(axis=0), ECART_MIN)

    image = Image.fromarray(rgb[fenetre]).resize((taille, taille), Image.Resampling.BILINEAR)
    lab = np.asarray(Image.fromarray(etiquette[fenetre]).resize((taille, taille), Image.Resampling.NEAREST))
    msk = np.asarray(Image.fromarray(masque[fenetre]).resize((taille, taille), Image.Resampling.NEAREST))
    x = _couleur(np.asarray(image), couleur, (h0, g0), cote / taille)
    if rng.random() < 0.3:  # compression JPEG
        tampon = io.BytesIO()
        Image.fromarray((x * 255).astype(np.uint8)).save(tampon, format="JPEG", quality=int(rng.integers(30, 85)))
        x = np.asarray(Image.open(tampon)).astype(np.float32) / 255.0
    if rng.random() < 0.25:  # mise au point imparfaite
        flou = Image.fromarray((x * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.6)))
        x = np.asarray(flou).astype(np.float32) / 255.0
    if rng.random() < 0.3:  # bruit du capteur
        x = x + rng.normal(0, rng.uniform(0.005, 0.03), x.shape).astype(np.float32)
    x = np.clip((x - moyenne) / ecart, -BORNE_ENTREE, BORNE_ENTREE) * msk[..., None]
    x = x.astype(np.float32).transpose(2, 0, 1)
    k = int(rng.integers(4))
    x, lab, msk = np.rot90(x, k, axes=(1, 2)), np.rot90(lab, k), np.rot90(msk, k)
    if rng.random() < 0.5:
        x, lab, msk = x[:, :, ::-1], lab[:, ::-1], msk[:, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(lab), np.ascontiguousarray(msk)


class FluxPatchs(torch.utils.data.IterableDataset):
    """Patchs tirés à l'infini : base selon POIDS_SOURCES, puis image au hasard dans la base."""

    def __init__(self, exemples: dict[str, list], graine: int = 0):
        self.exemples, self.graine = exemples, graine

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.graine, info.id if info else 0])
        sources = [s for s in POIDS_SOURCES if self.exemples.get(s)]
        poids = np.array([POIDS_SOURCES[s] for s in sources])
        poids = poids / poids.sum()
        while True:
            liste = self.exemples[sources[rng.choice(len(sources), p=poids)]]
            yield exemple_augmente(*liste[rng.integers(len(liste))], rng)


# ---------------------------------------------------------------- prédiction et évaluation


def entree(rgb: np.ndarray, masque: np.ndarray) -> np.ndarray:
    return normaliser_pour_reseau(rgb, masque, MULTIPLE)


@torch.no_grad()
def predire(modele: UNetAV, rgb: np.ndarray, masque: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probabilités « vaisseau » et « artère » de chaque pixel, (H, W) chacune."""
    appareil = next(modele.parameters()).device
    modele.eval()
    h, w = masque.shape
    y = torch.sigmoid(modele(torch.from_numpy(entree(rgb, masque))[None].to(appareil)).float())[0, :, :h, :w]
    y = y.cpu().numpy()
    return y[0] * masque, y[1]


def scores_image(vaisseau: np.ndarray, artere: np.ndarray, etiquette: np.ndarray, masque: np.ndarray) -> dict:
    """Dice des vaisseaux (seuil 0,5) et, si l'image est annotée en artères/veines, part des pixels
    de vaisseau (annotés et détectés) bien classés."""
    pred, verite = (vaisseau >= 0.5) & masque, (etiquette > 0) & masque
    dice = 2 * (pred & verite).sum() / max(1, pred.sum() + verite.sum())
    sortie = {"dice": float(dice)}
    connus = pred & np.isin(etiquette, (ARTERE, VEINE))
    if connus.sum() > 0:
        sortie["av_exactitude"] = float(((artere[connus] >= 0.5) == (etiquette[connus] == ARTERE)).mean())
        # gros vaisseaux seulement (ceux qui servent au CRAE/CRVE) : pixels à plus de 2 px du bord
        from scipy import ndimage as ndi

        gros = connus & (ndi.distance_transform_edt(verite) > 2)
        if gros.sum() > 0:
            sortie["av_exactitude_gros"] = float(((artere[gros] >= 0.5) == (etiquette[gros] == ARTERE)).mean())
    return sortie


def evaluer(modele, lignes: list[dict], premier_unet=None) -> dict:
    """Moyennes par base ; `premier_unet` (unet.UNet) ajoute le Dice du premier modèle."""
    from .unet import predire as predire_premier

    par_source: dict[str, list] = {}
    for li in lignes:
        rgb, etiquette, masque = charger_exemple(li)
        vaisseau, artere = predire(modele, rgb, masque)
        s = scores_image(vaisseau, artere, etiquette, masque)
        if premier_unet is not None:
            p = predire_premier(premier_unet, rgb, masque)
            s["dice_premier_unet"] = scores_image(p, p, etiquette, masque)["dice"]
        par_source.setdefault(li["source"], []).append(s)
    sortie = {}
    for source, v in par_source.items():
        cles = sorted({k for s in v for k in s})
        sortie[source] = {"images": len(v)} | {k: float(np.mean([s[k] for s in v if k in s])) for k in cles}
    return sortie


def _score_validation(resultats: dict) -> float:
    """Choix de l'itération : moyenne du Dice (moyenne des bases) et de l'exactitude artères/veines."""
    dice = np.mean([r["dice"] for r in resultats.values()])
    av = np.mean([r["av_exactitude"] for r in resultats.values() if "av_exactitude" in r])
    return float(0.5 * dice + 0.5 * av)


# ---------------------------------------------------------------- entraînement


def entrainer(
    donnees: Path,
    images: Path | None = None,
    iterations: int = 16000,
    lot: int = 12,
    graine: int = 0,
    travailleurs: int = 4,
    premier_unet: Path | None = None,
    sauvegarde: Path | None = None,
) -> tuple[UNetAV, dict]:
    """Entraîne et garde l'itération au meilleur score de validation, enregistrée dans `sauvegarde`
    à chaque progrès (un arrêt brutal ne perd donc pas le meilleur modèle). Si les calculs
    divergent (valeurs non finies), on repart du meilleur état, trois fois au plus, puis on
    s'arrête en gardant le meilleur modèle."""
    torch.manual_seed(graine)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    lignes = lire_index(donnees, images)
    exemples: dict[str, list] = {}
    for li in lignes:
        if li["partie"] == "entrainement":
            exemples.setdefault(li["source"], []).append(charger_exemple(li))
    validation = [li for li in lignes if li["partie"] == "validation"]
    print({s: len(v) for s, v in exemples.items()}, "| validation :", len(validation), flush=True)
    flux = torch.utils.data.DataLoader(
        FluxPatchs(exemples, graine), batch_size=lot, num_workers=travailleurs, persistent_workers=travailleurs > 0
    )
    modele = UNetAV().to(appareil).to(memory_format=torch.channels_last)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=1e-3, weight_decay=1e-4)
    planning = torch.optim.lr_scheduler.OneCycleLR(optimiseur, max_lr=1e-3, total_steps=iterations, pct_start=0.1)
    amp = appareil == "cuda"
    echelle = torch.amp.GradScaler(enabled=amp)
    meilleur, etat_meilleur, historique = -1.0, None, []
    debut = time.perf_counter()
    pertes_invalides, reprises, arret = 0, [], None

    def reprendre(it: int, raison: str) -> bool:
        """Revient au meilleur état ; renvoie False s'il faut arrêter."""
        reprises.append({"iteration": it, "raison": raison})
        print(f"itération {it} : {raison} -> retour au meilleur état ({len(reprises)}/3)", flush=True)
        if etat_meilleur is None or len(reprises) > 3:
            return False
        modele.load_state_dict(etat_meilleur)
        optimiseur.state.clear()
        return True

    for it, (x, lab, msk) in enumerate(flux, start=1):
        # garde-fou : une seule valeur hors de la plage du float16 abîme définitivement les
        # statistiques de normalisation du réseau (entraînement du 1er octobre 2026 perdu ainsi)
        x = torch.nan_to_num(x, 0.0).clamp(-BORNE_ENTREE, BORNE_ENTREE)
        x = x.to(appareil, non_blocking=True).to(memory_format=torch.channels_last)
        lab, msk = lab.to(appareil).long(), msk.to(appareil).float()
        vaisseau = (lab > 0).float()
        av_connu = ((lab == ARTERE) | (lab == VEINE)).float()
        with torch.autocast(appareil, dtype=torch.float16, enabled=amp):
            y = modele(x)
        y = y.float()
        bce = (F.binary_cross_entropy_with_logits(y[:, 0], vaisseau, reduction="none") * msk).sum() / msk.sum()
        p = torch.sigmoid(y[:, 0]) * msk
        dice = 1 - (2 * (p * vaisseau).sum() + 1) / (p.sum() + vaisseau.sum() + 1)
        perte = bce + dice
        if av_connu.sum() > 0:
            # classes équilibrées : artères et veines pèsent autant, quel que soit leur nombre de pixels
            cible = (lab == ARTERE).float()
            par_pixel = F.binary_cross_entropy_with_logits(y[:, 1], cible, reduction="none") * av_connu
            n_a, n_v = (cible * av_connu).sum(), ((1 - cible) * av_connu).sum()
            perte_av = 0.5 * (par_pixel * cible).sum() / n_a.clamp(min=1) + 0.5 * (
                par_pixel * (1 - cible)
            ).sum() / n_v.clamp(min=1)
            perte = perte + perte_av
        optimiseur.zero_grad(set_to_none=True)
        if torch.isfinite(perte):
            pertes_invalides = 0
            echelle.scale(perte).backward()
            echelle.unscale_(optimiseur)
            nn.utils.clip_grad_norm_(modele.parameters(), 1.0)  # évite les pas démesurés qui font diverger
            echelle.step(optimiseur)
            echelle.update()
        else:
            pertes_invalides += 1
            if pertes_invalides >= 5:
                pertes_invalides = 0
                if not reprendre(it, "perte non finie 5 fois de suite"):
                    arret = it
                    break
        planning.step()
        if it in (100, 500) or it % 1000 == 0 or it == iterations:
            abimes = [
                k for k, v in modele.state_dict().items() if v.is_floating_point() and not torch.isfinite(v).all()
            ]
            if abimes and not reprendre(it, f"poids ou statistiques non finis ({abimes[0]})"):
                arret = it
                break
            res = evaluer(modele, validation)
            score = _score_validation(res)
            historique.append({"iteration": it, "perte": float(perte), "score": score, "validation": res})
            ecoule = time.perf_counter() - debut
            print(
                f"itération {it:6d}  perte {float(perte):.4f}  score {score:.4f}  "
                + "  ".join(
                    f"{s} Dice {r['dice']:.3f}" + (f" A/V {r['av_exactitude']:.3f}" if "av_exactitude" in r else "")
                    for s, r in res.items()
                )
                + f"  ({ecoule / 60:.0f} min, reste ≈ {ecoule / it * (iterations - it) / 60:.0f} min)",
                flush=True,
            )
            if np.isfinite(score) and score > meilleur:
                meilleur = score
                etat_meilleur = {k: v.detach().clone() for k, v in modele.state_dict().items()}
                if sauvegarde is not None:
                    sauvegarde.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(etat_meilleur, sauvegarde)
        if it >= iterations:
            break
    if etat_meilleur is None:
        raise RuntimeError("aucune itération valide : pas de modèle à garder")
    modele.load_state_dict(etat_meilleur)
    infos = {
        "score_validation": meilleur,
        "iteration_retenue": max((h for h in historique if np.isfinite(h["score"])), key=lambda h: h["score"])[
            "iteration"
        ],
        "arret_anticipe": arret,
        "reprises": reprises,
        "iterations": iterations,
        "lot": lot,
        "taille_patch": TAILLE_PATCH,
        "poids_sources": POIDS_SOURCES,
        "duree_min": round((time.perf_counter() - debut) / 60, 1),
        "appareil": torch.cuda.get_device_name(0) if appareil == "cuda" else "cpu",
        "historique": historique,
    }
    premier = None
    if premier_unet is not None:
        from .unet import charger_modele

        premier = charger_modele(premier_unet, appareil)
    infos["test"] = evaluer(modele, [li for li in lignes if li["partie"] == "test"], premier)
    return modele, infos


def charger_modele(chemin: Path, appareil: str | None = None) -> UNetAV:
    appareil = appareil or ("cuda" if torch.cuda.is_available() else "cpu")
    modele = UNetAV()
    modele.load_state_dict(torch.load(chemin, map_location=appareil, weights_only=True))
    return modele.to(appareil).eval()


def exporter_onnx(poids: Path, sortie: Path) -> float:
    """Exporte en ONNX (hauteur et largeur libres, multiples de 32), sorties « logits » et
    « arteres ». Renvoie l'écart maximal entre PyTorch et ONNX Runtime (test de parité)."""
    import onnxruntime as ort

    modele = _DeuxSorties(charger_modele(poids, "cpu")).eval()
    h, w = torch.export.Dim("h32", min=2, max=64), torch.export.Dim("w32", min=2, max=64)
    torch.onnx.export(
        modele,
        (torch.zeros(1, 3, 576, 576),),
        str(sortie),
        input_names=["image"],
        output_names=["logits", "arteres"],
        dynamic_shapes=({2: MULTIPLE * h, 3: MULTIPLE * w},),
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    x = torch.randn(1, 3, 544, 800)
    with torch.no_grad():
        attendu = [t.numpy() for t in modele(x)]
    obtenu = ort.InferenceSession(str(sortie), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})
    return float(max(np.abs(a - o).max() for a, o in zip(attendu, obtenu, strict=True)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entraîne le U-Net multi-appareils (vaisseaux + artères/veines)")
    parser.add_argument("--donnees", type=Path, help="dossier de preparer_vaisseaux_multi (index.csv)")
    parser.add_argument("--images", type=Path, help="dossier des images, si ailleurs que <donnees>/images")
    parser.add_argument("--sortie", type=Path, default=Path("modeles/unet_av.pt"))
    parser.add_argument("--resultats", type=Path, default=Path("resultats/unet_av.json"))
    parser.add_argument("--premier-unet", type=Path, help="poids du premier U-Net, pour le comparer sur le test")
    parser.add_argument("--iterations", type=int, default=16000)
    parser.add_argument("--travailleurs", type=int, default=4)
    parser.add_argument("--exporter", type=Path, help="exporte ces poids en ONNX (même nom, extension .onnx)")
    args = parser.parse_args(argv)
    if args.exporter:
        onnx = args.exporter.with_suffix(".onnx")
        ecart = exporter_onnx(args.exporter, onnx)
        print(f"ONNX écrit dans {onnx}, écart maximal avec PyTorch : {ecart:.2e}")
        return 0 if ecart < 1e-3 else 1
    modele, infos = entrainer(
        args.donnees,
        args.images,
        args.iterations,
        travailleurs=args.travailleurs,
        premier_unet=args.premier_unet,
        sauvegarde=args.sortie,
    )
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    args.resultats.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modele.state_dict(), args.sortie)
    args.resultats.write_text(json.dumps(infos, indent=1, ensure_ascii=False))
    print(json.dumps(infos["test"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
