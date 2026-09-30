"""Choisit les exemples « yeux malades, diagnostic connu » de la démo, dans le test externe JSIEC-1000.

Les images viennent d'un hôpital jamais utilisé pour l'entraînement, et sont tirées au hasard dans
chaque catégorie (graine fixe) sans regarder la réponse du réseau : la démo montre donc aussi ses
erreurs. Images : Joint Shantou International Eye Centre (Cen et al., Nature Communications, 2021),
licence DbCL 1.0 ; réduites à 700 px et publiées avec cette attribution.

    python scripts/preparer_exemples_malades.py      # écrit demo/web/exemples/
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from vaisseaux.troubles import lire_index  # noqa: E402

# catégorie JSIEC -> (diagnostic affiché, trouble cherché par le réseau ou None pour un œil sain, précision)
CATEGORIES = {
    "0.0.Normal": ("Œil sain", None, ""),
    "1.0.DR2": ("Rétinopathie diabétique", "diabete", "stade modéré"),
    "1.1.DR3": ("Rétinopathie diabétique", "diabete", "stade sévère"),
    "10.0.Possible glaucoma": ("Glaucome", "glaucome", "glaucome possible"),
    "6.Maculopathy": ("Maculopathie", "dmla", "atteinte de la macula ; le réseau y cherche la DMLA, une de ses formes"),
    "11.Severe hypertensive retinopathy": ("Rétinopathie hypertensive", "hypertension", "forme sévère"),
    "9.Pathological myopia": ("Myopie forte", "myopie", "myopie pathologique"),
}
PAR_CATEGORIE = {"1.0.DR2": 1, "1.1.DR3": 1}  # deux exemples de diabète au total, un par stade
GRAINE = 2026
COTE = 700


def main() -> int:
    index = lire_index(RACINE / "data" / "troubles" / "index.csv")
    jsiec = index[index["source"] == "jsiec"]
    sortie = RACINE / "demo" / "web" / "exemples"
    sortie.mkdir(parents=True, exist_ok=True)
    hasard = random.Random(GRAINE)
    exemples = []
    for categorie, (diagnostic, trouble, precision) in CATEGORIES.items():
        origines = sorted(jsiec.loc[jsiec["categorie"] == categorie, "origine"])
        for origine in hasard.sample(origines, PAR_CATEGORIE.get(categorie, 2)):
            nom = f"jsiec_{len(exemples) + 1:02d}.jpg"
            image = Image.open(origine).convert("RGB")
            image.thumbnail((COTE, COTE))
            image.save(sortie / nom, quality=90)
            exemples.append(
                {
                    "fichier": nom,
                    "diagnostic": diagnostic,
                    "trouble": trouble,
                    "precision": precision,
                    "categorie": categorie.split(".", 2)[-1] if categorie[0].isdigit() else categorie,
                    "numero": sum(e["diagnostic"] == diagnostic for e in exemples) + 1,
                }
            )
    (sortie / "exemples.json").write_text(json.dumps(exemples, indent=1, ensure_ascii=False))
    print(f"{len(exemples)} exemples écrits dans {sortie}")
    for e in exemples:
        print(f"  {e['fichier']} : {e['diagnostic']} ({e['categorie']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
