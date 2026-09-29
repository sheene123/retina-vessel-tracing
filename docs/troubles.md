# Détection indicative de six troubles de l'œil

## Objectif

Le modèle reconnaît des annotations de fond d'œil issues d'ODIR-5K. Il ne mesure pas la tension
artérielle, la puissance optique d'une myopie ou l'état cardiaque d'une personne. Une sortie
« probable » est un signal à interpréter médicalement, pas un diagnostic.

## Protocole

- 6 392 yeux provenant de 3 358 patients ;
- séparation des plis par patient, pour qu'un même patient ne soit jamais présent dans
  l'entraînement et l'évaluation ;
- modèle multi-sortie exporté en ONNX pour l'exécution dans le navigateur ;
- calibrage indépendant de chaque sortie, puis conversion en niveaux « peu probable », « possible »
  et « probable » ;
- AUROC et intervalles de confiance à 95 % calculés hors pli.

Les classes sont la rétinopathie diabétique, le glaucome, la cataracte, la DMLA, la rétinopathie
hypertensive et la myopie forte. Les libellés proviennent du diagnostic ODIR-5K et ne constituent
pas une vérité clinique exhaustive.

## Résultats

| Trouble | Yeux évalués | Cas positifs | AUROC | IC 95 % |
|---|---:|---:|---:|---:|
| Rétinopathie diabétique | 6 388 | 1 718 | 0,84 | [0,82 ; 0,85] |
| Glaucome | 6 351 | 272 | 0,90 | [0,87 ; 0,92] |
| Cataracte | 6 392 | 301 | 0,98 | [0,96 ; 0,99] |
| DMLA | 6 223 | 279 | 0,92 | [0,89 ; 0,94] |
| Rétinopathie hypertensive | 6 392 | 192 | 0,82 | [0,78 ; 0,86] |
| Myopie forte | 6 391 | 256 | 0,99 | [0,99 ; 1,00] |

La parité entre le modèle PyTorch et l'export ONNX est de 0,979 en corrélation sur les sorties.
Les résultats détaillés, les prédictions et la figure ROC sont dans
[resultats/troubles/](../resultats/troubles/).

## Dans la démo

L'image importée est recadrée automatiquement sur le fond d'œil avant analyse. Le résultat est
présenté avec une phrase synthétique, une barre de probabilité et un niveau lisible. Le pourcentage
est une probabilité calibrée sur ODIR-5K, pas une certitude pour l'image fournie. Les résultats
techniques sont repliés afin de garder l'interface compréhensible.

La démo exécute le modèle localement dans le navigateur. Les images importées ne sont pas envoyées
à un serveur par cette interface.