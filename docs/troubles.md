# Détection indicative de six troubles de l'œil

## Objectif

Le modèle repère, sur une photo couleur du fond d'œil, des signes de six troubles : rétinopathie
diabétique, glaucome, cataracte, DMLA, rétinopathie hypertensive et myopie forte. Il reconnaît des
annotations de jeux de données publics : il ne mesure ni la tension artérielle, ni la correction
d'une myopie, ni l'état cardiaque d'une personne. Une sortie « probable » est un signal à faire
interpréter par un ophtalmologiste, pas un diagnostic.

## Données (version 0.2.0)

La première version n'était entraînée que sur ODIR-5K : elle se trompait souvent sur des images
venues d'ailleurs (manuels, sites web, autres appareils). La version 0.2.0 réunit quatre bases
publiques ([sources_troubles.py](../src/vaisseaux/sources_troubles.py)) :

| Base | Images | Troubles étiquetés |
|---|---:|---|
| ODIR-5K (Chine, plusieurs appareils) | 6 392 | les six |
| RFMiD (Inde, 3 appareils) | 3 200 | rétinopathie diabétique, glaucome (excavation papillaire), DMLA, myopie, rétinopathie hypertensive |
| SMDG-19 (15 jeux de glaucome) | 4 224 | glaucome |
| sjchoi86 | 601 | cataracte, glaucome |

- **Étiquettes partielles** : chaque base n'annote que certains troubles ; les autres sont ignorés
  par la perte au lieu d'être comptés comme absents. Les diagnostics incertains (« suspect »,
  fond tigré seul, trouble des milieux pour la cataracte) sont ignorés de la même façon.
- **Même préparation que la démo** : chaque image est recadrée sur l'œil et mise au carré par la
  fonction utilisée dans le navigateur (`preparer_fond_oeil`).
- **Doublons** : SMDG contient lui-même ODIR et JSIEC ; ces sous-jeux sont retirés, ainsi que ceux
  qui n'ont qu'une classe (un jeu 100 % glaucome apprendrait la caméra plutôt que la maladie).
  Les quasi-doublons restants sont proposés par une empreinte perceptuelle, puis confirmés par la
  corrélation du dessin fin des vaisseaux (deux yeux différents ne dépassent pas 0,45, un doublon
  dépasse 0,5) : 299 doublons sont regroupés, aucun n'est partagé avec le test externe.
- **Augmentations « monde réel »** : zoom jusqu'à un gros plan de la papille, compression JPEG,
  flou, basse résolution, dominante de couleur, traits et lettres comme sur une figure annotée.

## Protocole

- EfficientNet-B0 pré-entraîné sur ImageNet, 384 × 384, 6 sorties, 12 époques ; entraîné sur un
  GPU T4 (Kaggle, environ 3 h 15).
- Validation croisée en 5 plis groupée par patient et par doublon, stratifiée par base et par
  trouble ; AUROC avec IC 95 % par bootstrap sur les groupes.
- Calibrage de Platt de chaque sortie sur les prédictions hors pli : un « 70 % » correspond à
  environ 7 yeux atteints sur 10 dans une population comme celle des bases d'entraînement.
- **Test externe** : JSIEC-1000 (Joint Shantou International Eye Centre), 1 000 photos d'un
  hôpital jamais utilisé pour l'entraînement ni pour les réglages.
- Export ONNX vérifié sur de vraies images (écart maximal 9·10⁻⁵ avec PyTorch).

## Résultats

| Trouble | AUROC validation croisée | AUROC test externe JSIEC | Version 0.1.1 sur JSIEC |
|---|---|---|---|
| Rétinopathie diabétique | 0,90 [0,89 ; 0,91] | **0,94** [0,91 ; 0,97] (106 cas) | 0,92 |
| Glaucome | 0,90 [0,89 ; 0,90] | **1,00** [0,99 ; 1,00] (13 cas) | 0,94 |
| Cataracte | 0,99 [0,98 ; 0,99] | pas de cas dans JSIEC | — |
| DMLA | 0,94 [0,93 ; 0,95] | **0,95** [0,93 ; 0,97] (74 cas) | 0,92 |
| Rétinopathie hypertensive | 0,86 [0,83 ; 0,89] | **0,98** [0,96 ; 0,99] (15 cas) | 0,90 |
| Myopie forte | 1,00 [1,00 ; 1,00] | **1,00** [1,00 ; 1,00] (54 cas) | 0,98 |

Sur les seules images ODIR, le nouveau modèle fait jeu égal avec l'ancien (par exemple 0,83 contre
0,84 pour la rétinopathie diabétique) : le gain porte sur la généralisation à d'autres hôpitaux et
appareils, ce qui était le but.

### Pourquoi certains chiffres externes sont proches de 1

Un AUROC de 1,00 sur JSIEC ne veut pas dire une détection parfaite :

- ce sont des arrondis (0,997 pour le glaucome, 0,999 pour la myopie) ;
- l'AUROC mesure un classement (chaque œil atteint a-t-il un score plus haut que chaque œil sain ?), pas
  l'absence d'erreurs : au seuil « possible », 17 % des yeux sans glaucome sont signalés à tort en validation
  croisée, et 16 % dans JSIEC ;
- JSIEC compte peu de cas pour certains troubles (13 glaucomes, 15 rétinopathies hypertensives) : avec 13 cas
  tous détectés, la sensibilité réelle pourrait descendre vers 75 %, et l'intervalle de confiance par bootstrap
  est trop étroit ;
- ses photos montrent surtout des cas typiques, souvent avancés (« rétinopathie hypertensive sévère »,
  « myopie pathologique »), une seule maladie nette par image : les formes débutantes, les cas limites et les
  photos de mauvaise qualité, les plus difficiles, y sont rares.

La validation croisée (des milliers d'yeux de gravité et de qualité variables) est la mesure la plus
représentative. La démo et la fiche du registre affichent donc la qualité de détection d'après elle, avec la part
de cas repérés et de fausses alertes, et donnent le résultat externe avec son nombre de cas.

### Exemples à diagnostic connu dans la démo

La démo propose 12 photos de JSIEC (2 yeux sains, 2 par trouble sauf la cataracte, absente de JSIEC), tirées au
hasard avec une graine fixe sans regarder la réponse du réseau ([preparer_exemples_malades.py](../scripts/preparer_exemples_malades.py)).
Après l'analyse, le diagnostic posé par l'hôpital s'affiche à côté de la réponse du réseau, erreurs comprises.
Images : Joint Shantou International Eye Centre (Cen et al., *Nature Communications*, 2021), licence DbCL 1.0.

À lire avec prudence :

- dans JSIEC, la DMLA est approchée par la catégorie « maculopathie », et la rétinopathie
  hypertensive n'y est présente que sous sa forme sévère ; le glaucome ne compte que 13 cas,
  d'où un intervalle large ;
- dans RFMiD, le glaucome est approché par l'excavation de la papille ;
- les gros plans très serrés sur la papille restent difficiles pour le glaucome.

## Version 0.3.0 : professeur RETFound pour le glaucome

Le point faible de la version 0.2.0 : sur un gros plan serré de la papille (photo prise de près, papille
qui remplit l'image), le glaucome n'était plus reconnu (AUROC 0,872 sur JSIEC, 54 % des cas signalés).

- **Professeur** ([professeur.py](../src/vaisseaux/professeur.py)) : RETFound (ViT-L DINOv2, pré-entraîné
  sur ~1,6 million de fonds d'œil ; poids vérifiés par empreinte, chargés en mode « poids seulement »
  puis convertis en safetensors) affiné sur le glaucome en deux moitiés croisées, avec des gros plans de
  la papille. Seul, il atteint 0,910 en interne et 1,000 sur les gros plans JSIEC.
- **Élève** : le réseau de la démo apprend aussi l'avis calibré du professeur (distillation) et voit des
  gros plans de la papille pendant l'entraînement.

| Trouble | AUROC validation croisée | Fausses alertes au seuil « possible » | AUROC JSIEC (v0.3.0) | AUROC JSIEC (v0.2.0) |
|---|---|---|---|---|
| Rétinopathie diabétique | 0,89 | 22 % | 0,947 (106 cas) | 0,942 |
| Glaucome | 0,91 | 16 % | 0,998 (13 cas) | 0,997 |
| Cataracte | 0,98 | 2 % | pas de cas | — |
| DMLA | 0,94 | 6 % | 0,966 (74 cas) | 0,953 |
| Rétinopathie hypertensive | 0,86 | 15 % | 0,945 (15 cas) | 0,979 |
| Myopie forte | 1,00 | 1 % | 1,000 (54 cas) | 0,999 |

Glaucome sur gros plans de la papille (JSIEC) : **AUROC 1,000**, 100 % des cas signalés
(v0.2.0 : 0,872 et 54 %).

**Décision de publication.** La comparaison automatique avec la v0.2.0 a relevé une baisse sur la
rétinopathie hypertensive dans JSIEC (0,979 → 0,945, significative en bootstrap apparié, mais sur 15 cas
seulement ; stable en validation croisée sur 193 cas). La version a été publiée malgré tout pour le gain
sur le glaucome, décision écrite dans la fiche du registre. Cause probable : sur les gros plans,
l'étiquette de ce trouble rare était ignorée ; l'essai suivant ne recadre plus les images positives pour
un trouble hors papille. Retour à la v0.2.0 : `gh workflow run deploiement -f version=v0.2.0`.

Limite : la photo de manuel très serrée testée par l'utilisateur restait manquée (14 %, seuil 18 %) ; voir la version 0.3.2.

## Version 0.3.2 : même professeur, sans perdre l'hypertension

Essai suivant, avec une seule règle en plus : une image positive pour un trouble visible hors de la
papille (diabète, DMLA, hypertension, cataracte) n'est jamais recadrée en gros plan, pour que ces
troubles gardent tous leurs exemples. Comparaison sur JSIEC, mêmes images pour les trois modèles :

| Trouble | v0.2.0 | v0.3.0 / v0.3.1 | v0.3.2 |
|---|---|---|---|
| Rétinopathie diabétique | 0,942 | 0,947 | 0,943 |
| Glaucome (image entière) | 0,997 | 0,998 | 1,000 |
| Glaucome (gros plan serré) | 0,872 | 1,000 | 0,997 |
| DMLA | 0,953 | 0,966 | 0,953 |
| Rétinopathie hypertensive | 0,979 | 0,945 | 0,964 |
| Myopie forte | 0,999 | 1,000 | 1,000 |

Test apparié (bootstrap sur les mêmes images) : face à la v0.2.0, la v0.3.2 ne recule
significativement sur aucun trouble (hypertension -0,015, IC 95 % [-0,043 ; +0,004]) et progresse sur
le glaucome ; face à la v0.3.1, elle rend le gain sur la DMLA (-0,013, significatif). En validation
croisée, tout est stable (DMLA 0,943, hypertension 0,863). La photo de manuel très serrée testée par
l'utilisateur est désormais repérée (glaucome « possible », 22 %, seuil 18 %). Publiée en v0.3.2 :
aucune baisse significative face au modèle d'avant RETFound, le gain sur le glaucome conservé.

## Niveaux affichés dans la démo

Le seuil « probable » est commun : plus d'une chance sur deux. Le seuil « possible » dépend du
trouble : il est placé pour signaler environ 8 yeux atteints sur 10 en validation croisée, entre
3 % et 20 %. Sans cela, un trouble rare dans les données (la rétinopathie hypertensive ne concerne
que 2 % des yeux) n'atteint presque jamais 20 % même quand le réseau classe bien les patients :
avec un seuil unique de 20 %, seuls 7 % des cas de JSIEC étaient signalés, contre 87 % avec le
seuil adapté (et 94 % des yeux sans ce trouble restent « peu probable »).

| Trouble | « Possible » dès | Cas JSIEC signalés | Yeux JSIEC sans le trouble restés « peu probable » |
|---|---:|---:|---:|
| Rétinopathie diabétique | 20 % | 87 % | 85 % |
| Glaucome | 16 % | 100 % | 84 % |
| Cataracte | 20 % | — | — |
| DMLA | 8,4 % | 95 % | 82 % |
| Rétinopathie hypertensive | 3 % | 87 % | 94 % |
| Myopie forte | 20 % | 100 % | 96 % |

Les pourcentages eux-mêmes restent calibrés ; seul le mot affiché change.

## Cartes de chaleur et contrôle de qualité

- **« Voir les zones »** : le modèle ONNX exporte, en plus des scores, une carte 12 × 12 par trouble
  (carte d'activation de classe, CAM) : la couche finale d'EfficientNet est une moyenne des zones de
  l'image suivie d'une combinaison linéaire, donc la carte est exacte, sa moyenne redonne le score.
  La démo la superpose à l'image en rouge et jaune. Pour le glaucome, elle se concentre sur la
  papille, comme l'examen d'un ophtalmologiste ; ce n'est pas pour autant le contour d'une lésion.
- **Qualité de l'image** ([qualite.py](../src/vaisseaux/qualite.py)) : netteté (laplacien du canal
  vert), luminosité, surexposition et contraste. Sur JSIEC, la netteté sépare les 159 photos classées
  « fond d'œil flou » par les ophtalmologistes des 841 autres avec une AUROC de 0,996. Seuils :
  « image floue » sous 0,18 (81 % des photos floues repérées, 0,4 % des photos nettes signalées à
  tort), « un peu floue » sous 0,25 (96 % et 4 %). La démo avertit quand la qualité est moyenne ou
  insuffisante, au chargement et au-dessus des résultats.

## Reproduire

```bash
python -m vaisseaux.sources_troubles              # prépare et dédoublonne les images (data/troubles/)
python -m vaisseaux.troubles                      # entraîne, calibre, évalue (GPU)
python -m vaisseaux.troubles --recalculer         # refait calibrage et seuils sans réentraîner
python scripts/evaluer_externe.py --version v0.1.1   # compare une version du registre sur JSIEC
python scripts/tester_images.py --images <dossier> --modele nouveau=modeles/troubles.onnx:resultats/troubles/troubles_demo.json
```

L'entraînement peut aussi tourner sur Kaggle (`--sans-onnx --poids-initiaux <fichier>`), puis
l'export se fait en local avec `--exporter modeles/troubles.pt`. Les résultats détaillés, les
prédictions et la figure sont dans [resultats/troubles/](../resultats/troubles/).

## Dans la démo

L'image importée est recadrée automatiquement sur le fond d'œil ; une angiographie ou une photo en
noir et blanc est refusée avec un message clair. Le résultat est présenté avec une phrase
synthétique, une barre par trouble et un niveau lisible ; les chiffres techniques (seuils, AUROC
en validation croisée et sur l'hôpital jamais vu) sont repliés. Le modèle s'exécute dans le
navigateur : les images ne sont envoyées à aucun serveur.
