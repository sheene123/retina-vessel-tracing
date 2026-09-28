# Les métriques de segmentation prédisent-elles l'erreur des marqueurs vasculaires ?

## Question

En oculomique, on mesure les vaisseaux de la rétine (densité, calibre, tortuosité, complexité
du réseau) comme marqueurs de santé, notamment du risque cardiovasculaire. Ces mesures sont
calculées sur une segmentation automatique des vaisseaux, et les segmenteurs sont choisis sur
des métriques pixel : Dice, exactitude, AUC. **Une segmentation mieux notée donne-t-elle des
marqueurs plus justes ?** Si ce n'est pas le cas, on sélectionne les modèles sur le mauvais
critère.

## Ce qui existe déjà

- Giesser et al. (*Investigative Ophthalmology & Visual Science*, 2024) mesurent la
  reproductibilité test-retest de 76 marqueurs calculés par AutoMorph : certains, comme la
  tortuosité, varient fortement d'une acquisition à l'autre alors que les segmentations se
  ressemblent (F1 de 0,82 à 0,85 entre test et retest).
- Les travaux sur les métriques topologiques (clDice, CVPR 2021 ; cbDice, MICCAI 2024) montrent
  que le Dice ignore les coupures et les écarts de calibre, sans relier ces défauts aux marqueurs
  cliniques.
- *Metrics Reloaded* (Nature Methods, 2024) recommande de choisir la métrique selon la tâche
  finale, sans étude dédiée aux marqueurs vasculaires rétiniens.

Cette étude relie directement les métriques de segmentation à l'erreur de chaque marqueur, sur
des erreurs réelles et sur des erreurs contrôlées.

## Protocole

**Images.** Les 20 images de test DRIVE et l'annotation du premier expert. Le U-Net est entraîné
sur les images d'entraînement (18, plus 2 pour choisir la meilleure itération) ; les seuils des
chaînes Frangi sont ceux choisis sur l'entraînement dans le benchmark du projet.

**19 méthodes de segmentation**, en trois familles.

| Famille | Méthodes | Rôle |
|---|---|---|
| Frangi | 6 chaînes de prétraitement (canal vert, CLAHE, flou gaussien, médian, bilatéral, NL-means), seuillées | segmenteurs classiques |
| U-Net | un U-Net (Dice 0,823, AUC 0,980 sur le test) à trois seuils : 0,3, 0,5, 0,7 | segmenteur appris, compromis sensibilité / spécificité |
| Perturbations contrôlées de l'annotation | amincissement, épaississement (1 et 2 px), petits vaisseaux supprimés (2 niveaux), coupures (20 et 60), faux positifs (5 et 15 % de la surface), contours bruités | chaque perturbation isole **un seul type d'erreur** |

**9 métriques de segmentation**, calculées dans le champ de vue face à l'annotation : exactitude,
sensibilité, spécificité, Dice, MCC, clDice, erreur sur le nombre de composantes (β0), erreur
sur le nombre de boucles (β1), distance de Hausdorff à 95 % (HD95).

**8 marqueurs vasculaires** ([biomarqueurs.py](../src/vaisseaux/biomarqueurs.py)) : densité
vasculaire, densité de longueur (squelette), dimension fractale (comptage de boîtes), calibre
moyen, calibre des gros vaisseaux (décile supérieur), tortuosité (arc / corde par segment entre
bifurcations), densité de bifurcations, nombre de fragments. **Plus une tâche** : le tracé d'un
vaisseau entre deux points (F1 face au chemin de l'expert, 10 paires par image) quand le coût du
graphe est tiré de la segmentation.

**Erreur d'un marqueur** : écart relatif à la valeur calculée sur l'annotation experte (écart
absolu pour le nombre de fragments).

**Pouvoir prédictif d'une métrique pour un marqueur.** Pour chaque image, on classe les 19
méthodes selon la métrique et selon la justesse du marqueur, puis on calcule la corrélation de
rang de Spearman entre les deux classements. On en fait la moyenne sur les 20 images, avec un
intervalle de confiance à 95 % par bootstrap sur les images. La valeur 1 signifie que la
métrique classe les méthodes exactement comme la justesse du marqueur. La valeur 0 signifie
qu'elle n'apporte aucune information, et une valeur négative qu'elle induit en erreur. Le calcul
est fait sur toutes les méthodes, puis sur les seuls segmenteurs réels (Frangi et U-Net).

## Résultats

Tableaux complets : [resultats/etude_metriques/etude.md](../resultats/etude_metriques/etude.md).
380 segmentations (20 images × 19 méthodes), calculées en 99 s.

![Pouvoir prédictif, segmenteurs réels](../resultats/etude_metriques/pouvoir_predictif_segmenteurs_reels.png)

**1. Les métriques pixel prédisent le calibre et la densité, pas la géométrie fine du réseau.**
Sur une même image et parmi les segmenteurs réels (Frangi et U-Net), la méthode la mieux classée
par le Dice est aussi la plus juste pour le calibre des gros vaisseaux (+0,81 [+0,75 ; +0,86]) et
pour la fragmentation (+0,82). En revanche, le Dice ne dit rien de la tortuosité
(+0,07 [-0,15 ; +0,29]), de la densité de longueur (+0,19 [-0,10 ; +0,45]) ni de la qualité du
tracé (+0,02 [-0,23 ; +0,26]). Pour les bifurcations, il oriente même plutôt dans le mauvais
sens (-0,18 [-0,45 ; +0,11]). Le MCC et l'exactitude se comportent comme le Dice.

**2. Pour choisir un modèle, le Dice moyen suffit pour certains marqueurs seulement.**
En classant les 9 segmenteurs réels sur leurs moyennes, le meilleur Dice désigne bien le modèle
le plus juste pour la densité (+0,95), la dimension fractale (+0,93), le calibre (+0,82 à +0,88)
et la fragmentation (+0,93). Pour la tortuosité, les bifurcations et le tracé, les intervalles de
confiance traversent zéro : on ne peut pas s'y fier.

**3. Les perturbations contrôlées révèlent des angles morts précis.**

| Erreur introduite | Ce que disent les métriques | Ce que deviennent les marqueurs |
|---|---|---|
| 60 coupures dans les vaisseaux | Dice 0,984, clDice 0,982 : quasi parfait | 31 fragments de plus, erreur de tracé de 10 % |
| Petits vaisseaux supprimés | Dice 0,898, meilleur que le U-Net | densité de longueur fausse de 45 %, bifurcations de 56 %, tracé de 25 % |
| Vaisseaux amincis | clDice 0,999 : topologie jugée parfaite | calibre faux de 35 %, densité de 37 % |
| Vaisseaux épaissis d'1 px | clDice 0,973 | calibre faux de 69 % |
| Contours bruités | Dice 0,837 | nombre de bifurcations multiplié par 7 (erreur de 610 %) |

Le Dice est aveugle aux coupures et aux petits vaisseaux manquants ; le clDice est aveugle aux
erreurs de calibre. Les deux familles sont complémentaires : sur l'ensemble des méthodes, le
Dice prédit le calibre (+0,87), le clDice la densité de longueur (+0,74), les bifurcations
(+0,64) et la fragmentation (+0,67), l'erreur β0 la fragmentation (+0,89).

**4. À Dice égal, le seuil décide du marqueur juste.** Les trois seuils du U-Net donnent le même
Dice (0,819 à 0,823). Pourtant, le seuil haut (0,7) divise presque par deux l'erreur de calibre
(16 % contre 27 % au seuil 0,3), tandis que le seuil bas donne des bifurcations et une densité de
longueur plus justes (29 % contre 38 %, 12,5 % contre 19,3 %). Il n'existe pas de meilleur seuil
unique : il faut le choisir selon le marqueur étudié.

**5. Le U-Net ne trace pas mieux que Frangi.** Malgré un Dice de 0,82 contre 0,70, le U-Net
donne la même qualité de tracé que la meilleure chaîne Frangi (11,8 % d'erreur contre 12,0 %),
ce qui confirme le résultat du benchmark de tracé à une autre échelle.

## Recommandations

1. Ne jamais évaluer une segmentation destinée à des mesures vasculaires sur le seul Dice (ou
   la seule exactitude). Publier au minimum **Dice ou MCC, clDice et erreur β0**, qui couvrent des
   angles morts différents.
2. Pour la **tortuosité, les bifurcations et le tracé**, aucune métrique de segmentation n'est
   un bon substitut : valider ces marqueurs directement contre les valeurs issues de
   l'annotation experte.
3. **Choisir le seuil de binarisation selon le marqueur**, pas selon le Dice.
4. Le comptage des bifurcations est très sensible au bruit des contours : élaguer les
   petites branches du squelette avant de compter.

## Limites

- Une seule base (DRIVE, 20 images de test, 565 × 584 pixels) et un seul annotateur : les
  marqueurs de référence héritent de ses choix. La variabilité entre experts n'est pas mesurée
  (le second observateur n'est pas dans le miroir utilisé).
- 9 segmenteurs réels seulement, d'où des intervalles larges en sélection de modèle.
- Marqueurs implémentés dans ce projet, en pixels, et non validés contre un logiciel clinique
  (SIVA, VAMPIRE). Les marqueurs artère/veine (CRAE, CRVE, rapport artères/veines) ne sont pas
  inclus faute d'étiquettes artère/veine libres d'accès.
- Les perturbations sont synthétiques : elles isolent un type d'erreur, mais ne reproduisent pas
  exactement les erreurs d'un vrai modèle.

## Suite

- Étendre à FIVES, CHASE_DB1 et HRF, et à des modèles pré-entraînés publics (AutoMorph, RETFound).
- Ajouter les marqueurs artère/veine avec des étiquettes AV (RITE, AV-DRIVE).
- Comparer l'erreur des marqueurs à la variabilité entre deux experts, pour savoir quelle
  précision est atteignable.
- Proposer une métrique orientée marqueurs, combinant calibre et topologie.

## Reproduire

```bash
pip install -e ".[dev,ml]"
python scripts/telecharger_drive.py
python -m vaisseaux.unet        # environ 2 minutes sur GPU
python -m vaisseaux.etude       # environ 1 à 2 minutes sur 10 cœurs
```

## Références

- Giesser S. D. et al. *Evaluating the impact of retinal vessel segmentation metrics on retest reliability in a clinical setting: a comparative analysis using AutoMorph.* Investigative Ophthalmology & Visual Science, 2024.
- Maier-Hein L. et al. *Metrics reloaded: recommendations for image analysis validation.* Nature Methods, 2024.
- Shit S. et al. *clDice: a novel topology-preserving loss function for tubular structure segmentation.* CVPR, 2021.
- Shi P. et al. *Centerline boundary Dice loss for vascular segmentation.* MICCAI, 2024.
- Staal J. et al. *Ridge-based vessel segmentation in color images of the retina.* IEEE Transactions on Medical Imaging, 2004.
- Zhou Y. et al. *AutoMorph: automated retinal vascular morphology quantification via a deep learning pipeline.* Translational Vision Science & Technology, 2022.
