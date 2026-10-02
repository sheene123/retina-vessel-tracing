# Comment évaluer objectivement un tracé face à une vérité terrain experte ?

Ce document explique les choix d'évaluation du projet. La question est plus délicate
qu'il n'y paraît : on compare une **ligne** (le tracé calculé) à une **surface** (le
masque des vaisseaux dessiné par un annotateur), et cette surface est elle-même incertaine.

## 1. Ce que l'on compare vraiment

L'algorithme produit un chemin d'un pixel de large entre deux points. La vérité terrain
de DRIVE est une carte binaire de tous les vaisseaux, dessinée à la main. Trois
conséquences :

- **Les métriques de segmentation ne s'appliquent pas telles quelles.** Un tracé parfait
  d'un pixel de large a un Dice très faible face à un vaisseau de 8 pixels de large.
- **« Rester dans un vaisseau » ne suffit pas.** Un tracé qui saute sur le vaisseau voisin
  à un croisement reste à 100 % dans des pixels annotés, et il est pourtant faux.
- **L'annotation n'est pas la vérité.** Les bords des vaisseaux sont incertains à 1 ou
  2 pixels près, les capillaires fins sont annotés de façon inégale, et deux experts ne
  produisent pas le même masque.

## 2. Construire une référence pour un tracé

Pour chaque image de test, on tire des paires de points **sur le squelette** de
l'annotation (graine fixe, donc paires reproductibles). On garde les paires reliées le
long du squelette par un chemin de 40 à 150 pixels, loin du bord du champ de vue. Le
**chemin géodésique sur le squelette** entre les deux points sert de référence : c'est le
tracé qu'aurait fait l'annotateur.

Cette référence a ses propres défauts, qu'il faut garder en tête en lisant les chiffres :
- la squelettisation crée de petites branches parasites et des boucles ;
- en 2D, un croisement artère/veine apparaît connecté : la référence peut changer de
  vaisseau au croisement, exactement comme l'algorithme. Seules des étiquettes
  artère/veine permettraient de trancher.

## 3. Mesures de tracé

| Mesure | Définition | Ce qu'elle détecte | Angle mort |
|---|---|---|---|
| Précision à τ | part des pixels du tracé à moins de τ px d'un vaisseau annoté | sorties du vaisseau | ne voit pas un changement de vaisseau |
| Couverture à τ | part de la référence à moins de τ px du tracé | mauvais vaisseau, raccourci | sensible aux défauts du squelette |
| F1 à τ | moyenne harmonique des deux | synthèse | masque le type d'erreur |
| HD95 | 95ᵉ centile de la distance de Hausdorff symétrique | écart géométrique maximal, robuste aux points isolés | ignore l'ordre du parcours |
| Fréchet discrète | plus courte « laisse » pour parcourir les deux courbes dans le même sens | allers-retours, boucles | très sensible à un seul écart |
| Ratio de longueur | longueur du tracé / longueur de la référence | < 1 raccourci, > 1 détour | compense les erreurs opposées |
| Plus long écart | plus longue portion continue hors vaisseau (au-delà de τ) | l'erreur grave : traverser le fond | nulle si le saut se fait vers un autre vaisseau |

La tolérance τ = 2 px correspond à l'incertitude des bords annotés et au demi-calibre
des vaisseaux moyens. La rendre locale, proportionnelle au calibre mesuré par une
transformée en distance, est une piste d'amélioration.

## 4. Mesures pixel, pour juger le prétraitement

Le prétraitement produit une carte de rehaussement ; on la compare au masque annoté.

- **Uniquement dans le champ de vue.** Le fond noir hors du disque est trivialement
  « négatif » ; l'inclure gonfle la spécificité et l'exactitude.
- **L'exactitude est trompeuse.** Environ 87 % des pixels du champ de vue sont du fond,
  si bien qu'une image vide obtient 0,87 d'exactitude. On lui préfère l'**AUC** (sans
  seuil), le **Dice** et surtout le **MCC**, robuste au déséquilibre des classes (Chicco &
  Jurman, 2020).
- **Le seuil de binarisation est choisi sur l'entraînement**, jamais sur le test.
- **La topologie compte.** Un vaisseau coupé en deux perd peu de Dice mais casse le
  tracé. Le **clDice** (Shit et al., 2021) compare squelettes et surfaces et pénalise les
  coupures. La fonction CAL (connectivité, aire, longueur ; Gegúndez-Arias et al., 2012)
  suit la même idée.
- **Accord inter-experts.** Sur la partie test officielle de DRIVE, un second observateur
  a annoté les images. Ses scores face au premier donnent une borne réaliste : un
  algorithme « meilleur que le second observateur » mesure surtout sa ressemblance au
  style du premier. Le benchmark la calcule quand `2nd_manual/` est présent (archive
  officielle ; le miroir public utilisé par défaut ne la contient pas).

## 5. Rigueur statistique

- **L'unité statistique est l'image**, pas la paire de points : les paires d'une même
  image partagent contraste, bruit et pathologie. Les intervalles de confiance sont donc
  obtenus par **bootstrap sur les images** (on rééchantillonne des images entières).
- **Comparaisons appariées** : chaque configuration est évaluée sur les mêmes images et
  les mêmes paires. On compare les moyennes par image avec un **test de Wilcoxon
  apparié**, puis on corrige la multiplicité des comparaisons par la méthode de **Holm**.
- **Séparation stricte** : le paramètre de coût α et le seuil sont choisis sur les
  20 images d'entraînement, puis la mesure est faite une seule fois sur les 20 images de
  test.
- **Taille d'échantillon** : 20 images, c'est peu. Une différence plus petite que la
  largeur des intervalles de confiance n'est pas une conclusion.

## 6. Ce que disent les résultats

Voir [resultats/evaluation.md](../resultats/evaluation.md) pour les tableaux complets.

Premier benchmark : 20 images de test, 10 paires par image (200 tracés par
configuration), 6 chaînes de prétraitement.

**1. Le tracé est fiable localement, moins globalement.** La précision atteint 0,995 : le
tracé ne quitte presque jamais un vaisseau, et le plus long écart hors vaisseau fait
moins d'un pixel en moyenne. La couverture n'est pourtant que de 0,83 : en moyenne,
17 % de la référence n'est pas suivie, parce que le tracé emprunte une autre route
**tout en restant dans des vaisseaux** (autre branche, croisement, défaut du squelette).
C'est l'erreur dominante. Elle est invisible si l'on se contente de vérifier que « le
tracé est dans un vaisseau ».

**2. Une meilleure carte pixel ne donne pas un meilleur tracé.** Le flou gaussien améliore
nettement toutes les mesures pixel face au canal vert brut : AUC de 0,868 à 0,905, MCC
de 0,593 à 0,669, clDice de 0,623 à 0,703 (p Holm < 10⁻⁵). Il n'améliore pas le tracé :
le F1 passe de 0,884 à 0,876, sans différence significative. Le filtre bilatéral, meilleur
que le canal brut au pixel (AUC 0,887), est de loin le **pire** pour le tracé (F1 0,772,
p Holm < 10⁻⁵). Hypothèse à vérifier : le lissage élargit les vaisseaux et comble
l'espace entre vaisseaux voisins, ce qui ouvre des raccourcis au plus court chemin.
Leçon : **il faut évaluer la tâche elle-même, pas une mesure de substitution**.

**3. Significatif ne veut pas dire important.** Le NL-means améliore l'AUC de façon « très
significative » (les 20 images vont dans le même sens, p Holm < 10⁻⁵, le minimum
atteignable avec 20 images), mais de 0,006 seulement, et sans effet sur le tracé.
Il faut toujours lire la taille d'effet et l'intervalle de confiance avant la p-valeur.

**4. Le réglage retenu est simple.** Sur l'entraînement, α = 1 est choisi pour toutes les
configurations. Un coût plus contrasté (α = 2 ou 3) n'aide pas.

**5. Guider le tracé par le U-Net réduit les écarts, sans gain de F1 démontré.** Deux
configurations ajoutées remplacent la carte de Frangi par la probabilité « vaisseau » d'un
U-Net : le premier, entraîné sur DRIVE seul, et le U-Net multi-appareils
([mesures_zones.md](mesures_zones.md)). Aucun des deux n'a vu les 20 images de test. Le choix
de α sur l'entraînement est optimiste pour eux, car ils ont appris sur ces images ; α = 1 est
retenu dans les deux cas.

| Carte | F1 tracé | Couverture | Fréchet (px) | HD95 (px) |
|---|---|---|---|---|
| CLAHE + NL-means (meilleur filtre) | 0,885 [0,854 ; 0,915] | 0,836 | 10,1 [7,3 ; 12,8] | 9,0 |
| U-Net DRIVE | 0,908 [0,886 ; 0,928] | 0,858 | 5,7 [4,3 ; 7,5] | 4,8 |
| U-Net multi-appareils | 0,921 [0,898 ; 0,940] | 0,880 | 6,0 [4,1 ; 8,5] | 5,2 |

L'écart au chemin de l'expert est presque divisé par deux (Fréchet, HD95), et la couverture
gagne 4 points. Le F1 monte de 0,885 à 0,921, mais l'écart n'est **pas significatif** face au
canal vert brut (p Holm = 0,66) : 20 images ne suffisent pas à le démontrer. La démo propose
ce tracé (« Réseau U-Net »), qui lance le réseau au premier usage.

## 7. Limites et suite

- DRIVE est petit (40 images de 565 × 584 pixels acquises en 2004) et une seule
  annotation existe pour l'entraînement. La segmentation a depuis été validée sur HRF,
  LES-AV, FIVES et CHASE_DB1 ([mesures_zones.md](mesures_zones.md)) ; le tracé, lui, n'est
  évalué que sur DRIVE.
- Les étiquettes artère/veine (base RITE, construite sur DRIVE) permettraient de
  vérifier que le tracé reste sur le **même** vaisseau aux croisements.
- Une évaluation centrée sur la tâche mesurerait l'usage réel : combien de clics et de
  corrections faut-il à un utilisateur pour tracer un vaisseau donné ?

## Références

- Buades A., Coll B., Morel J.-M. *A non-local algorithm for image denoising.* CVPR, 2005.
- Chicco D., Jurman G. *The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation.* BMC Genomics, 2020.
- Dijkstra E. W. *A note on two problems in connexion with graphs.* Numerische Mathematik, 1959.
- Eiter T., Mannila H. *Computing discrete Fréchet distance.* Rapport technique, TU Vienne, 1994.
- Frangi A. F. et al. *Multiscale vessel enhancement filtering.* MICCAI, 1998.
- Gegúndez-Arias M. E. et al. *A function for quality evaluation of retinal vessel segmentations.* IEEE Transactions on Medical Imaging, 2012.
- Hart P. E., Nilsson N. J., Raphael B. *A formal basis for the heuristic determination of minimum cost paths.* IEEE Transactions on Systems Science and Cybernetics, 1968.
- Immerkær J. *Fast noise variance estimation.* Computer Vision and Image Understanding, 1996.
- Maier-Hein L. et al. *Metrics reloaded: recommendations for image analysis validation.* Nature Methods, 2024.
- Shit S. et al. *clDice — a novel topology-preserving loss function for tubular structure segmentation.* CVPR, 2021.
- Staal J. et al. *Ridge-based vessel segmentation in color images of the retina.* IEEE Transactions on Medical Imaging, 2004.
- Taha A. A., Hanbury A. *Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool.* BMC Medical Imaging, 2015.
- Zuiderveld K. *Contrast limited adaptive histogram equalization.* Graphics Gems IV, 1994.
