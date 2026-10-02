# Mesures vasculaires autour de la papille

Les marqueurs historiques de la démo (densité, calibre, tortuosité… sur toute l'image, en pixels)
changent avec la résolution et le cadrage. Les logiciels de recherche en oculomique (SIVA, VAMPIRE,
AutoMorph) mesurent donc dans des **anneaux centrés sur la papille** et expriment les longueurs **en
diamètres de papille**. Le module [zones.py](../src/vaisseaux/zones.py) fait de même. Depuis la
v0.4.0, les vaisseaux viennent d'un **U-Net multi-appareils** qui distingue aussi artères et veines
([unet_av.py](../src/vaisseaux/unet_av.py)).

## Papille et zones

- **Centre** : zone claire où convergent les gros vaisseaux (la clarté seule se laisse tromper par
  un bord surexposé), affinée sur le point le plus clair du voisinage.
- **Diamètre** : profil radial de la clarté (médiane par anneau, pour ignorer les vaisseaux qui
  traversent la papille) ; le bord est là où la clarté redescend au quart de la hauteur entre le
  fond et le pic. Si la valeur sort des proportions habituelles (1/12 à 1/4 du champ de vue), on
  retient 1/7 du champ de vue et la démo le signale.
- **Zone B** : de 0,5 à 1 diamètre de papille du bord (1 à 1,5 depuis le centre) ; **zone C** : de
  0,5 à 2 diamètres du bord (1 à 2,5 depuis le centre). 1 diamètre de papille ≈ 1 800 µm : les
  valeurs en µm sont des estimations, la taille réelle de la papille variant d'une personne à l'autre.

## Le U-Net multi-appareils (v0.4.0)

Le premier U-Net n'avait vu que 18 photos d'un seul appareil (DRIVE). Sur les photos d'autres
appareils, il dessinait les vaisseaux trop larges ou en manquait, et les mesures dépendaient surtout
de l'appareil (voir « Repères » plus bas).

**Données** ([preparer_vaisseaux_multi.py](../scripts/preparer_vaisseaux_multi.py)) : 926 photos
de 5 bases, chacune préparée comme dans la démo (isolement de l'œil, 800 pixels au plus).

| Base | Origine | Annotations | Entraînement | Validation | Test |
|---|---|---|---|---|---|
| DRIVE_AV (RITE) | Pays-Bas | artères / veines | 16 | 4 | 20 |
| HRF-AV | Allemagne et Tchéquie | artères / veines | 21 | 3 | 21 |
| LES-AV | Royaume-Uni | artères / veines | 9 | 2 | 11 |
| FIVES | Chine | vaisseaux | 570 | 23 | 198 |
| CHASE_DB1 | Royaume-Uni (enfants) | vaisseaux | 16 | 4 | 8 (4 enfants) |

Chaque base vient d'un appareil photo différent. Les parties « test » officielles (patients séparés pour CHASE_DB1) ne servent qu'à l'évaluation
finale. 9 photos de FIVES sont écartées : la démo n'y trouve pas le fond d'œil.

**Modèle et entraînement** : U-Net à 5 niveaux (5,2 millions de paramètres, 20 Mo en ONNX), deux
sorties : « vaisseau » (toutes les bases) et « artère plutôt que veine » (pixels annotés artère ou
veine seulement, classes équilibrées). Patchs de 384 pixels, augmentations d'appareil (gain et
gamma par canal, contraste, éclairage inégal, flou, bruit, JPEG, échelle de 0,7 à 1,35). 16 000
itérations en 62 minutes sur un GPU T4 (Kaggle). L'itération retenue (15 000) est celle du meilleur
score de validation.

Deux essais ont échoué avant celui-ci, et les corrections sont restées dans le code :

- après augmentation, un canal presque uniforme donnait des entrées jusqu'à 94 000, hors de la
  plage du float16. Les entrées sont maintenant bornées (écart-type plancher 0,02, valeurs à ±8),
  dans la démo aussi : sur le premier U-Net, le Dice ne change pas à 0,001 près ;
- une divergence tardive (itération 9 600) a fait perdre le meilleur état, qui n'était gardé qu'en
  mémoire. Le meilleur modèle est maintenant enregistré à chaque progrès, les gradients sont
  écrêtés, et en cas de divergence on repart du meilleur état.

### Résultats sur les parties test

Dice des vaisseaux (seuil 0,5), les deux modèles sur les mêmes images ; écart apparié avec IC 95 %
par bootstrap :

| Base (test) | Premier U-Net | U-Net multi-appareils | Écart [IC 95 %] |
|---|---|---|---|
| DRIVE_AV (20) | 0,844 | 0,837 | −0,008 [−0,012 ; −0,004] |
| HRF-AV (21) | 0,735 | 0,795 | +0,060 [+0,052 ; +0,068] |
| LES-AV (11) | 0,641 | 0,868 | +0,228 [+0,140 ; +0,326] |
| CHASE_DB1 (8) | 0,598 | 0,806 | +0,208 [+0,170 ; +0,246] |
| FIVES (198) | 0,648 | 0,870 | |

Sur les annotations d'origine de DRIVE (validation du registre), le recul est de 0,823 à 0,812.

Artères et veines, sur les 52 photos de test annotées : 94 à 97 % des pixels de vaisseau bien
classés (97 à 99 % sur les gros vaisseaux) ; **92 % des segments de la zone B bien classés**, contre
73 % pour la méthode sans apprentissage ([evaluer_arteres_veines.py](../scripts/evaluer_arteres_veines.py)).

### Règles fixées avant les résultats, et décision

Fixées avant l'entraînement : le nouveau modèle ne remplace l'ancien que si les mesures en zones ne
sont pas moins fiables et s'il ne recule pas sur DRIVE ; l'AVR ne s'affiche que si sa corrélation
avec l'AVR de l'expert atteint 0,6 ; un repère « yeux sains » ne s'affiche que si les yeux des
autres appareils tombent, en médiane, entre ses 25e et 75e percentiles.

Le modèle **enfreint deux règles, de peu** : le Dice recule sur DRIVE (−0,008, significatif mais
faible), et la tortuosité est un peu moins proche de l'expert (0,72 → 0,67 sur 52 images, écart
non testé). Il a été **publié quand même, par décision humaine**, pour les gains sur tous les autres
appareils et sur les autres mesures. Ces deux reculs sont écrits dans la fiche du registre.

## Mesures et fiabilité

La même mesure, calculée sur les vaisseaux du U-Net et sur ceux de l'expert, sur les **52 photos de
test** de DRIVE_AV, HRF-AV et LES-AV ([fiabilite_zones.py](../scripts/fiabilite_zones.py),
[fiabilite_zones.json](../resultats/fiabilite_zones.json)). Pour CRAE, CRVE et AVR, la référence
utilise les vaisseaux et les classes artère/veine de l'expert.

| Mesure | Premier U-Net : corrélation de rang / écart médian | U-Net multi-appareils |
|---|---|---|
| Calibre des 6 plus gros vaisseaux, zone B | 0,80 / 20 % | **0,85 / 12 %** |
| Tortuosité pondérée, zone C | 0,72 / 1 % | 0,67 / 1 % |
| Densité vasculaire, zone C | 0,76 / 19 % | **0,92 / 11 %** |
| Densité de longueur, zone C | 0,84 / 14 % | **0,88 / 9 %** |
| CRAE (artérioles), zone B | — | 0,73 / 16 % |
| CRVE (veinules), zone B | — | 0,85 / 12 % |
| AVR (CRAE / CRVE) | — | **0,16** / 14 % |

Ces corrélations mêlent trois bases : une partie de l'accord vient des différences entre bases.
Base par base, CRAE est moins bien suivi (HRF 0,61, LES-AV 0,24 sur 11 images) que CRVE (0,79 à 0,89).

La démo affiche aussi une **marge** : la mesure refaite avec un U-Net un peu plus strict ou un peu
plus permissif (seuil 0,4 et 0,6 au lieu de 0,5).

### L'AVR : un résultat négatif qui demeure

Avec les vaisseaux de l'expert, les classes du modèle donnent un AVR très proche de celui de
l'expert (corrélation 0,85). Avec la chaîne complète, c'est 0,16 : le classement n'est plus en
cause, ce sont les **largeurs** mesurées sur la segmentation. L'AVR est un rapport de deux
largeurs de quelques pixels, qui varie peu d'un œil à l'autre (0,58 à 0,73 entre les quartiles des
yeux sains) : une erreur d'un pixel suffit à le fausser. La démo n'affiche donc que CRAE et CRVE.
Piste : mesurer la largeur au dixième de pixel, sur le profil de probabilité du U-Net plutôt que sur
la segmentation binaire.

## Repères « yeux sains »

[reperes_sains.py](../scripts/reperes_sains.py) passe 706 yeux sains de RFMiD (trois appareils, Inde)
et de JSIEC (Chine) dans la chaîne de la démo ([reperes_zones.json](../resultats/reperes_zones.json),
687 papilles mesurées). Contrôle du biais d'appareil : les photos de test de DRIVE_AV, HRF-AV et
LES-AV, elles aussi surtout saines, devraient tomber autour du 50e percentile. Rang centile médian :

| Mesure | Premier U-Net (DRIVE / HRF / LES) | U-Net multi-appareils (DRIVE / HRF / LES) |
|---|---|---|
| Calibre, zone B | 95 / 84 / 12 | 86 / 58 / 34 |
| Tortuosité, zone C | 87 / 66 / 21 | 83 / 56 / 65 |
| Densité, zone C | 94 / 95 / 6 | 91 / 67 / 11 |
| Densité de longueur, zone C | 31 / 70 / 13 | **32 / 67 / 26** |
| CRAE | — | 75 / 54 / 31 |
| CRVE | — | 83 / 57 / 43 |

Le biais d'appareil a nettement baissé, surtout pour HRF, mais il reste fort pour DRIVE (photos
plus petites, 565 pixels de large) et pour la densité. **Seule la densité de longueur passe le
contrôle** : la démo situe cette mesure parmi les yeux sains, et seulement elle. CRAE manque le
contrôle de peu (75,4 pour DRIVE).

Les repères restent plausibles : CRAE médian 152 µm, CRVE 232 µm et AVR 0,65, des ordres de
grandeur proches de ceux des grandes cohortes. Ce ne sont pas des normes cliniques : ils viennent de
deux bases, sans âge ni tension artérielle connus.

## Premier U-Net (avant v0.4.0)

Fiabilité sur les 107 images annotées, toutes parties confondues :
[fiabilite_zones_unet_drive.json](../resultats/fiabilite_zones_unet_drive.json). Classement des
artères et veines sans apprentissage (clarté rouge et verte, deux groupes par quadrant) : 69 à 73 %
de segments bien classés et un AVR sans valeur
([arteres_veines.json](../resultats/arteres_veines.json)). Repères yeux sains sans contrôle du biais
d'appareil : [reperes_zones_unet_drive_sans_controle.json](../resultats/reperes_zones_unet_drive_sans_controle.json).
