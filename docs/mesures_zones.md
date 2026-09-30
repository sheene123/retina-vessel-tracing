# Mesures vasculaires autour de la papille

Les marqueurs historiques de la démo (densité, calibre, tortuosité… sur toute l'image, en pixels)
changent avec la résolution et le cadrage. Les logiciels de recherche en oculomique (SIVA, VAMPIRE,
AutoMorph) mesurent donc dans des **anneaux centrés sur la papille** et expriment les longueurs **en
diamètres de papille**. Le module [zones.py](../src/vaisseaux/zones.py) fait de même, sans entraîner de
modèle. Ces mesures ne dépendent plus de la résolution ni du cadrage ; elles dépendent encore de
l'appareil photo, à travers la segmentation du U-Net (voir « Repères »).

## Papille et zones

- **Centre** : zone claire où convergent les gros vaisseaux (la clarté seule se laisse tromper par
  un bord surexposé), affinée sur le point le plus clair du voisinage.
- **Diamètre** : profil radial de la clarté (médiane par anneau, pour ignorer les vaisseaux qui
  traversent la papille) ; le bord est là où la clarté redescend au quart de la hauteur entre le
  fond et le pic. Si la valeur sort des proportions habituelles (1/12 à 1/4 du champ de vue), on
  retient 1/7 du champ de vue et la démo le signale : c'est arrivé sur 5 des 107 images annotées.
- **Zone B** : de 0,5 à 1 diamètre de papille du bord (1 à 1,5 depuis le centre) ; **zone C** : de
  0,5 à 2 diamètres du bord (1 à 2,5 depuis le centre). 1 diamètre de papille ≈ 1 800 µm : les
  valeurs en µm sont des estimations, la taille réelle de la papille variant d'une personne à l'autre.

## Mesures et fiabilité

Fiabilité mesurée sur les 107 images annotées de DRIVE_AV (RITE), HRF-AV et LES-AV : la même mesure
calculée sur les vaisseaux du U-Net et sur ceux de l'expert.

| Mesure | Classe les yeux comme l'expert (corrélation de rang) | Écart relatif médian |
|---|---|---|
| Calibre des 6 plus gros vaisseaux, zone B | 0,79 | 19 % (le U-Net dessine les vaisseaux un peu plus larges) |
| Tortuosité pondérée, zone C | 0,63 | 1 % |
| Densité vasculaire, zone C | 0,63 | 19 % |
| Densité de longueur, zone C (par diamètre de papille) | 0,83 | 13 % |

La démo affiche aussi une **marge** : la mesure refaite avec un U-Net un peu plus strict ou un peu
plus permissif (seuil 0,4 et 0,6 au lieu de 0,5). Une mesure qui bouge beaucoup dépend fortement
de la segmentation.

## Repères « yeux sains » : pourquoi la démo n'en affiche pas

[reperes_sains.py](../scripts/reperes_sains.py) a passé 706 yeux sains de RFMiD (trois appareils,
Inde) et de JSIEC (Chine) dans exactement la même chaîne que la démo (isolement de l'œil, U-Net,
papille, zones), pour situer chaque mesure parmi des yeux sains
([reperes_zones.json](../resultats/reperes_zones.json), 667 papilles mesurables) :

| Mesure | 5 % | 25 % | médiane | 75 % | 95 % |
|---|---|---|---|---|---|
| Calibre des gros vaisseaux, zone B (µm estimés) | 79 | 97 | 111 | 132 | 168 |
| Tortuosité, zone C | 1,078 | 1,089 | 1,098 | 1,111 | 1,151 |
| Densité vasculaire, zone C | 6,1 % | 9,0 % | 10,4 % | 12,0 % | 14,5 % |
| Densité de longueur, zone C | 1,36 | 2,09 | 2,44 | 2,88 | 3,80 |

Mais les yeux de DRIVE et de HRF, eux aussi majoritairement sains, tombent largement au-dessus de
ces repères, alors que leur papille occupe la même part du champ de vue (0,12 à 0,13) :

| Base | Calibre, zone B : au-dessus du 95e percentile | Densité, zone C : au-dessus du 95e percentile |
|---|---|---|
| DRIVE_AV (40 images) | 52 % (médiane 170 µm contre 111) | 28 % |
| HRF-AV (45 images) | 29 % (médiane 140 µm) | 44 % |

Le U-Net, entraîné sur DRIVE seulement, ne dessine pas les vaisseaux de la même façon sur les photos
d'autres appareils : les repères mesurent surtout l'appareil, pas la biologie. Une photo saine de
DRIVE apparaîtrait « très haut » : la démo n'affiche donc aucune comparaison à des yeux sains. Des
mesures en zones et en diamètres de papille ne dépendent plus de la résolution ni du cadrage, mais
elles dépendent encore de l'appareil à travers la segmentation. Pour des repères utilisables, il
faudrait un U-Net entraîné sur plusieurs appareils (FIVES, CHASE_DB1, HRF, STARE) et des repères par
appareil ou par âge.

## Artères et veines : un résultat négatif

Les marqueurs vasculaires les mieux validés en clinique sont le calibre équivalent des artérioles
(CRAE), des veinules (CRVE) et leur rapport (AVR), liés à l'hypertension et au risque
cardiovasculaire. Il faut pour cela distinguer artères et veines. Sans entraîner de modèle, on a
essayé la méthode classique : dans la zone B, une artère est plus claire que la veine voisine ; un
regroupement en deux classes par quadrant autour de la papille les sépare
([evaluer_arteres_veines.py](../scripts/evaluer_arteres_veines.py)). Réglages choisis sur les
parties « training », résultats sur les parties « test » des trois bases annotées :

| Vaisseaux | Segments bien classés | AVR : corrélation avec l'AVR de l'expert |
|---|---|---|
| de l'expert (classement seul) | 69 à 71 % | 0,38 à 0,62 |
| du U-Net (chaîne de la démo) | 72 à 73 % | -0,62 à 0,20 |

Ajouter la largeur ou le reflet central n'aide pas. À 70 % de segments bien classés, l'AVR n'a pas
de valeur : la démo ne l'affiche donc pas. Il faudrait un modèle entraîné à distinguer artères et
veines ; les trois bases utilisées ici (DRIVE_AV, HRF-AV, LES-AV, 107 images) le permettraient.
