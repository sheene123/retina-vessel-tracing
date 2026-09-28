---
title: Tracé des vaisseaux rétiniens
emoji: 👁️
colorFrom: red
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Cliquez deux points, Dijkstra suit le vaisseau
---

# Tracé des vaisseaux rétiniens par plus court chemin

Démo interactive du projet [retina-vessel-tracing](https://github.com/sheene123/retina-vessel-tracing).

- **Tracer** : cliquez deux points sur un vaisseau ; l'image est un graphe dont chaque pixel est un sommet
  et un plus court chemin (Dijkstra ou A\*) suit le vaisseau.
- **Comparer** : deux prétraitements tracés côte à côte, et les cartes « vaisseaux » des six chaînes.
- **Évaluer** : sur les images DRIVE, le tracé est noté face à l'annotation experte (précision, couverture,
  F1, distance de Fréchet).
- **Tester vos images** : JPEG, PNG ou TIFF.

Le code Python du projet tourne **dans votre navigateur** grâce à [Pyodide](https://pyodide.org) :
aucune image n'est envoyée à un serveur. Les exemples proviennent du jeu de test public DRIVE (Staal et al.,
*IEEE TMI*, 2004), téléchargés depuis un miroir public et non redistribués ici.
