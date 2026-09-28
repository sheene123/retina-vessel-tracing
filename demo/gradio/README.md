---
title: Tracé des vaisseaux rétiniens
emoji: 👁️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Cliquez deux points, Dijkstra suit le vaisseau
---

# Tracé des vaisseaux rétiniens par plus court chemin

Démo interactive du projet [retina-vessel-tracing](https://github.com/sheene123/retina-vessel-tracing) :
l'image de fond d'œil est modélisée comme un graphe dont chaque pixel est un sommet ; à partir de deux
points cliqués, un plus court chemin (Dijkstra ou A\*) sur une carte de coût issue du filtre de Frangi
suit le vaisseau de l'un à l'autre.

Les images d'exemple proviennent du jeu de test public DRIVE (Staal et al., *IEEE TMI*, 2004) et sont
téléchargées au démarrage depuis un miroir public ; elles ne sont pas redistribuées ici.
