# ia-sport
Analyse automatique des actions en basket-ball (YOLOv8, C3D, I3D)

Ce projet vise à détecter et analyser automatiquement des actions de basket-ball à partir de vidéos à l’aide de la vision par ordinateur et du deep learning. Il combine :

YOLOv8 pour la détection d’objets (joueurs, ballon, panier, états du tir)

C3D pour la reconnaissance d’actions à partir de clips vidéo courts

I3D (Inflated 3D ConvNet) pour une reconnaissance d’actions plus robuste et temporelle

Des règles spatiales et heuristiques pour améliorer l’interprétation des tirs (lay-up, dunk, jump shot)

Le projet a été réalisé dans le cadre du module IA & Sport.

Les modeles utilisés: https://drive.google.com/drive/folders/15BEv9xbFA01Qay-NmA8whOSEt8nH-6uX?usp=sharing

# Objectifs
Détecter automatiquement les joueurs, le ballon et le panier dans une vidéo de basket

Identifier les types de tirs :

Jump shot

Lay-up

Dunk

Shot block

Combiner détection spatiale (YOLO) et analyse temporelle (C3D / I3D)

Générer une vidéo annotée avec les actions détectées

# Structure du projet

# Structure du projet

```text
├── Basketball.ipynb            # Script principal (issu du notebook Colab)
├── dataset/                    # Dataset pour I3D (train / val / test)
│   ├── train/
│   ├── val/
│   └── test/
├── basketball_actions/         # Dataset pour C3D (dunk / jump-shoot / lay-up)
├── runs/                       # Résultats YOLO (poids, labels, prédictions)
├── c3d_basketball_best.pth     # Meilleur modèle C3D entraîné
├── best.pt                     # Poids YOLOv8 entraînés
├── output.mp4                  # Vidéo de sortie annotée
└── README.md
```

# Exécution du projet

1) Entraînement YOLOv8 (détection d’objets)

```text
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=100)
```

Classes détectées :

- player / person
- ball
- rim / basket
- ball-in-basket
- player-jump-shot
- player-layup-dunk
- player-shot-block

2) Inférence YOLO sur une vidéo
```text
model.predict(
    source="vid3.mp4",
    conf=0.25,
    save=True,
    save_txt=True
    )
```
Génère :
une vidéo annotée
des fichiers .txt par frame (bounding boxes + scores)

4) Entraînement du modèle I3D
Backbone : ResNet-50 I3D pré-entraîné (Kinetics)

Fine-tuning progressif (warmup + finetune)

```text
python train_i3d.py
```
Classes I3D :
- layup
- dunk
- jump_shoot

5) YOLO + C3D
```text
process_video(
  video_path,
  labels_player_path,
  output_path,
  yolo_model,
  c3d_model,
  C3D_CLASSES
  )
```

YOLO avec règles spatiales (analyse événementielle avancée)

En complément des modèles d’apprentissage profond, le projet intègre une approche hybride combinant YOLO et des règles spatiales et temporelles inspirées de la connaissance métier du basket-ball.

Principe

YOLO détecte par frame :
- joueurs
- ballon
- panier

états du tir (jump-shot, layup/dunk, ball-in-basket)

Des règles logiques exploitent la co-occurrence, la durée et la position spatiale de ces classes pour :

- détecter le début d’un tir
- déterminer s’il est réussi ou raté
- distinguer lay-up et dunk via des heuristiques géométriques

Exemples de règles utilisées

Détection d’un tir :
- présence de player-jump-shot ou player-layup-dunk pendant plusieurs frames consécutives

Lay-up vs Dunk :
- comparaison entre :

- - la hauteur du joueur (bbox)
  - la position verticale du panier

- si la tête/bras du joueur dépasse significativement le niveau de l’arceau → dunk

Filtrage temporel :
- fenêtres glissantes
- seuils de confiance
- lissage pour éviter les faux positifs

Avantages de cette approche

- Pas besoin d’un gros dataset annoté
- Interprétable et contrôlable
- Complémentaire aux modèles C3D / I3D
- Permet une détection événementielle assez fiable (début / fin de tir)

# Visualisation avancée

Le module inclut également :
ralenti automatique lors d’un tir
zoom dynamique sur le joueur tireur

affichage en direct :
type de tir
distance au panier

Cette partie s’appuie sur :
Roboflow Inference
Supervision
sports.basketball (court detection & geometry)

# Résultats principaux

YOLOv8 :

Détection fiable des joueurs, ballon et panier

Bonne localisation spatiale des actions

Affichage trop court du type de tir sur la vidéo, ce qui peut nuire à la lisibilité


C3D :

Correct pour des clips courts

Sensible au bruit et au contexte

Toujour donne jumpshoot pour tout les types de tir


I3D :

Meilleure stabilité temporelle

Meilleure séparation entre lay-up / dunk / jump-shot

Précision test typique : ~48% (selon split)


YOLO avec regles spatiaux :

Vidéos finales annotées avec :

bounding boxes

type de tir


Limites

Dataset limité en taille

Sensible aux angles de caméra

Confusion possible lay-up / dunk sans informations 3D
