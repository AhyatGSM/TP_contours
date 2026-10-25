# TP – Détection de contours

## Le but

Comparer trois détecteurs de contours vus en cours, Sobel, Laplace et Canny, sur les 40 images de la base de l'université de Groningen. Pour chaque image, on compare les contours trouvés par le programme avec des contours tracés à la main, et on calcule des mesures pour savoir quel détecteur s'en sort le mieux.

## Ce que fait le programme

Une fenêtre s'ouvre avec des curseurs pour choisir l'image, le détecteur (0 = Sobel, 1 = Laplace, 2 = Canny), le flou, les seuils et le voisinage. À chaque changement, le programme affiche l'image, la référence tracée à la main, les contours trouvés et une image des erreurs : vert pour les contours corrects, rouge pour les contours en trop, bleu pour les contours oubliés. Les mesures (P, TFP, TFN, précision, rappel, F-mesure) sont écrites en haut de la fenêtre.

Il y a aussi quelques touches, à taper en cliquant d'abord sur la fenêtre (pas dans le terminal) :

- `a` : teste plusieurs réglages sur les 40 images, avec et sans voisinage, et enregistre les moyennes dans `tableau_moyennes.csv` (le détail image par image est dans `resultats_auto.csv`) ;
- `l` : applique les réglages actuels des curseurs aux 40 images et ajoute les résultats dans `resultats_lot.csv` ;
- `s` : enregistre une capture de la fenêtre ;
- `q` : quitter.

Les réglages testés par la touche `a` sont dans la liste `SERIE_AUTO`, en haut de `main.cpp`.

## Le résultat

Avec le voisinage, Sobel obtient la meilleure performance (P = 0,41), devant Canny (0,32) et Laplace (0,22). Les détails et l'interprétation sont dans le rapport.

## Organisation du dossier

Les images doivent être rangées comme ceci :

```
TP_contours/
├── CMakeLists.txt
├── main.cpp
├── detecteurs.h
├── evaluation.h
├── edgedetector.h
├── laplacianZC.h
└── contours/
    └── images/
        ├── basket.pgm, bear.pgm, ...
        └── gt/
            └── basket_gt_binary.pgm, ...
```

## Compiler et lancer

Depuis le dossier du TP (sous Linux ou WSL, avec OpenCV installé) :

```bash
mkdir build
cd build
cmake ..
make
./tp_contours
```

Le programme doit être lancé depuis le dossier `build`, car il cherche les images dans `../contours/images`. Les fichiers CSV et les captures sont créés dans `build`.

## Les fichiers

- `main.cpp` : la fenêtre, les curseurs et le traitement des 40 images
- `detecteurs.h` : les trois détecteurs
- `evaluation.h` : la comparaison avec la référence et le calcul des mesures
- `edgedetector.h` et `laplacianZC.h` : les classes du cours (chapitres 7 et 6) utilisées pour Sobel et Laplace
- `CMakeLists.txt` : pour la compilation