# Analyse des CFA à partir des données InserJeunes

Ce dépôt contient le code et les éléments de reproductibilité associés à un travail de recherche sur la performance d’insertion des centres de formation d’apprentis (CFA) en France, à partir des données publiques **InserJeunes** diffusées par le ministère de l’Éducation nationale.

L’objectif de l’analyse est d’étudier les associations entre le **taux de poursuite d’études** et plusieurs indicateurs d’insertion professionnelle des CFA :
- le **taux d’emploi à 6 mois** ;
- la **valeur ajoutée d’emploi à 6 mois** ;
- le **taux d’emploi à 12 mois**.

La version révisée de l’analyse repose principalement sur :
- des modèles avec effets fixes **CFA + année** ;
- une distinction explicite entre **CFA historiques** et **nouveaux CFA** ;
- des robustesses en **modèle fractionnaire** pour les variables de taux ;
- une exploration complémentaire en **panel dynamique GMM**, fournie pour transparence mais non retenue dans l’analyse principale.

---

## Source des données

Les données utilisées proviennent du portail officiel des données de l’Éducation nationale :

**InserJeunes Apprentissage par centre de formation d’apprentis (CFA)**  
https://data.education.gouv.fr/explore/assets/fr-en-inserjeunes-cfa/

Le fichier CSV doit être téléchargé depuis cette source puis placé dans le dossier `data/` du dépôt, sous le nom :

```text
data/fr-en-inserjeunes-cfa.csv
