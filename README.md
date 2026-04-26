# Analyse des CFA à partir des données InserJeunes

Ce dépôt contient le code et les éléments de reproductibilité associés à un travail de recherche sur la performance d’insertion des centres de formation d’apprentis (CFA) en France, à partir des données publiques **InserJeunes** diffusées par le ministère de l’Éducation nationale.

L’objectif de l’analyse est d’étudier les associations entre le **taux de poursuite d’études** et plusieurs indicateurs d’insertion professionnelle des CFA :
- le **taux d’emploi à 6 mois** ;
- la **valeur ajoutée d’emploi à 6 mois** ;
- le **taux d’emploi à 12 mois**.

L’approche repose sur des modèles linéaires avec :
- effets fixes **région + année** ;
- effets fixes **CFA + année** ;
- tests de robustesse sur un sous-échantillon de CFA observés sur au moins trois années.

---

## Source des données

Les données utilisées proviennent du portail officiel des données de l’Éducation nationale :

**InserJeunes Apprentissage par centre de formation d’apprentis (CFA)**  
https://data.education.gouv.fr/explore/assets/fr-en-inserjeunes-cfa/

Le fichier CSV doit être téléchargé depuis cette source puis placé localement pour être utilisé par le script.

---

## Contenu du dépôt

- `analysedonnees.py` : script principal de nettoyage, estimation et production des sorties
- `README.md` : présentation du projet
- éventuellement :
  - un dossier `output/` ou équivalent contenant les tableaux et graphiques exportés
  - le fichier de données CSV téléchargé séparément depuis le portail officiel

---

## Variables mobilisées

Le script exploite les colonnes suivantes du jeu de données :

- `Année`
- `UAI`
- `Libellé`
- `Région`
- `Taux poursuite études`
- `Taux emploi 6 mois`
- `VA emploi 6 mois`
- `Taux emploi 12 mois`

Ces variables sont renommées dans le script de la manière suivante :

- `year`
- `uai`
- `label`
- `region`
- `study`
- `emp6`
- `va6`
- `emp12`

---

## Méthode

Le script réalise les étapes suivantes :

1. lecture du fichier CSV en testant plusieurs séparateurs ;
2. vérification de la présence des colonnes utiles ;
3. nettoyage et conversion des variables ;
4. constitution de l’échantillon principal ;
5. construction d’un sous-échantillon de robustesse composé des CFA observés sur au moins trois années ;
6. estimation de neuf modèles :
   - 3 modèles avec effets fixes région + année ;
   - 3 modèles avec effets fixes CFA + année ;
   - 3 modèles de robustesse sur sous-échantillon stable ;
7. export de tableaux de résultats ;
8. production de graphiques descriptifs et d’un graphique de comparaison des coefficients.

---

## Modèles estimés

Les principales spécifications économétriques sont les suivantes :

### 1. Effets fixes région + année

\[
Y_{it} = \alpha + \beta_1 \, study_{it} + \gamma_r + \delta_t + \varepsilon_{it}
\]

### 2. Effets fixes CFA + année

\[
Y_{it} = \alpha + \beta_1 \, study_{it} + \mu_i + \delta_t + \varepsilon_{it}
\]

où :
- \(Y_{it}\) représente alternativement `emp6`, `va6` ou `emp12` ;
- `study` est le taux de poursuite d’études ;
- \(\gamma_r\) désigne les effets fixes région ;
- \(\delta_t\) désigne les effets fixes année ;
- \(\mu_i\) désigne les effets fixes CFA.

Les erreurs standards sont robustes à l’hétéroscédasticité (`HC1`).

---
## Spécification exploratoire sur l’endogénéité

Une spécification exploratoire en panel dynamique GMM a également été testée afin d’examiner l’endogénéité potentielle du taux de poursuite d’études. Ces estimations sont mises à disposition dans ce dépôt pour des raisons de transparence et de reproductibilité. Elles ne sont toutefois pas retenues dans l’analyse principale de l’article, car les diagnostics économétriques usuels associés à ces modèles (notamment les tests de Hansen et d’autocorrélation d’ordre 2) se révèlent défavorables dans les spécifications considérées. Cette difficulté tient vraisemblablement à la structure courte, déséquilibrée et glissante du panel mobilisé. Les résultats principaux reposent donc sur les spécifications à effets fixes, les robustesses en modèle fractionnaire et la distinction entre CFA historiques et nouveaux CFA.

## Installation

### Dépendances Python

Le script nécessite les bibliothèques suivantes :

```bash
pip install pandas numpy matplotlib statsmodels
