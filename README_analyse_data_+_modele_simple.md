# Rapport – Prédiction de la note des animés TV

## Objectif du projet

L’objectif est de prédire la **note moyenne d’un animé TV** à partir de ses caractéristiques (genres principalement), afin de comparer plusieurs modèles de régression simples :

- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression

---

# 1. Analyse exploratoire des données (EDA)

## 1.1 Structure des données

Le dataset contient des animés de type **TV uniquement**, avec les variables suivantes :

- Genres (one-hot encoded)
- Nombre d’épisodes
- Features catégorielles transformées

La variable cible est :

- **rating** (note moyenne de l’animé)

---

## 1.2 Distribution des notes

Les notes des animés sont généralement :

- centrées autour de valeurs moyennes (~6 à 8)
- avec peu d’extrêmes

=> Cela rend la prédiction difficile car :
- faible variance de la cible
- peu de “signal fort” dans les features

---

## 1.3 Analyse des variables

Les genres sont les variables les plus importantes :

- Chaque animé peut appartenir à plusieurs genres
- Les relations sont indirectes (pas de causalité simple)

=> Exemple :
- “Josei”, “Seinen”, “Sports” influencent positivement la note
- “Kids” et “Vampire” sont plutôt négatifs

---

## 1.4 Conclusion EDA

- Les données sont propres et exploitables
- Les variables explicatives sont principalement catégorielles
- Les relations semblent **faibles à modérées et non strictement linéaires**

---

# 2. Les relations sont-elles linéaires ?

## Réponse courte : PARTIELLEMENT

### Arguments :

### Oui (un peu de linéarité)
- Les modèles linéaires atteignent déjà :
  - R² ≈ 0.42
- Certaines variables ont un effet constant :
  - ex : “Seinen” augmente la note
  - ex : “Kids” diminue la note

---

### Non (limite importante)

- Le R² reste faible (~0.42)
- Les genres interagissent entre eux (non capté par un modèle linéaire)
- Relations complexes :
  - un anime “Fantasy + Action + Seinen” ≠ somme simple des effets

---

### Conclusion

> Les relations sont partiellement linéaires, mais une grande partie de la structure des données est non linéaire.

---

# 3. Score de base à battre (baseline)

Le **modèle de référence** est :

### => Linear Regression

Résultats :

- MAE = 0.5006
- RMSE = 0.6628
- R² = 0.4237

---

## Interprétation

Cela signifie que :

- en moyenne, l’erreur de prédiction est d’environ ±0.5 point de note
- le modèle explique environ 42% de la variance

---

## Score de base à battre

- MAE < 0.50
- RMSE < 0.66
- R² > 0.42

---

# 4. Résultats des modèles linéaires

| Modèle | MAE | RMSE | R² |
|--------|------|------|------|
| Linear Regression | 0.5006 | 0.6628 | 0.4237 |
| Ridge Regression | 0.5001 | 0.6634 | 0.4226 |
| Lasso Regression | 0.5208 | 0.6900 | 0.3754 |

---

## Conclusion modèles

Les modèles linéaires obtiennent des performances proches, ce qui indique que la structure du problème ne bénéficie pas fortement de la régularisation.

---

# 5. Analyse des variables importantes

## Impact positif :
- Josei (+0.47)
- Seinen (+0.33)
- Shounen (+0.27)
- Sports (+0.26)
- Drama (+0.24)
- Mystery (+0.23)

## Impact négatif :
- Kids (-0.31)
- Vampire (-0.29)

---

# 6. Conclusion générale

- Les modèles linéaires donnent une baseline solide (R² ≈ 0.42)
- Les relations sont partiellement linéaires mais limitées
- Les genres expliquent une partie de la note, mais pas tout

