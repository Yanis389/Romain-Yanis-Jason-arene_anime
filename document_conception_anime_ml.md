# Document de conception – Projet ML Anime Rating Prediction

## 1. Problème et cible

### Objectif
Le projet vise à prédire la **note moyenne d’un animé TV** à partir de ses caractéristiques descriptives (genres, nombre d’épisodes, etc.).

### Type de problème
- Type : **Régression**
- Cible : variable continue → `rating` (note moyenne)

### Décision métier simulée
Le modèle pourrait aider à :
- estimer le succès critique d’un animé avant sa sortie
- orienter la production (genres, formats)
- comprendre les facteurs influençant la qualité perçue

---

## 2. Dataset

### Source
- Anime Recommendations Database (Kaggle)

### Description
- Nombre d’observations : ~7 000 à 12 000 animés (selon filtrage TV)
- Nombre de variables :
  - genres (multi-label)
  - episodes
  - type (filtré ici sur TV)
  - rating (cible)

### Type de cible
- Régression (valeurs continues entre ~1 et 10)

###  Distribution de la cible
- distribution centrée autour de 6–8
- peu de valeurs extrêmes

=> Conséquences :
- dataset relativement équilibré
- pas besoin de stratification (régression)

### 🔀 Split des données
- Train : 80%
- Test : 20%
- Validation : optionnel (ou cross-validation)

---

## 3. Les algos candidats (L’Arène)

Nous comparons plusieurs modèles sur le même dataset :

### Linear Regression
- baseline simple
- interprétable
- permet de vérifier la linéarité des relations

### Ridge Regression
- version régularisée de la régression linéaire
- réduit le surapprentissage potentiel
- utile en présence de nombreuses variables (genres)

### Lasso Regression
- régularisation L1
- sélection automatique de variables
- permet d’identifier les features les plus importantes

### Decision Tree Regressor
- modèle non linéaire basé sur des règles de décision successives
- facile à interpréter visuellement
- peut surapprendre si non contraint (profondeur, min samples)
### Random Forest Regressor
- ensemble d’arbres de décision entraînés sur des sous-échantillons
- réduit fortement le surapprentissage par agrégation (bagging)
- robuste et performant par défaut sur données tabulaires
### Extra Trees Regressor
- variante de Random Forest avec splits encore plus aléatoires
- réduit la variance au prix d’un léger biais supplémentaire
- souvent plus rapide à entraîner
### XGBoost
- boosting de modèles faibles (arbres) entraînés séquentiellement
- optimise une fonction de perte via gradient boosting
- très performant mais sensible aux hyperparamètres
### LightGBM (si autorisé)
- implémentation optimisée du gradient boosting (histogrammes)
- très rapide sur grands datasets
- efficace mais peut surapprendre sur petits jeux de données
### MLP Regressor
- réseau de neurones feedforward (perceptron multicouche)
- capable de modéliser des relations très non linéaires
- nécessite normalisation et réglage fin des hyperparamètres
---

## 4. Plan d’évaluation

### Métriques utilisées

#### MAE (Mean Absolute Error)
- erreur moyenne absolue
- interprétable directement (écart en points de note)

#### RMSE (Root Mean Squared Error)
- pénalise davantage les grosses erreurs
- utile pour détecter les mauvaises prédictions

#### R² (coefficient de détermination)
- proportion de variance expliquée par le modèle
- 0 = modèle nul, 1 = parfait

---

### Choix des métriques

- MAE : interprétation métier directe
- RMSE : contrôle des erreurs extrêmes
- R² : qualité globale du modèle

---

### Protocole d’évaluation

- même split train/test pour tous les modèles
- random_state fixé pour reproductibilité
- comparaison directe des scores

Option avancée :
- cross-validation k-fold pour plus de robustesse

---

## 6. 👥 Répartition des rôles

### 👤 Romain – Modèles linéaires + EDA
- analyse exploratoire des données
- Linear / Ridge / Lasso
- interprétation des coefficients

### 👤 Jason – Modèles arbres
- Decision Tree
- Random Forest
- Extra Trees
- feature importance

### 👤 Yanis – Boosting + optimisation
- XGBoost / Gradient Boosting
- tuning hyperparamètres
- optimisation des performances

---

## 7. Questions ouvertes

Certaines décisions restent ouvertes et seront prises après exploration :

- Faut-il conserver toutes les variables ou réduire la dimension ?
- Une PCA est-elle pertinente ou inutile ici ?
- Faut-il inclure les films/OVA à l’avenir ?
- Les genres rares doivent-ils être regroupés ?
- Les features numériques doivent-elles être normalisées ?

---

## 🧠 Conclusion

Ce projet suit une démarche CRISP-DM :

1. Compréhension du problème
2. Analyse du dataset
3. Modélisation comparative
4. Évaluation rigoureuse

L’objectif est de construire un modèle interprétable et performant pour prédire la note moyenne des animés TV.
