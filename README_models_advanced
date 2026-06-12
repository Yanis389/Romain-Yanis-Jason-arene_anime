# Rapport – Modèles avancés (Boosting & Réseaux de neurones)

## Auteur
- Yanis Helali (Personne 3 – Modèles avancés)

---

## Objectif du projet

L’objectif de cette partie est de tester des modèles de **machine learning avancés** afin d’améliorer les performances obtenues avec :

- les modèles linéaires (baseline)
- les modèles à arbres (Random Forest)

Les modèles testés sont :

- XGBoost
- LightGBM
- MLP Regressor (réseau de neurones)

---

# 1. Préparation des données

## 1.1 Dataset

Le dataset utilisé est une version **nettoyée et encodée** commune au groupe :

- uniquement des animés de type TV
- genres transformés en variables binaires
- données prêtes pour le machine learning

Variables utilisées :

- Nombre d’épisodes
- Genres encodés (Action, Comedy, Drama, etc.)

Variable cible :

- **rating** (note moyenne de l’animé)

---

## 1.2 Découpage train / test

- 80% entraînement / 20% test
- `random_state = 42` pour reproductibilité
- même split pour tous les modèles

=> permet une comparaison équitable

---

# 2. Modèles testés

## 2.1 XGBoost

XGBoost est un modèle basé sur le **Gradient Boosting**, utilisant plusieurs arbres de décision pour corriger les erreurs progressivement.

### Résultats :

- RMSE : 0.580  
- MAE : 0.426  
- R² : 0.559  

---

## 2.2 LightGBM

LightGBM est également un algorithme de **boosting**, optimisé pour la vitesse et les performances sur les données tabulaires.

### Résultats :

- RMSE : 0.567  
- MAE : 0.415  
- R² : 0.578  

---

## 2.3 MLP (Réseau de neurones)

Le MLP (Perceptron Multicouche) est un modèle de **deep learning simple**, basé sur des couches de neurones.

### Résultats :

- RMSE : 0.672  
- MAE : 0.495  
- R² : 0.407  

---

# 3. Comparaison des modèles

| Modèle    | RMSE  | MAE   | R²    |
|-----------|-------|-------|-------|
| LightGBM  | 0.567 | 0.415 | 0.578 |
| XGBoost   | 0.580 | 0.426 | 0.559 |
| MLP       | 0.672 | 0.495 | 0.407 |

---

## Classement final :

👉 LightGBM > XGBoost > MLP

---

# 4. Analyse des résultats

## 4.1 Boosting vs autres modèles

- Les modèles de boosting (XGBoost, LightGBM) surpassent :
  - les modèles linéaires (R² ≈ 0.42)
  - les arbres simples

- Ils capturent mieux les **relations non linéaires** entre les variables

---

## 4.2 Pourquoi LightGBM gagne ?

- Dataset relativement **petit (~3000 lignes)**
- Variables simples (genres + épisodes)
- LightGBM est optimisé pour ce type de problème

---

## 4.3 Réseau de neurones (MLP)

- Performances moins bonnes (R² ≈ 0.40)
- Moins adapté aux données tabulaires
- Besoin de plus de données pour être efficace

---

# 5. Importance des variables

Grâce à XGBoost, il est possible d’analyser l’importance des variables :

- Les genres sont les variables les plus influentes
- Certaines catégories impactent fortement la note
- Les interactions entre genres jouent un rôle important

---

# 6. Conclusion générale

- Les modèles avancés améliorent significativement la performance
- Le boosting est la meilleure approche sur ce dataset
- LightGBM est le modèle champion
- Les réseaux de neurones ne sont pas adaptés ici

---

# 7. Sauvegarde du modèle

Le modèle final est sauvegardé avec `joblib` :

- modèle entraîné
- scaler (si nécessaire)
- liste des features

Cela permet de :

- réutiliser le modèle sans réentraîner
- l’intégrer dans une WebApp

---

# 8. Réponse aux questions du projet

## Peut-on dépasser les Random Forest ?

✅ Oui, grâce aux modèles de boosting  
LightGBM et XGBoost obtiennent de meilleurs résultats

---

## Le boosting est-il plus performant ?

✅ Oui  
Il permet de mieux capturer la complexité des données

---

## Le réseau de neurones apporte-t-il un gain ?

❌ Non dans ce cas  
Le MLP est moins performant que les modèles de boosting

---

# Conclusion finale

> Le modèle LightGBM est le meilleur choix pour ce problème, car il offre le meilleur compromis entre performance, rapidité et capacité à modéliser des relations complexes.