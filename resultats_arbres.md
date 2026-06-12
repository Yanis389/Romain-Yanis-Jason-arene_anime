## Objectif du projet

L’objectif est de prédire la note moyenne d’un animé TV à partir de ses caractéristiques, en testant trois modèles à base d’arbres et en les comparant entre eux:

- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor

---

# 1. Préparation des données

## 1.1 Structure des données

On part du dataset nettoyé 

Variables utilisées :

- Genres 
- Nombre d’épisodes
- Nombre de membres 

La variable cible est :

- rating (note moyenne de l’animé)

---

## 1.2 Découpage train / test

- 80 % entraînement / 20 % test
- `random_state=42` pour des résultats reproductibles
- Les 3 modèles sont entraînés et testés sur le même découpage => comparaison équitable

---

# 2. Comparaison des modèles à arbres

| Modèle | MAE | RMSE | R² |
|--------|------|------|------|
| Random Forest | 0.430 | 0.583 | 0.555 |
| Extra Trees | 0.454 | 0.640 | 0.462 |
| Decision Tree | 0.571 | 0.780 | 0.202 |

=> Classement : Random Forest > Extra Trees > Decision Tree

Observations :

- Un seul arbre (Decision Tree) est faible (R² = 0.20) car il sur-apprend les données d’entraînement
- Les modèles d’ensemble (qui moyennent 100 arbres) font beaucoup mieux
- Sur ce dataset, la Random Forest est devant les Extra Trees : son hasard plus limité capte mieux la structure
---

# 3. Importance des variables

Importance donnée par la Random Forest (le total fait 1) :

| Variable | Importance |
|----------|------------|
| members | 0.608 |
| episodes | 0.134 |
| Comedy | 0.019 |
| Fantasy | 0.017 |
| Adventure | 0.015 |
| Slice of Life | 0.015 |

---

# 4. Conclusion générale

- Les modèles à arbres d’ensemble battent la baseline linéaire 
- La Random Forest est le meilleur des trois modèles testés
- Un arbre unique est insuffisant 
- La popularité est la variable la plus importante, devant le nombre d’épisodes
- Les genres jouent un rôle mineur dans la prédiction de la note
