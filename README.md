# 🎌 Prédiction de la note des animés TV

##  Équipe
- Romain Pintre — Modèles linéaires
- Jason Dahmoun — Modèles à arbres
- Yanis Helali — Modèles avancés & sélection finale

---

#  Objectif du projet

L’objectif est de prédire la note moyenne d’un animé TV à partir de ses caractéristiques, en comparant plusieurs modèles de machine learning et en sélectionnant le meilleur.

---

#  1. Préparation des données

Le dataset a été nettoyé :

- Suppression des doublons  
- Filtrage des animés TV  
- Gestion des valeurs manquantes  
- Encodage des genres (One-Hot Encoding)

 Dataset final : `anime_tv_clean.csv`

---

#  2. Baseline – Modèles linéaires

Modèles testés :

- Linear Regression  
- Ridge  
- Lasso  

Résultats :

- R² ≈ 0.42  

 Conclusion :

- Base de référence  
- Relations partiellement linéaires  

---

#  3. Modèles à arbres

Modèles testés :

- Decision Tree  
- Random Forest  
- Extra Trees  

Résultats :

- Random Forest → R² ≈ 0.55  

 Conclusion :

- Meilleure gestion des interactions  
- Amélioration nette par rapport à la baseline  

---

#  4. Modèles avancés

Modèles testés :

- XGBoost  
- LightGBM  
- MLP  

Résultats (comparaison avec R²) :

- LightGBM → 0.58 
- XGBoost → 0.56  
- MLP → 0.40  

 Conclusion :

- Le boosting est la meilleure approche  
- Les relations sont non linéaires  

---

#  5. Sélection du modèle final

Une arène finale est mise en place avec :

- Validation croisée (cross-validation)
- Optimisation des hyperparamètres (GridSearch)

Critère choisi :

 **RMSE (Root Mean Squared Error)**

Pourquoi ?

- mesure l’erreur réelle  
- pénalise les grosses erreurs  
- plus adapté à la sélection finale  

---

##  Résultat final

- Champion : **LightGBM**
- RMSE avant tuning : 0.567
- RMSE après tuning : 0.551 
- Gain : ~0.016

 Le tuning améliore légèrement les performances

---

#  6. Analyse

- Les données présentent des relations non linéaires  
- Les modèles de boosting capturent mieux ces relations  
- Le réseau de neurones n’est pas adapté aux données tabulaires  

---

#  7. Déploiement

Le modèle final est sauvegardé avec joblib :

- modèle entraîné  
- features  

 utilisable dans une application  

---


# Application web

Une application Streamlit a été développée pour tester le modèle :
Lien de la web app :
https://arene-des-algos-helali-yanis-ft4phwnt8hfkccdawhuy8i.streamlit.app/
Fonctionnalités :

- sélection des genres
- choix du nombre d’épisodes
- prédiction de la note
- affichage du résultat en temps réel

---

#  Conclusion finale

- Les modèles linéaires donnent une base solide  
- Les arbres améliorent nettement les performances  
- Le boosting (LightGBM) est le plus performant  

 Le modèle final est sélectionné avec RMSE car il reflète mieux l’erreur réelle  

 **LightGBM est le modèle retenu**

---

#  Améliorations possibles

- Ajouter des variables (studio, durée, etc.)
- Feature engineering  
- Plus de tuning  
- Ensemble de modèles  

