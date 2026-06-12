import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# =========================
# 1. METRIQUES
# =========================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# scoring sklearn (car cross_val_score maximise -> on met négatif)
def rmse_scorer(model, X, y):
    return -cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=5
    ).mean()


# ----------------------
# Chargement des données
# ----------------------
df = pd.read_csv("anime_tv_clean.csv")

# ----------------------
# Séparation features / target
# ----------------------
X = df.drop(columns=["rating"])
y = df["rating"]


# =========================
# 2. MODELES
# =========================
artefact = {
    "artefact_1": joblib.load("modele-LinReg.joblib"),
    "artefact_2": joblib.load("model_tree.joblib"),
    "artefact_3": joblib.load("modele-LightGBM.joblib"),
}

models = {
    "model_1": artefact["artefact_1"].get("modele"),
    "model_2": artefact["artefact_2"],
    "model_3": artefact["artefact_3"].get("modele"),
}


# =========================
# 3. SPLIT FINAL HOLDOUT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# 4. ARÈNE (CROSS VALIDATION)
# =========================
results = []

for name, model in models.items():

    score = rmse_scorer(model, X_train, y_train)

    results.append({
        "model": name,
        "cv_rmse": score
    })

leaderboard = pd.DataFrame(results).sort_values(by="cv_rmse")

print("\n LEADERBOARD (CV)")
print(leaderboard)



# =========================
# 5. CHAMPION BRUT
# =========================
champion_name = leaderboard.iloc[0]["model"]
champion_model = models[champion_name]

print(f"\n Champion brut : {champion_name}")

# =========================
# 6. EVALUATION CHAMPION BRUT
# =========================
champion_model.fit(X_train, y_train)
y_pred = champion_model.predict(X_test)

rmse_brut = rmse(y_test, y_pred)
mae_brut = mean_absolute_error(y_test, y_pred)

print("\n CHAMPION BRUT (TEST)")
print(f"RMSE : {rmse_brut:.4f}")
print(f"MAE  : {mae_brut:.4f}")


# =========================
# 7. GRIDSEARCH (TUNING CHAMPION)
# =========================
print("\n TUNING DU CHAMPION (GridSearchCV)")

#  ADAPTE selon ton modèle réel
param_grid = {}


if champion_name == "model_1":
    param_grid = {
        "fit_intercept": [True, False],
        "positive": [False]
    }


elif champion_name == "model_2":
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

elif champion_name == "model_3":
    param_grid = {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [ 3, 5, 7],
        "num_leaves": [15, 31, 63], 
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_samples": [10, 20, 50]
    }


grid = GridSearchCV(
    estimator=champion_model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\n Meilleurs hyperparamètres :")
print(grid.best_params_)


# =========================
# 8. EVALUATION CHAMPION TUNÉ
# =========================
y_pred_tuned = best_model.predict(X_test)

rmse_tuned = rmse(y_test, y_pred_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)

print("\n CHAMPION TUNÉ (TEST)")
print(f"RMSE : {rmse_tuned:.4f}")
print(f"MAE  : {mae_tuned:.4f}")


# =========================
# 9. COMPARAISON
# =========================
gain_rmse = rmse_brut - rmse_tuned

print("\n GAIN DU TUNING")
print(f"Gain RMSE : {gain_rmse:.4f}")


# =========================
# 10. ANALYSE RESIDUS
# =========================
residuals = y_test - y_pred_tuned

print("\n ANALYSE CHAMPION TUNÉ")
print(f"Moyenne résidus : {np.mean(residuals):.4f}")
print(f"Std résidus     : {np.std(residuals):.4f}")



# =========================
# 9. SAUVEGARDE
# =========================


joblib.dump(champion_model, "champion.joblib")
print("\n Champion sauvegardé en joblib (.joblib)")
