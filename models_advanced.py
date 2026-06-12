# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# =========================
# Chargement dataset
# =========================
df = pd.read_csv("anime_tv_clean.csv")

# Fix LightGBM (espaces dans colonnes)
df.columns = df.columns.str.replace(" ", "_")

print(" Dataset chargé")
print(df.shape)


# =========================
# Features / Target
# =========================
X = df.drop("rating", axis=1)
y = df["rating"]


# =========================
# Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# Fonction d'évaluation
# =========================
def evaluate_model(name, model, X_test, y_test):
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print(f"\n{name}")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"R²   : {r2:.3f}")

    return rmse, mae, r2


# =========================
# XGBoost
# =========================
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_rmse, xgb_mae, xgb_r2 = evaluate_model(
    "XGBoost", xgb, X_test, y_test
)


# =========================
# Importance des variables
# =========================
importance = xgb.feature_importances_

indices = np.argsort(importance)[-10:]

plt.figure(figsize=(8,5))
plt.barh(range(len(indices)), importance[indices])
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title("Top 10 des variables importantes (XGBoost)")
plt.show()


# =========================
# LightGBM
# =========================

lgbm = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    random_state=42,
    verbose=-1  
)


lgbm.fit(X_train, y_train)

lgbm_rmse, lgbm_mae, lgbm_r2 = evaluate_model(
    "LightGBM", lgbm, X_test, y_test
)


# =========================
# MLP (Neural Network)
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    max_iter=300,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

mlp_rmse, mlp_mae, mlp_r2 = evaluate_model(
    "MLP", mlp, X_test_scaled, y_test
)


# =========================
# Leaderboard
# =========================
results = pd.DataFrame({
    "Model": ["XGBoost", "LightGBM", "MLP"],
    "RMSE": [xgb_rmse, lgbm_rmse, mlp_rmse],
    "MAE": [xgb_mae, lgbm_mae, mlp_mae],
    "R2": [xgb_r2, lgbm_r2, mlp_r2]
})

results = results.sort_values(by="R2", ascending=False)

print("\n=== Leaderboard ===")
print(results)


# =========================
# Champion
# =========================
champion_name = results.iloc[0]["Model"]
print("\n🏆 Champion :", champion_name)


# =========================
# Sauvegarde modèle
# =========================
def sauvegarder_modele(modele, scaler, chemin="modele.joblib"):
    artefact = {
        "modele": modele,
        "scaler": scaler,
        "features": list(X.columns)
    }
    joblib.dump(artefact, chemin)
    print(f"\n✅ Modèle sauvegardé dans {chemin}")


# Sauvegarde du meilleur modèle
if champion_name == "XGBoost":
    sauvegarder_modele(xgb, None)

elif champion_name == "LightGBM":
    sauvegarder_modele(lgbm, None)

else:
    sauvegarder_modele(mlp, scaler)