import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ----------------------
# Chargement des données
# ----------------------
df = pd.read_csv("anime_tv_clean.csv")

# ----------------------
# Séparation features / target
# ----------------------
X = df.drop(columns=["rating"])
y = df["rating"]

# ----------------------
# Split train / test
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------
# Fonction d'évaluation
# ----------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

# ----------------------
# Modèles simples
# ----------------------
models_results = []

models_results.append(
    evaluate_model(
        "Linear Regression",
        LinearRegression(),
        X_train, X_test, y_train, y_test
    )
)

models_results.append(
    evaluate_model(
        "Ridge Regression",
        Ridge(alpha=1.0),
        X_train, X_test, y_train, y_test
    )
)

models_results.append(
    evaluate_model(
        "Lasso Regression",
        Lasso(alpha=0.01, max_iter=10000),
        X_train, X_test, y_train, y_test
    )
)

# ----------------------
# Tableau final
# ----------------------
results_df = pd.DataFrame(models_results)
print("\n===== COMPARAISON =====")
print(results_df)


# ----------------------
# Top coefficients (Linear Regression)
# ----------------------
print("\nTop coefficients (Linear Regression) :")
lr = LinearRegression().fit(X_train, y_train)

coef = pd.DataFrame({
    "feature": X.columns,
    "coef": lr.coef_
}).sort_values(by="coef", key=abs, ascending=False)

print(coef.head(10))