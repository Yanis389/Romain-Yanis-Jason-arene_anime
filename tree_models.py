import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("anime_tv_clean.csv")

y = df["rating"]

X = df.drop(columns=["rating", "anime_id"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Animés pour l'entraînement :", X_train.shape[0])
print("Animés pour le test :", X_test.shape[0])


def evaluer(model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    return mae, rmse, r2


arbre = DecisionTreeRegressor(random_state=42)
foret = RandomForestRegressor(n_estimators=100, random_state=42)
extra = ExtraTreesRegressor(n_estimators=100, random_state=42)

resultats = [
    ("Decision Tree", *evaluer(arbre)),
    ("Random Forest", *evaluer(foret)),
    ("Extra Trees", *evaluer(extra)),
]

comparaison = pd.DataFrame(resultats, columns=["Modele", "MAE", "RMSE", "R2"])
comparaison = comparaison.sort_values("R2", ascending=False)

print()
print("Comparaison des modeles a arbres :")
print(comparaison.round(3).to_string(index=False))

importances = pd.Series(foret.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print()
print("Importance des variables (Random Forest) :")
print(importances.head(10).round(3).to_string())
