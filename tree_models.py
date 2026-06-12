import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("anime_tv_clean.csv")

y = df["rating"]

X = df.drop(columns=["rating", "anime_id"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Animés pour l'entraînement :", X_train.shape[0])
print("Animés pour le test :", X_test.shape[0])


def evaluer(model, nom):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print(nom)
    print("  MAE  :", round(mae, 3))
    print("  RMSE :", round(rmse, 3))
    print("  R2   :", round(r2, 3))
    return mae, rmse, r2


arbre = DecisionTreeRegressor(random_state=42)
evaluer(arbre, "Decision Tree")

foret = RandomForestRegressor(n_estimators=100, random_state=42)
evaluer(foret, "Random Forest")
