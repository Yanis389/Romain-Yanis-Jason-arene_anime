import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("anime_tv_clean.csv")

y = df["rating"]

X = df.drop(columns=["rating", "anime_id"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Animés pour l'entraînement :", X_train.shape[0])
print("Animés pour le test :", X_test.shape[0])
