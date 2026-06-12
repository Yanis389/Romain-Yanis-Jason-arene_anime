import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("anime_tv_clean.csv")

y = df["rating"]

X = df.drop(columns=["rating", "anime_id"])

print("Nombre d'animés :", X.shape[0])
print("Nombre de variables :", X.shape[1])
print("Variables utilisées :", X.columns.tolist())
