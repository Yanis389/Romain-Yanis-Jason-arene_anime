import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Chargement
anime = pd.read_csv("anime.csv")

# ------------------
# Nettoyage de base
# ------------------

anime = anime.drop_duplicates()

# Garder uniquement les animés TV
anime = anime[anime["type"] == "TV"]

# Gestion des notes manquantes
anime["rating"] = anime["rating"].fillna(anime["rating"].median())

# Episodes
anime["episodes"] = anime["episodes"].replace("Unknown", pd.NA)
anime["episodes"] = pd.to_numeric(anime["episodes"], errors="coerce")
anime["episodes"] = anime["episodes"].fillna(anime["episodes"].median())

# Suppression lignes sans genre
anime = anime.dropna(subset=["genre"])

# ------------------
# Encodage genres
# ------------------

genres = anime["genre"].str.split(", ")

mlb = MultiLabelBinarizer()

genre_encoded = pd.DataFrame(
    mlb.fit_transform(genres),
    columns=mlb.classes_,
    index=anime.index
)

anime = pd.concat(
    [anime.drop(columns=["genre", "name", "type"]),
     genre_encoded],
    axis=1
)

# ------------------
# Sauvegarde
# ------------------

anime.to_csv("anime_tv_clean.csv", index=False)

print(anime.shape)
print(anime.head())