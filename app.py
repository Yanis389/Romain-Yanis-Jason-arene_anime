import streamlit as st
import pandas as pd
import random
import joblib


@st.cache_resource
def charger_modele():
    return joblib.load("champion.joblib").get("modele")


model = charger_modele()
colonnes = list(model.feature_names_in_)
genres = [c for c in colonnes if c not in ("episodes", "members")]

st.title("Prédiction de la note d'un animé")
st.write("Choisis les caractéristiques d'un animé TV pour estimer sa note moyenne (sur 10).")

if "genres" not in st.session_state:
    st.session_state.genres = []


def genres_au_hasard():
    st.session_state.genres = random.sample(genres, 3)


st.button("Genres au hasard", on_click=genres_au_hasard)

choix_genres = st.multiselect(
    "Genres (3 maximum)",
    genres,
    max_selections=3,
    key="genres",
)

membres = st.number_input("Nombre de membres (popularité)", min_value=0, value=10000, step=1000)
episodes = st.number_input("Nombre d'épisodes", min_value=1, value=12, step=1)

if st.button("Prédire la note"):
    ligne = {c: 0 for c in colonnes}
    ligne["members"] = membres
    ligne["episodes"] = episodes
    for g in choix_genres:
        ligne[g] = 1

    entree = pd.DataFrame([ligne])[colonnes]
    note = model.predict(entree)[0]

    st.success(f"Note prédite : {note:.2f} / 10")
    if note >= 7:
        st.write("Plutôt bien noté.")
    elif note < 5:
        st.write("Plutôt mal noté.")
