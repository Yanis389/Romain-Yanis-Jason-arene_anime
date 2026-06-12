import joblib

def sauvegarder_modele(modele, scaler, chemin="modele.joblib"):
    artefact = {
        "modele": modele,
        "scaler": scaler
    }
    joblib.dump(artefact, chemin)
    print(f" Modèle sauvegardé dans {chemin}")
    sauvegarder_modele(xgb, None)
    