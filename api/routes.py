from fastapi import APIRouter
from schemas import PredictPayload
import config as cfg
import pickle
import pandas as pd

router = APIRouter(prefix="/api")

# Chargement du label_encoder une fois, pour éviter de le faire à chaque requête
with open(f"{cfg.DATA_DIR}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


@router.post("/{version}/predict")
def predict(version: str, payload: PredictPayload):
    if version not in ["v1", "v2"]:
        return {"error": f"La version {version} n'est pas existante. Utilisez v1 ou v2"}

    model_path = f"{cfg.MODELS_DIR}/model_{version}.pkl"
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Conversion du payload en DataFrame pour générer la prédiction
    input_data = pd.DataFrame([payload.model_dump(by_alias=True)])

    # Génération de la prédiction (label le plus probable)
    prediction = loaded_model.predict(input_data)[0]
    probabilities = loaded_model.predict_proba(input_data)[0]
    confiance = probabilities.max()

    # On récupère le label correspondant
    label = label_encoder.inverse_transform([prediction])[0]

    return {"prediction": label, "confiance": float(confiance)}
