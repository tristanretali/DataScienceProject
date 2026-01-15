from fastapi import APIRouter
from schemas import PredictPayload
from config import MODELS_DIR

router = APIRouter(prefix="/api")


@router.post("/{version}/predict")
def predict(version: str, payload: PredictPayload):
    if version not in ["v1", "v2"]:
        return {"error": f"La version {version} n'est pas existante. Utilisez v1 ou v2"}

    # TODO Charger le modèle quand entraîné et faire prédiction puis retourner le résultat
    model_path = f"{MODELS_DIR}/model_{version}.pkl"
    return payload.model_dump(by_alias=True)
