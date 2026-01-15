from fastapi import APIRouter, Depends, HTTPException
from schemas import PredictPayload
from config import MODELS_DIR

router = APIRouter(prefix="/api")


@router.post("/predict")
def predict(payload: PredictPayload):
    return payload.model_dump(by_alias=True)
