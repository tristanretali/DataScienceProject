from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/api")


@router.get("/predict")
def predict():
    return {"status": "API is working"}
