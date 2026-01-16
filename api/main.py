from fastapi import FastAPI
from routes import router
from ML_pipeline import train_models
import os
import config as cfg

app = FastAPI(title="Prédiction pour  une entreprise vendant des voitures d'occasion")


@app.on_event("startup")
def startup():
    """Entraîne les modèles au démarrage s'ils n'existent pas"""

    model_v1 = os.path.join(cfg.MODELS_DIR, "model_v1.pkl")
    model_v2 = os.path.join(cfg.MODELS_DIR, "model_v2.pkl")

    if not (os.path.exists(model_v1) and os.path.exists(model_v2)):
        print("Entraînement des modèles en cours")
        train_models()
        print("Modèles entraînés")
    else:
        print("Les modèles existent déjà")


app.include_router(router)
