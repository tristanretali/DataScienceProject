from typing import Any, Dict, List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from routes import router


app = FastAPI(title="Prédiction pour  une entreprise vendant des voitures d'occasion")


app.include_router(router)
