from enum import Enum
from pydantic import BaseModel, Field


class Transmission(str, Enum):
    Manual = "Manual"
    Automatic = "Automatic"


class FuelType(str, Enum):
    Electric = "Electric"
    Diesel = "Diesel"
    Petrol = "Petrol"


class Condition(str, Enum):
    New = "New"
    Like_New = "Like New"
    Used = "Used"


# Les '...' servent à indiquer que le champ est obligatoire
# On utilise 'alias' pour correspondre aux noms des colonnes du DataFrame et donc celles avec lesquels le modèle a été entraîné
class PredictPayload(BaseModel):
    year: int = Field(..., alias="Year")
    engine_size: float = Field(..., alias="Engine Size")
    fuel_type: FuelType = Field(..., alias="Fuel Type")
    transmission: Transmission = Field(..., alias="Transmission")
    mileage: int = Field(..., alias="Mileage")
    condition: Condition = Field(..., alias="Condition")
    model: str = Field(..., alias="Model")
