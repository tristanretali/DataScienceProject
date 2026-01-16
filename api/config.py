from pathlib import Path
import os


PROJECT_DIR = Path(__file__).parent.parent

# Récupération du chemin menant au dataset
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Récupération du chemin menant aux modèles
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
