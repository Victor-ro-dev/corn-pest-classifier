import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 15
LEARNING_RATE = 1e-3
DATASET_DIR = os.getenv("DATASET_DIR", str(BASE_DIR / "dataset"))

# Mapeamento oficial das classes do projeto (Escopo: Cultura do Milho)
CLASS_NAMES = [
    "lagarta_cartucho",  # Spodoptera frugiperda (GBIF larva + KaraAgro AI Maize)
    "lagarta_espiga",    # Helicoverpa zea (GBIF larva + espiga)
    "vaquinha",          # Diabrotica speciosa (GBIF adulto)
    "milho_sadio",       # Classe negativa de controle (KaraAgro AI Maize "healthy")
]
NUM_CLASSES = len(CLASS_NAMES)