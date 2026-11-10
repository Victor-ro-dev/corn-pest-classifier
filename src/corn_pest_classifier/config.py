import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# Treinamento em 2 Estágios (Transfer Learning + Fine-Tuning)
WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", "8"))
FINETUNE_EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "12"))
EPOCHS = WARMUP_EPOCHS + FINETUNE_EPOCHS

LEARNING_RATE = 1e-3
FINETUNE_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.35
EARLY_STOPPING_PATIENCE = 5
LR_PATIENCE = 2
UNFREEZE_TOP_LAYERS = 35

DATASET_DIR = os.getenv("DATASET_DIR", str(BASE_DIR / "dataset"))
# Diretório opcional do Data Flywheel (US03) com fotos confirmadas pelos produtores no WhatsApp
FLYWHEEL_CONFIRMED_DIR = os.getenv(
    "FLYWHEEL_CONFIRMED_DIR",
    str(BASE_DIR.parent / "agri_rag" / "data" / "dataset_flywheel" / "confirmados"),
)

# Mapeamento oficial das classes do projeto (Escopo: Cultura do Milho)
CLASS_NAMES = [
    "lagarta_cartucho",  # Spodoptera frugiperda (GBIF larva + KaraAgro AI Maize)
    "lagarta_espiga",    # Helicoverpa zea (GBIF larva + espiga)
    "vaquinha",          # Diabrotica speciosa (GBIF adulto)
    "milho_sadio",       # Classe negativa de controle (KaraAgro AI Maize / PlantVillage "healthy")
]
NUM_CLASSES = len(CLASS_NAMES)
