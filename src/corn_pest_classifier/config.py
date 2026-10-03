import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 15
LEARNING_RATE = 1e-3
DATASET_DIR = os.getenv("DATASET_DIR", str(BASE_DIR / "dataset"))
NUM_CLASSES = 2