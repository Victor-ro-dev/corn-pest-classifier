import sys
from pathlib import Path

# Garante a resolução dos módulos locais
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import keras
import tensorflow as tf

try:
    from corn_pest_classifier.config import BATCH_SIZE, DATASET_DIR, IMG_SIZE
except ImportError:
    from config import BATCH_SIZE, DATASET_DIR, IMG_SIZE


def build_dataset(dataset_dir: str | None = None):
    target_dir = dataset_dir or DATASET_DIR

    # Conversor -> M3D
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        target_dir,
        validation_split=0.2,
        subset="both",
        seed=1,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = getattr(train_ds, "class_names", [])
    print(f"Classes identificadas no dataset ({len(class_names)}): {class_names}")
    if len(class_names) < 2:
        print(
            f"\n[AVISO CRÍTICO] Foi encontrada apenas {len(class_names)} classe no diretório '{target_dir}'. "
            "Para treinar um classificador binário, garanta subpastas separadas "
            "(ex.: 'dataset/praga' e 'dataset/saudavel').\n"
        )

    # Data Augmentation
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
    ])

    # Otimização do Pipeline
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

