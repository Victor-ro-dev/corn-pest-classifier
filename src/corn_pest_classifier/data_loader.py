import sys
from pathlib import Path

# Garante a resolução dos módulos locais
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import keras
import tensorflow as tf

try:
    from corn_pest_classifier.config import BATCH_SIZE, CLASS_NAMES, DATASET_DIR, IMG_SIZE
except ImportError:
    from config import BATCH_SIZE, CLASS_NAMES, DATASET_DIR, IMG_SIZE


def build_dataset(dataset_dir: str | None = None):
    target_path = Path(dataset_dir or DATASET_DIR)

    # Filtra e carrega estritamente as classes oficiais do projeto que existem no diretório
    classes_oficiais_presentes = [
        c for c in CLASS_NAMES if (target_path / c).is_dir()
    ]
    filtro_classes = classes_oficiais_presentes if classes_oficiais_presentes else None

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        str(target_path),
        class_names=filtro_classes,
        validation_split=0.2,
        subset="both",
        seed=1,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = getattr(train_ds, "class_names", [])
    print(f"Classes ativas no dataset ({len(class_names)}): {class_names}")
    faltantes = [c for c in CLASS_NAMES if c not in class_names]
    if faltantes:
        print(
            f"[AVISO] Classes esperadas do projeto ausentes no diretório: {faltantes}\n"
            f"Classes oficiais esperadas ({len(CLASS_NAMES)}): {CLASS_NAMES}"
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

