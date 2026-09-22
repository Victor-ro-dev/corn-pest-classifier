import keras
import tensorflow as tf
from config import BATCH_SIZE, DATASET_DIR, IMG_SIZE


def build_dataset():
    # Conversor -> M3D
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="both",
        seed=1,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    # Data Agument
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
    ])

    # Otimização do Pipeline
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
