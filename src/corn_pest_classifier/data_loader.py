import shutil
import sys
from pathlib import Path

# Garante a resolução dos módulos locais
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import keras
import tensorflow as tf

try:
    from corn_pest_classifier.config import (
        BATCH_SIZE,
        CLASS_NAMES,
        DATASET_DIR,
        FLYWHEEL_CONFIRMED_DIR,
        IMG_SIZE,
    )
except ImportError:
    from config import (
        BATCH_SIZE,
        CLASS_NAMES,
        DATASET_DIR,
        FLYWHEEL_CONFIRMED_DIR,
        IMG_SIZE,
    )

VALID_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_augmentation_pipeline() -> keras.Sequential:
    """Pipeline realista de Data Augmentation simulando fotos de celular na lavoura (sol, sombra, zoom e ângulo)."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.20),
            keras.layers.RandomZoom(height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15)),
            keras.layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
            keras.layers.RandomContrast(0.20),
            keras.layers.RandomBrightness(factor=0.15, value_range=(0, 255)),
        ],
        name="Field_Photo_Augmentation",
    )


def sync_flywheel_samples(dataset_dir: Path, flywheel_dir: Path | None = None) -> int:
    """Importa fotos reais confirmadas pelos produtores no WhatsApp (Data Flywheel US03) para o dataset de treino."""
    fw_path = Path(flywheel_dir or FLYWHEEL_CONFIRMED_DIR)
    if not fw_path.exists():
        return 0

    alias_map = {
        "spodoptera_frugiperda": "lagarta_cartucho",
        "lagarta_cartucho": "lagarta_cartucho",
        "helicoverpa_zea": "lagarta_espiga",
        "lagarta_espiga": "lagarta_espiga",
        "diabrotica_speciosa": "vaquinha",
        "vaquinha": "vaquinha",
        "milho_sadio": "milho_sadio",
    }

    copied = 0
    for folder in fw_path.iterdir():
        if not folder.is_dir():
            continue
        target_class = alias_map.get(folder.name.lower())
        if not target_class:
            continue

        dest_dir = dataset_dir / target_class
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_file in folder.iterdir():
            if img_file.suffix.lower() not in VALID_IMG_EXTENSIONS:
                continue
            # Ignora arquivos simulados de teste unitário (< 200 bytes)
            if img_file.stat().st_size < 200:
                continue
            dest_file = dest_dir / f"flywheel_{img_file.name}"
            if not dest_file.exists():
                shutil.copy2(img_file, dest_file)
                copied += 1

    if copied > 0:
        print(f"[Data Flywheel US03] +{copied} novas fotos confirmadas no campo incorporadas ao treino!")
    return copied


def compute_class_weights(dataset_dir: Path, active_classes: list[str]) -> dict[int, float]:
    """Calcula pesos balanceados inversamente proporcionais à frequência de cada classe:
    w_c = Total / (Num_Classes * Count_c)
    Evita que a CNN fique enviesada para a classe majoritária.
    """
    counts: dict[int, int] = {}
    for idx, cls_name in enumerate(active_classes):
        cls_folder = dataset_dir / cls_name
        if cls_folder.exists():
            imgs = [
                f
                for f in cls_folder.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_IMG_EXTENSIONS
            ]
            counts[idx] = max(len(imgs), 1)
        else:
            counts[idx] = 1

    total_samples = sum(counts.values())
    num_classes = max(len(active_classes), 1)

    weights = {
        idx: round(total_samples / (num_classes * count), 4)
        for idx, count in counts.items()
    }
    print("Balanceamento de Classes (Class Weights):")
    for idx, cls_name in enumerate(active_classes):
        print(f"  - [{idx}] {cls_name}: {counts[idx]} fotos (peso = {weights[idx]:.3f}x)")
    return weights


def build_dataset(dataset_dir: str | None = None):
    """Constrói os pipelines `train_ds` e `val_ds` otimizados com cache, prefetch e pesos de classe."""
    target_path = Path(dataset_dir or DATASET_DIR)

    # 1. Sincroniza automaticamente fotos validadas no Data Flywheel (US03), se existirem
    sync_flywheel_samples(target_path)

    # 2. Filtra e carrega estritamente as classes oficiais do projeto que existem no diretório
    classes_oficiais_presentes = [
        c for c in CLASS_NAMES if (target_path / c).is_dir()
    ]
    filtro_classes = classes_oficiais_presentes if classes_oficiais_presentes else None

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        str(target_path),
        class_names=filtro_classes,
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = list(getattr(train_ds, "class_names", []))
    print(f"Classes ativas no dataset ({len(class_names)}): {class_names}")
    faltantes = [c for c in CLASS_NAMES if c not in class_names]
    if faltantes:
        print(
            f"[AVISO] Classes esperadas do projeto ausentes no diretório: {faltantes}\n"
            f"Classes oficiais esperadas ({len(CLASS_NAMES)}): {CLASS_NAMES}"
        )

    # 3. Calcula os pesos de balanceamento para compensar desbalanceamento de amostras
    class_weights = compute_class_weights(target_path, class_names) if class_names else {}

    # 4. Data Augmentation de Campo + Pipeline de Alta Performance (AUTOTUNE)
    data_augmentation = get_augmentation_pipeline()

    train_ds = train_ds.shuffle(buffer_size=512, seed=42)
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    # Preserva atributos úteis nos datasets retornados
    train_ds.class_names = class_names
    val_ds.class_names = class_names
    train_ds.class_weights = class_weights

    return train_ds, val_ds
