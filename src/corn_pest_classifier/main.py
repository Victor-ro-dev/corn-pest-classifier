import json
import sys
from pathlib import Path

# Garante compatibilidade de imports em qualquer ambiente (local, Colab, CLI)
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import tensorflow as tf

try:
    from corn_pest_classifier.config import (
        DROPOUT_RATE,
        EARLY_STOPPING_PATIENCE,
        FINETUNE_EPOCHS,
        FINETUNE_LEARNING_RATE,
        LEARNING_RATE,
        LR_PATIENCE,
        NUM_CLASSES,
        UNFREEZE_TOP_LAYERS,
        WARMUP_EPOCHS,
        WEIGHT_DECAY,
    )
    from corn_pest_classifier.data_loader import build_dataset
    from corn_pest_classifier.mobilenet_custom import (
        build_custom_mobilenet,
        unfreeze_for_finetuning,
    )
    from corn_pest_classifier.trainer import CustomTrainer
except ImportError:
    from config import (
        DROPOUT_RATE,
        EARLY_STOPPING_PATIENCE,
        FINETUNE_EPOCHS,
        FINETUNE_LEARNING_RATE,
        LEARNING_RATE,
        LR_PATIENCE,
        NUM_CLASSES,
        UNFREEZE_TOP_LAYERS,
        WARMUP_EPOCHS,
        WEIGHT_DECAY,
    )
    from data_loader import build_dataset
    from mobilenet_custom import build_custom_mobilenet, unfreeze_for_finetuning
    from trainer import CustomTrainer


def export_quantized_tflite(
    model: tf.keras.Model,
    output_path: str = "milho_pragas_model.tflite",
    quantize: bool = True,
) -> Path:
    """Converte o modelo Keras validado para TFLite com quantização dinâmica de pesos."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    out_file = Path(output_path)
    out_file.write_bytes(tflite_model)
    size_kb = out_file.stat().st_size / 1024.0
    print(f"✓ Modelo TFLite exportado com sucesso: '{out_file}' ({size_kb:.1f} KB)")
    return out_file


def main():
    print("1. Construindo o Dataset com Balanceamento de Classes e Data Augmentation de Campo...")
    train_ds, val_ds = build_dataset()

    class_names = list(getattr(train_ds, "class_names", []))
    class_weights = getattr(train_ds, "class_weights", {})
    qtd_classes = len(class_names) if class_names else NUM_CLASSES

    print(f"\n2. Instanciando a MobileNetV2 Híbrida para {qtd_classes} classes...")
    model = build_custom_mobilenet(
        num_classes=qtd_classes,
        dropout_rate=DROPOUT_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    model.summary()

    trainer = CustomTrainer(
        model=model,
        learning_rate=LEARNING_RATE,
        class_weights=class_weights,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        lr_patience=LR_PATIENCE,
    )

    print(f"\n3A. [FASE 1 - WARMUP DA CABEÇA] Treinando camadas customizadas ({WARMUP_EPOCHS} épocas, LR={LEARNING_RATE})...")
    trainer.fit(train_ds, val_ds, epochs=WARMUP_EPOCHS, stage_name="Fase 1: Warmup")
    warmup_epochs_ran = len(trainer.history["train_loss"])

    unfrozen = unfreeze_for_finetuning(model, unfreeze_top_layers=UNFREEZE_TOP_LAYERS)
    if unfrozen > 0 and FINETUNE_EPOCHS > 0:
        print(
            f"\n3B. [FASE 2 - FINE-TUNING] Ajustando {unfrozen} camadas superiores "
            f"({FINETUNE_EPOCHS} épocas, LR={FINETUNE_LEARNING_RATE})..."
        )
        trainer.set_learning_rate(FINETUNE_LEARNING_RATE)
        trainer.fit(train_ds, val_ds, epochs=FINETUNE_EPOCHS, stage_name="Fase 2: Fine-Tuning")

    print("\n4. Gerando Gráficos de Evolução e Matriz de Confusão por Classe...")
    trainer.plot_training_curves("training_curves.png", warmup_epochs_ran=warmup_epochs_ran)
    eval_report = trainer.evaluate_per_class(val_ds, class_names, "confusion_matrix.png")

    print("\n5. Exportando Modelo Quantizado (.tflite) e Metadados para Deploy (WhatsApp / AgriRAG)...")
    tflite_file = export_quantized_tflite(model, "milho_pragas_model.tflite", quantize=True)

    metadata = {
        "model_file": str(tflite_file.name),
        "size_kb": round(tflite_file.stat().st_size / 1024.0, 2),
        "class_names": class_names,
        "best_val_accuracy": round(float(trainer.best_val_acc), 4),
        "best_val_loss": round(float(trainer.best_val_loss), 4),
        "per_class_metrics": eval_report.get("per_class", {}),
    }
    Path("model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("✓ Metadados de produção salvos em 'model_metadata.json'!")


if __name__ == "__main__":
    main()
