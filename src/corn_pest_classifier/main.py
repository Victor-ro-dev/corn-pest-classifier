import sys
from pathlib import Path

# Garante compatibilidade de imports em qualquer ambiente (local, Colab, CLI)
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import tensorflow as tf

try:
    from corn_pest_classifier.config import EPOCHS, LEARNING_RATE, NUM_CLASSES
    from corn_pest_classifier.data_loader import build_dataset
    from corn_pest_classifier.mobilenet_custom import build_custom_mobilenet
    from corn_pest_classifier.trainer import CustomTrainer
except ImportError:
    from config import EPOCHS, LEARNING_RATE, NUM_CLASSES
    from data_loader import build_dataset
    from mobilenet_custom import build_custom_mobilenet
    from trainer import CustomTrainer


def main():
    print("1. Construindo o Dataset...")
    train_ds, val_ds = build_dataset()

    # Detecta automaticamente o número real de classes do dataset
    class_names = getattr(train_ds, "class_names", [])
    qtd_classes = len(class_names) if class_names else NUM_CLASSES
    print(f"2. Instanciando a Arquitetura Customizada para {qtd_classes} classes...")
    model = build_custom_mobilenet(num_classes=qtd_classes)
    model.summary() # Exibe a arquitetura algébrica no console

    print("3. Iniciando o Loop de Treinamento Customizado (Nível 1)...")
    trainer = CustomTrainer(model, learning_rate=LEARNING_RATE)
    trainer.fit(train_ds, val_ds, epochs=EPOCHS)

    print("4. Exportando modelo leve (.tflite) para Deploy via WhatsApp...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open("milho_pragas_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("Modelo salvo com sucesso em 'milho_pragas_model.tflite'!")


if __name__ == "__main__":
    main()