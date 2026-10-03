import sys
from pathlib import Path

# Adiciona o diretório atual ao path para compatibilidade de importação
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    from corn_pest_classifier.data_loader import build_dataset
except ImportError:
    from data_loader import build_dataset


def visualizar_lote():
    """
    Carrega o dataset e gera uma visualização gráfica de 9 imagens do lote,
    salvando a figura em disco e abrindo a janela do matplotlib.
    """
    print("Carregando datasets...")
    train_ds, val_ds = build_dataset()

    class_names = getattr(train_ds, "class_names", [])
    print(f"Classes detectadas ({len(class_names)}): {class_names}")

    # Pega exatamente 1 lote (batch) de treino
    for images, labels in train_ds.take(1):
        print("\n[INFO DO LOTE]")
        print(f" - Formato do tensor (Batch, Altura, Largura, Canais): {images.shape}")
        print(f" - Tipo de dado: {images.dtype}")
        print(f" - Valor min do pixel: {float(tf.reduce_min(images)):.1f}, max: {float(tf.reduce_max(images)):.1f}")

        # -------------------------------------------------------------
        # 1. Grade 3x3 com 9 imagens do lote
        # -------------------------------------------------------------
        fig, axes = plt.subplots(3, 3, figsize=(11, 11))
        fig.suptitle("Amostras do Lote de Treinamento", fontsize=16, fontweight="bold")

        qtd_imagens = min(9, len(images))
        for i in range(qtd_imagens):
            ax = axes[i // 3, i % 3]
            img = images[i].numpy()

            # Ajusta valores caso estejam normalizados entre 0 e 1 ou em 0 e 255
            if img.max() <= 1.0:
                img_display = (img * 255).astype(np.uint8)
            else:
                img_display = img.astype(np.uint8)

            label_idx = int(labels[i])
            nome_classe = class_names[label_idx] if label_idx < len(class_names) else f"Classe {label_idx}"

            ax.imshow(img_display)
            ax.set_title(f"{nome_classe} (ID: {label_idx})", fontsize=11, fontweight="medium")
            ax.axis("off")

        # Desativa eixos excedentes caso o batch tenha menos de 9 fotos
        for j in range(qtd_imagens, 9):
            axes[j // 3, j % 3].axis("off")

        plt.tight_layout()
        caminho_salvar_lote = "preview_batch.png"
        plt.savefig(caminho_salvar_lote, dpi=150)
        print(f"\n[OK] Imagem do lote salva em: {caminho_salvar_lote}")

        # -------------------------------------------------------------
        # 2. Demonstração do Data Augmentation na 1ª imagem
        # -------------------------------------------------------------
        data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.2),
        ])

        fig_aug, axes_aug = plt.subplots(3, 3, figsize=(11, 11))
        fig_aug.suptitle("Efeito do Data Augmentation na mesma imagem (9 variações)", fontsize=14, fontweight="bold")

        primeira_imagem = images[0]
        for i in range(9):
            ax = axes_aug[i // 3, i % 3]
            img_aug = data_augmentation(tf.expand_dims(primeira_imagem, 0), training=True)[0].numpy()

            if img_aug.max() <= 1.0:
                img_display = (img_aug * 255).astype(np.uint8)
            else:
                img_display = img_aug.astype(np.uint8)

            ax.imshow(img_display)
            ax.set_title(f"Variação #{i + 1}", fontsize=10)
            ax.axis("off")

        plt.tight_layout()
        caminho_salvar_aug = "preview_augmentation.png"
        plt.savefig(caminho_salvar_aug, dpi=150)
        print(f"[OK] Amostra de Data Augmentation salva em: {caminho_salvar_aug}")

        # Exibe na tela (caso haja interface gráfica ativa)
        plt.show()
        break


if __name__ == "__main__":
    visualizar_lote()
