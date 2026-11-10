import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


class CustomTrainer:
    """Motor de treinamento customizado com @tf.function, GradientTape, Class Weights,
    ReduceLROnPlateau, Early Stopping, Restauração dos Melhores Pesos e Relatório de Métricas.
    """

    def __init__(
        self,
        model: keras.Model,
        learning_rate: float = 1e-3,
        class_weights: dict[int, float] | None = None,
        early_stopping_patience: int = 5,
        lr_patience: int = 2,
        min_lr: float = 1e-6,
    ):
        self.model = model
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction="none")
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

        self.early_stopping_patience = early_stopping_patience
        self.lr_patience = lr_patience
        self.min_lr = min_lr

        # Tensor de pesos de classe para balanceamento dentro do @tf.function
        num_outputs = model.output_shape[-1] or 4
        if class_weights:
            weights_list = [float(class_weights.get(i, 1.0)) for i in range(num_outputs)]
        else:
            weights_list = [1.0] * num_outputs
        self.class_weight_tensor = tf.constant(weights_list, dtype=tf.float32)

        # Métricas acumuladas por época (Média real de todos os batches)
        self.train_loss_metric = keras.metrics.Mean(name="train_loss")
        self.val_loss_metric = keras.metrics.Mean(name="val_loss")
        self.train_acc_metric = keras.metrics.SparseCategoricalAccuracy(name="train_acc")
        self.val_acc_metric = keras.metrics.SparseCategoricalAccuracy(name="val_acc")

        # Estado global dos melhores pesos entre estágios
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_weights: list[Any] | None = None
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

    def set_learning_rate(self, new_lr: float) -> None:
        """Atualiza dinamicamente o Learning Rate do otimizador."""
        self.optimizer.learning_rate.assign(float(new_lr))

    @tf.function
    def train_step(self, x_batch, y_batch):
        y_int = tf.cast(y_batch, tf.int32)
        sample_weights = tf.gather(self.class_weight_tensor, y_int)

        with tf.GradientTape() as tape:
            probs = self.model(x_batch, training=True)
            per_sample_loss = self.loss_fn(y_int, probs)
            weighted_loss = tf.reduce_mean(per_sample_loss * sample_weights)
            if self.model.losses:
                total_loss = weighted_loss + tf.add_n(self.model.losses)
            else:
                total_loss = weighted_loss

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.train_loss_metric.update_state(total_loss)
        self.train_acc_metric.update_state(y_int, probs)
        return total_loss

    @tf.function
    def test_step(self, x_batch, y_batch):
        y_int = tf.cast(y_batch, tf.int32)
        val_probs = self.model(x_batch, training=False)
        per_sample_loss = self.loss_fn(y_int, val_probs)
        val_loss = tf.reduce_mean(per_sample_loss)

        self.val_loss_metric.update_state(val_loss)
        self.val_acc_metric.update_state(y_int, val_probs)
        return val_loss

    def fit(
        self,
        train_ds,
        val_ds,
        epochs: int,
        stage_name: str = "Treinamento",
    ) -> dict[str, list[float]]:
        """Executa o loop de épocas com ReduceLROnPlateau, EarlyStopping e salvamento dos melhores pesos."""
        # Se o dataset já trouxe class_weights calculados, atualiza o tensor
        ds_weights = getattr(train_ds, "class_weights", None)
        if ds_weights:
            num_outputs = self.model.output_shape[-1] or len(ds_weights)
            w_list = [float(ds_weights.get(i, 1.0)) for i in range(num_outputs)]
            self.class_weight_tensor = tf.constant(w_list, dtype=tf.float32)

        epochs_without_improvement = 0
        lr_wait = 0

        for epoch in range(epochs):
            start_time = time.time()
            self.train_loss_metric.reset_state()
            self.train_acc_metric.reset_state()
            self.val_loss_metric.reset_state()
            self.val_acc_metric.reset_state()

            for x_batch_train, y_batch_train in train_ds:
                self.train_step(x_batch_train, y_batch_train)

            for x_batch_val, y_batch_val in val_ds:
                self.test_step(x_batch_val, y_batch_val)

            train_loss = float(self.train_loss_metric.result())
            train_acc = float(self.train_acc_metric.result())
            val_loss = float(self.val_loss_metric.result())
            val_acc = float(self.val_acc_metric.result())
            current_lr = float(self.optimizer.learning_rate.numpy())

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - start_time

            # Verifica se houve melhora na validação (menor val_loss ou maior val_acc com empate de loss)
            improved = False
            if val_loss < (self.best_val_loss - 1e-4) or (
                val_acc > self.best_val_acc and val_loss <= self.best_val_loss * 1.02
            ):
                self.best_val_loss = min(val_loss, self.best_val_loss)
                self.best_val_acc = max(val_acc, self.best_val_acc)
                self.best_weights = self.model.get_weights()
                improved = True
                epochs_without_improvement = 0
                lr_wait = 0
                try:
                    self.model.save("best_milho_pragas.keras")
                except Exception:
                    pass
            else:
                epochs_without_improvement += 1
                lr_wait += 1

            badge = "★ [MELHOR MODELO SALVO]" if improved else ""
            print(
                f"[{stage_name}] Época {epoch + 1:02d}/{epochs:02d} ({elapsed:.1f}s) | "
                f"LR: {current_lr:.1e} | "
                f"Treino Loss: {train_loss:.4f} - Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2%} {badge}"
            )

            # ReduceLROnPlateau
            if lr_wait >= self.lr_patience:
                new_lr = max(current_lr * 0.5, self.min_lr)
                if new_lr < current_lr:
                    self.set_learning_rate(new_lr)
                    print(f"  📉 [ReduceLROnPlateau] Reduzindo Learning Rate para {new_lr:.1e}")
                lr_wait = 0

            # Early Stopping
            if epochs_without_improvement >= self.early_stopping_patience:
                print(
                    f"  🛑 [Early Stopping] Sem melhora por {self.early_stopping_patience} épocas seguidas. Encerrando estágio."
                )
                break

        # Restaura os melhores pesos validados
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(
                f"✓ Melhores pesos restaurados (Melhor Val Loss: {self.best_val_loss:.4f} | Melhor Val Acc: {self.best_val_acc:.2%})"
            )

        return self.history

    def plot_training_curves(
        self,
        save_path: str = "training_curves.png",
        warmup_epochs_ran: int | None = None,
    ) -> None:
        """Plota e salva as curvas de Loss e Acurácia de Treino vs Validação."""
        if not self.history["train_loss"]:
            return

        epochs_range = range(1, len(self.history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # 1. Curva de Loss
        ax1.plot(epochs_range, self.history["train_loss"], "o-", label="Treino Loss", color="#1f77b4")
        ax1.plot(epochs_range, self.history["val_loss"], "s-", label="Validação Loss", color="#d62728")
        if warmup_epochs_ran and warmup_epochs_ran < len(epochs_range):
            ax1.axvline(x=warmup_epochs_ran + 0.5, color="gray", linestyle="--", label="Início Fine-Tuning")
        ax1.set_title("Evolução da Perda (Loss)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Época")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Curva de Acurácia
        ax2.plot(epochs_range, self.history["train_acc"], "o-", label="Treino Acurácia", color="#2ca02c")
        ax2.plot(epochs_range, self.history["val_acc"], "s-", label="Validação Acurácia", color="#ff7f0e")
        if warmup_epochs_ran and warmup_epochs_ran < len(epochs_range):
            ax2.axvline(x=warmup_epochs_ran + 0.5, color="gray", linestyle="--", label="Início Fine-Tuning")
        ax2.set_title("Evolução da Acurácia", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Época")
        ax2.set_ylabel("Acurácia")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Curvas de treinamento salvas em: {save_path}")

    def evaluate_per_class(
        self,
        val_ds,
        class_names: list[str],
        save_path: str = "confusion_matrix.png",
    ) -> dict[str, Any]:
        """Gera a Matriz de Confusão e calcula Precisão, Recall e F1-Score por classe na validação."""
        y_true_all: list[int] = []
        y_pred_all: list[int] = []

        for x_batch, y_batch in val_ds:
            probs = self.model(x_batch, training=False).numpy()
            preds = np.argmax(probs, axis=1)
            y_true_all.extend(y_batch.numpy().astype(int).tolist())
            y_pred_all.extend(preds.astype(int).tolist())

        num_classes = max(len(class_names), 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true_all, y_pred_all):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1

        print("\n=== RELATÓRIO DE DESEMPENHO POR CLASSE (VALIDAÇÃO) ===")
        metrics_by_class: dict[str, dict[str, float]] = {}
        for i, cls_name in enumerate(class_names):
            tp = int(cm[i, i])
            fp = int(np.sum(cm[:, i]) - tp)
            fn = int(np.sum(cm[i, :]) - tp)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            metrics_by_class[cls_name] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int(np.sum(cm[i, :])),
            }
            print(
                f"  - {cls_name:<18} | Precisão: {precision:.1%} | Recall: {recall:.1%} | "
                f"F1-Score: {f1:.1%} ({tp}/{int(np.sum(cm[i, :]))} acertos)"
            )

        # Plota a Matriz de Confusão
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            title="Matriz de Confusão - Validação (MobileNetV2 Milho)",
            ylabel="Classe Real",
            xlabel="Classe Predita",
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.0 if cm.size > 0 else 0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold",
                )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Matriz de Confusão salva em: {save_path}")

        return {"confusion_matrix": cm.tolist(), "per_class": metrics_by_class}
