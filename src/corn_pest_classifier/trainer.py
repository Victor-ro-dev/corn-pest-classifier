import time

import tensorflow as tf
from tensorflow import keras


class CustomTrainer:
    def __init__(self, model, learning_rate: float):
        self.model = model
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        # Métricas de acompanhamento
        self.train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    @tf.function # Compila este bloco em um grafo C++ de alta performance
    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            logits = self.model(x_batch, training=True)
            loss_value = self.loss_fn(y_batch, logits)
            
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y_batch, logits)
        return loss_value

    @tf.function
    def test_step(self, x_batch, y_batch):
        val_logits = self.model(x_batch, training=False)
        val_loss = self.loss_fn(y_batch, val_logits)
        self.val_acc_metric.update_state(y_batch, val_logits)
        return val_loss

    def fit(self, train_ds, val_ds, epochs: int):
        for epoch in range(epochs):
            print(f"\nÉpoca {epoch + 1}/{epochs}")
            start_time = time.time()
            
            # Loop de treino batch por batch
            last_train_loss = 0.0
            for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
                last_train_loss = self.train_step(x_batch_train, y_batch_train)
                
            train_acc = self.train_acc_metric.result()
            self.train_acc_metric.reset_state()
            
            # Loop de validação batch por batch
            val_loss_total = 0.0
            val_steps = 0
            for x_batch_val, y_batch_val in val_ds:
                v_loss = self.test_step(x_batch_val, y_batch_val)
                val_loss_total += float(v_loss)
                val_steps += 1

            val_loss_avg = val_loss_total / max(val_steps, 1)
            val_acc = self.val_acc_metric.result()
            self.val_acc_metric.reset_state()

            elapsed = time.time() - start_time
            print(
                f"Tempo: {elapsed:.2f}s | "
                f"Treino Loss: {float(last_train_loss):.4f} - Acurácia: {float(train_acc):.4f} | "
                f"Val Loss: {val_loss_avg:.4f} - Acurácia: {float(val_acc):.4f}"
            )