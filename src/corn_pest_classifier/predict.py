"""
predict.py — Módulo exportável de inferência para o classificador de pragas do milho.
Pode ser importado por APIs (FastAPI, Flask) ou bots de mensageria (WhatsApp, Telegram).
"""

from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

from corn_pest_classifier.config import CLASS_NAMES

DESCRICOES = {
    "lagarta_cartucho": "Praga detectada: Lagarta-do-cartucho (Spodoptera frugiperda). Sinal de raspagem ou dano no cartucho.",
    "lagarta_espiga": "Praga detectada: Lagarta-da-espiga (Helicoverpa zea). Danos nos estilos-estigmas ou ponta da espiga.",
    "vaquinha": "Praga detectada: Vaquinha-verde-amarela (Diabrotica speciosa). Dano foliar perfurado pelo besouro adulto.",
    "milho_sadio": "Planta sadia. Nenhum dano grave ou praga alvo identificada.",
}


class CornPestClassifier:
    """Motor de inferência TFLite para classificação de pragas do milho."""

    def __init__(self, model_path: str | Path = "milho_pragas_model.tflite"):
        caminho = Path(model_path)
        if not caminho.exists():
            desktop_path = Path.home() / "Desktop" / caminho.name
            if desktop_path.exists():
                caminho = desktop_path
            else:
                raise FileNotFoundError(f"Modelo não encontrado em '{model_path}'.")

        self.interpreter = tf.lite.Interpreter(model_path=str(caminho))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        shape = self.input_details[0]["shape"]
        self.target_height = shape[1]
        self.target_width = shape[2]

    def predict(self, image_input: str | Path | Image.Image) -> dict:
        """Recebe um caminho de imagem ou objeto PIL.Image e retorna a predição."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
            caminho_str = str(image_input)
        else:
            img = image_input.convert("RGB")
            caminho_str = "objeto_em_memoria"

        img_resized = img.resize((self.target_width, self.target_height), Image.Resampling.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], arr)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        if np.max(output_data) > 1.0 or np.min(output_data) < 0.0 or not np.isclose(np.sum(output_data), 1.0, atol=0.05):
            exp = np.exp(output_data - np.max(output_data))
            probs = exp / np.sum(exp)
        else:
            probs = output_data

        top_idx = int(np.argmax(probs))
        top_classe = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"classe_{top_idx}"

        detalhes = {
            CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"classe_{i}": float(p)
            for i, p in enumerate(probs)
        }

        return {
            "arquivo": caminho_str,
            "classe": top_classe,
            "confianca": float(probs[top_idx]),
            "probabilidades": detalhes,
            "diagnostico": DESCRICOES.get(top_classe, ""),
        }
