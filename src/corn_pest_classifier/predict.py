"""
predict.py — Módulo exportável de inferência para o classificador de pragas do milho.
Pode ser importado por APIs (FastAPI, Flask) ou webhooks de WhatsApp (Evolution API, Z-API, etc.).
Suporta entrada via caminho de arquivo, bytes, PIL.Image ou string Base64 (padrão Evolution API).
"""

import base64
import binascii
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

try:
    from corn_pest_classifier.config import CLASS_NAMES
except ImportError:
    from config import CLASS_NAMES

DESCRICOES = {
    "lagarta_cartucho": "Praga detectada: Lagarta-do-cartucho (Spodoptera frugiperda). Sinal de raspagem ou dano no cartucho.",
    "lagarta_espiga": "Praga detectada: Lagarta-da-espiga (Helicoverpa zea). Danos nos estilos-estigmas ou ponta da espiga.",
    "vaquinha": "Praga detectada: Vaquinha-verde-amarela (Diabrotica speciosa). Dano foliar perfurado pelo besouro adulto.",
    "milho_sadio": "Planta sadia. Nenhum dano grave ou praga alvo identificada.",
}


def is_base64_image_string(value: str) -> bool:
    """Detecta se uma string é um payload Base64 de imagem (com ou sem data URI da Evolution API)."""
    if not isinstance(value, str):
        return False
    s = value.strip()
    if s.startswith("data:image/") and ";base64," in s:
        return True
    # Assinaturas clássicas de JPEG (/9j/), PNG (iVBOR), WebP (UklGR) ou strings longas sem quebras de caminho
    if s.startswith(("/9j/", "iVBOR", "UklGR")) and len(s) > 64:
        return True
    if len(s) > 256 and "/" not in s[:30] and "\\" not in s[:30]:
        return True
    return False


def decode_base64_to_bytes(b64_string: str) -> bytes:
    """Decodifica string Base64 (padrão Evolution API: pura ou com prefixo data:image/...;base64,)."""
    cleaned = b64_string.strip()
    if "," in cleaned and cleaned.startswith("data:"):
        cleaned = cleaned.split(",", 1)[1].strip()
    # Remove espaços/quebras de linha eventuais do payload JSON
    cleaned = "".join(cleaned.split())
    # Ajusta padding '=' se necessário
    rem = len(cleaned) % 4
    if rem:
        cleaned += "=" * (4 - rem)
    return base64.b64decode(cleaned)


def load_rgb_image(image_input: str | bytes | bytearray | Path | Image.Image) -> tuple[Image.Image, str]:
    """Carrega e normaliza qualquer entrada (Path, Base64 da Evolution API, bytes ou PIL.Image) para RGB."""
    if isinstance(image_input, Image.Image):
        img = ImageOps.exif_transpose(image_input).convert("RGB")
        return img, "pil_em_memoria"

    if isinstance(image_input, (bytes, bytearray)):
        raw = bytes(image_input)
        # Se os bytes forem texto Base64 codificado em ASCII/UTF-8
        if raw.startswith((b"data:image/", b"/9j/", b"iVBOR")):
            raw = decode_base64_to_bytes(raw.decode("ascii", errors="ignore"))
        with Image.open(io.BytesIO(raw)) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
        return img, "bytes_em_memoria"

    if isinstance(image_input, str) and is_base64_image_string(image_input):
        raw = decode_base64_to_bytes(image_input)
        with Image.open(io.BytesIO(raw)) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
        return img, "base64_evolution"

    caminho = Path(image_input)
    with Image.open(caminho) as opened:
        img = ImageOps.exif_transpose(opened).convert("RGB")
    return img, str(caminho)


class CornPestClassifier:
    """Motor de inferência TFLite para classificação de pragas do milho (compatível com Evolution API Base64)."""

    def __init__(self, model_path: str | Path = "milho_pragas_model.tflite"):
        caminho = Path(model_path)
        if not caminho.exists():
            desktop_path = Path.home() / "Desktop" / caminho.name
            if desktop_path.exists():
                caminho = desktop_path
            else:
                raise FileNotFoundError(f"Modelo não encontrado em '{model_path}'.")

        self.interpreter = Interpreter(model_path=str(caminho))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        shape = self.input_details[0]["shape"]
        self.target_height = shape[1]
        self.target_width = shape[2]

    def predict(self, image_input: str | bytes | bytearray | Path | Image.Image) -> dict:
        """Recebe caminho de arquivo, string Base64 (Evolution API), bytes ou PIL.Image e retorna a predição."""
        img, origem_str = load_rgb_image(image_input)

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
            "arquivo": origem_str,
            "classe": top_classe,
            "confianca": float(probs[top_idx]),
            "probabilidades": detalhes,
            "diagnostico": DESCRICOES.get(top_classe, ""),
        }

    def predict_base64(self, base64_str: str) -> dict:
        """Atalho explícito para classificar diretamente o campo `base64` vindo do webhook da Evolution API."""
        raw_bytes = decode_base64_to_bytes(base64_str)
        return self.predict(raw_bytes)
