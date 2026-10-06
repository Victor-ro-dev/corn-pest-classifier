#!/usr/bin/env python3
"""
testar_modelo.py — Motor de Inferência para testar o modelo TensorFlow Lite (.tflite).

Uso:
    # 1. Testar uma foto específica:
    python testar_modelo.py caminho/para/sua_foto.jpg

    # 2. Testar abrindo janela gráfica com a foto e a predição:
    python testar_modelo.py caminho/para/sua_foto.jpg --visual

    # 3. Teste automático (seleciona amostras das pastas em dataset/):
    python testar_modelo.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except ImportError:
    print("[ERRO] TensorFlow não encontrado. Instale com: pip install tensorflow")
    sys.exit(1)

# As 4 classes oficiais treinadas no modelo
CLASS_NAMES = [
    "lagarta_cartucho",  # Spodoptera frugiperda
    "lagarta_espiga",    # Helicoverpa zea
    "vaquinha",          # Diabrotica speciosa
    "milho_sadio",       # Planta/folha sadia (controle)
]

# Diagnósticos e recomendações rápidas
DESCRICOES = {
    "lagarta_cartucho": "Praga detectada: Lagarta-do-cartucho (Spodoptera frugiperda). Sinal de raspagem ou dano no cartucho.",
    "lagarta_espiga": "Praga detectada: Lagarta-da-espiga (Helicoverpa zea). Danos nos estilos-estigmas (cabelo) ou ponta da espiga.",
    "vaquinha": "Praga detectada: Vaquinha-verde-amarela (Diabrotica speciosa). Dano foliar mastigado/furado pelo besouro adulto.",
    "milho_sadio": "Planta sadia. Nenhum dano grave ou praga alvo identificada.",
}


class CornPestClassifier:
    """Motor de inferência para o modelo TensorFlow Lite de pragas do milho."""

    def __init__(self, model_path: str = "milho_pragas_model.tflite"):
        caminho = Path(model_path)
        if not caminho.exists():
            # Tenta encontrar no Desktop se não estiver na pasta atual
            desktop_path = Path.home() / "Desktop" / caminho.name
            if desktop_path.exists():
                caminho = desktop_path
            else:
                raise FileNotFoundError(
                    f"Modelo '{model_path}' não encontrado na pasta atual nem no Desktop."
                )

        self.model_path = caminho
        # Inicializa o interpretador TFLite
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Obtém dimensões esperadas pelo modelo (geralmente [1, 224, 224, 3])
        self.input_shape = self.input_details[0]["shape"]
        self.target_height = self.input_shape[1]
        self.target_width = self.input_shape[2]

    def preprocess(self, img_path: Path) -> np.ndarray:
        """Carrega e redimensiona a imagem para o formato esperado pelo modelo."""
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((self.target_width, self.target_height), Image.Resampling.BILINEAR)

        # Converte para array float32 com valores de 0 a 255 (mesmo input do dataset)
        arr = np.array(img_resized, dtype=np.float32)
        # Adiciona dimensão do lote: (1, 224, 224, 3)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self, img_path: str | Path) -> dict:
        """Executa a inferência e retorna as probabilidades por classe."""
        caminho = Path(img_path)
        tensor_entrada = self.preprocess(caminho)

        # Alimenta o tensor de entrada e roda o motor
        self.interpreter.set_tensor(self.input_details[0]["index"], tensor_entrada)
        self.interpreter.invoke()

        # Coleta as probabilidades calculadas
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        # Se a saída não for probabilidade direta (ex: logits), aplica softmax
        if np.max(output_data) > 1.0 or np.min(output_data) < 0.0 or not np.isclose(np.sum(output_data), 1.0, atol=0.05):
            exp = np.exp(output_data - np.max(output_data))
            probs = exp / np.sum(exp)
        else:
            probs = output_data

        top_idx = int(np.argmax(probs))
        top_classe = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"classe_{top_idx}"
        top_conf = float(probs[top_idx])

        detalhes = {}
        for i, p in enumerate(probs):
            nome = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"classe_{i}"
            detalhes[nome] = float(p)

        return {
            "arquivo": str(caminho),
            "classe": top_classe,
            "confianca": top_conf,
            "probabilidades": detalhes,
            "descricao": DESCRICOES.get(top_classe, ""),
        }


def exibir_resultado(resultado: dict):
    """Renderiza a resposta do motor com barras de probabilidade no terminal."""
    print("\n" + "=" * 65)
    print("🌽 RESULTADO DA INFERÊNCIA — MOTOR DE IA (MobileNetV2 TFLite)")
    print("=" * 65)
    print(f"📁 Imagem analisada: {resultado['arquivo']}")
    print(f"🎯 Classe detectada: {resultado['classe'].upper()}")
    print(f"📊 Certeza / Confiança: {resultado['confianca'] * 100:.2f}%")
    print("-" * 65)
    print("Distribuição de Probabilidades:")

    # Ordena as probabilidades da maior para a menor
    ordenadas = sorted(
        resultado["probabilidades"].items(), key=lambda x: x[1], reverse=True
    )
    for nome, prob in ordenadas:
        barras = int(prob * 30)
        barra_ascii = "█" * barras + "░" * (30 - barras)
        destaque = " <--- TOP 1" if nome == resultado["classe"] else ""
        print(f"  {nome:<18} [{barra_ascii}] {prob * 100:6.2f}%{destaque}")

    print("-" * 65)
    print(f"💡 Diagnóstico: {resultado['descricao']}")
    print("=" * 65 + "\n")


def plotar_resultado(img_path: Path, resultado: dict):
    """Exibe uma janela gráfica com a foto e a predição caso matplotlib esteja disponível."""
    try:
        import matplotlib.pyplot as plt

        img = Image.open(img_path)
        plt.figure(figsize=(7, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"Predição: {resultado['classe'].upper()} ({resultado['confianca']*100:.1f}%)",
            fontsize=14,
            fontweight="bold",
            color="green" if resultado["classe"] == "milho_sadio" else "darkred",
        )
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[AVISO] Não foi possível abrir janela visual: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Testa o modelo .tflite em uma imagem de campo do milho"
    )
    parser.add_argument(
        "imagem",
        nargs="?",
        default=None,
        help="Caminho da foto para teste (ex: foto.jpg). Se omitido, testa amostras do dataset.",
    )
    parser.add_argument(
        "--modelo",
        default="milho_pragas_model.tflite",
        help="Caminho do arquivo .tflite (padrão: milho_pragas_model.tflite)",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Abre janela gráfica do matplotlib exibindo a foto e a predição",
    )

    args = parser.parse_args()

    # Reconfigura encoding para evitar falhas no console do Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    try:
        motor = CornPestClassifier(model_path=args.modelo)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar motor de IA: {e}")
        return

    # Caso uma imagem específica tenha sido informada
    if args.imagem:
        caminho_foto = Path(args.imagem)
        if not caminho_foto.exists():
            print(f"[ERRO] Imagem não encontrada: {caminho_foto}")
            return
        res = motor.predict(caminho_foto)
        exibir_resultado(res)
        if args.visual:
            plotar_resultado(caminho_foto, res)
        return

    # Modo demonstração automática: busca fotos nas pastas de dataset
    print("[*] Nenhuma imagem informada. Procurando fotos de teste no dataset...")
    amostras_encontradas = []
    dataset_dir = Path("dataset")

    if dataset_dir.exists():
        for classe in CLASS_NAMES:
            pasta_classe = dataset_dir / classe
            if pasta_classe.is_dir():
                fotos = list(pasta_classe.glob("*.jpg")) + list(pasta_classe.glob("*.jpeg"))
                if fotos:
                    amostras_encontradas.append((classe, fotos[0]))

    if not amostras_encontradas:
        # Se não achou nas pastas de classes, procura qualquer jpg na pasta dataset
        fotos_gerais = list(dataset_dir.rglob("*.jpg"))[:3]
        for f in fotos_gerais:
            amostras_encontradas.append(("amostra", f))

    if not amostras_encontradas:
        print("[INFO] Nenhuma foto encontrada na pasta dataset para o teste automático.")
        print("Uso: python testar_modelo.py caminho/para/sua_foto.jpg")
        return

    print(f"Encontradas {len(amostras_encontradas)} fotos para teste. Executando inferências:\n")
    for rotulo_real, foto in amostras_encontradas:
        print(f">>> Testando amostra da pasta: [{rotulo_real}]")
        res = motor.predict(foto)
        exibir_resultado(res)
        if args.visual:
            plotar_resultado(foto, res)


if __name__ == "__main__":
    main()
