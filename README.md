# 🌽 Corn Pest Classifier — MobileNetV2 Customizada & Treinamento

Classificador de visão computacional para identificação de pragas em plantações de milho, utilizando uma arquitetura **MobileNetV2 customizada** (com blocos residuais invertidos e convoluções separáveis em profundidade) implementada em **TensorFlow / Keras**, com suporte a treinamento acelerado por GPU no **Google Colab** e exportação otimizada para **TensorFlow Lite (`.tflite`)**.

---

## 📌 Funcionalidades Principais

- **Arquitetura MobileNetV2 Customizada (`mobilenet_custom.py`):**
  - Implementação de blocos residuais invertidos (*Inverted Residual Blocks*) com expansão 1x1, convolução em profundidade 3x3 (*DepthwiseConv2D*) e projeção linear 1x1.
  - Normalização por lote (*Batch Normalization*) e ativação ReLU6.
  - Conexão residual (*Skip Connection*) para preservação de gradientes.
- **Loop de Treinamento Customizado (`trainer.py`):**
  - Controle de baixo nível com `tf.GradientTape` e otimização Adam.
  - Compilação dos passos de treino e teste em grafos de alta performance via `@tf.function`.
  - Métricas em tempo real por época: Loss e Acurácia de Treino e Validação.
- **Pipeline de Dados Robusto (`data_loader.py`):**
  - Carregamento com `keras.utils.image_dataset_from_directory`.
  - Pipeline de Data Augmentation integrado (Rotações e Flips aleatórios).
  - Otimização de I/O assíncrono com `prefetch` e `AUTOTUNE`.
- **Exportação para Borda (`.tflite`):**
  - Conversão automatizada para TensorFlow Lite ao final do treino, pronta para deploy em dispositivos móveis, microcontroladores ou APIs leves (ex.: bots de WhatsApp).
- **Suporte ao Google Colab & Servidor MCP:**
  - Notebook interativo pronto (`treinamento_colab.ipynb`).
  - Configuração do novo servidor MCP oficial do Colab (`colab-mcp`) para agentes locais.

---

## 📂 Estrutura do Projeto

```text
corn_pest_classifier/
├── .agents/
│   └── mcp_config.json          # Configuração do Servidor MCP do Google Colab
├── dataset/                     # Diretório de dados (ignorado pelo git)
│   ├── praga/                   # Imagens de pragas (ex: lagartas do milho)
│   └── saudavel/                # Imagens de folhas/plantas sadias (controle)
├── src/
│   └── corn_pest_classifier/
│       ├── __init__.py
│       ├── config.py            # Hiperparâmetros (Batch size, Epochs, LR, Dimensões)
│       ├── data_loader.py       # Pipeline tf.data e Data Augmentation
│       ├── download_gbif.py     # Script utilitário para coletar dados do portal GBIF
│       ├── main.py              # Ponto de entrada do pipeline de treino e exportação
│       ├── mobilenet_custom.py  # Definição modular da arquitetura MobileNetV2
│       ├── trainer.py           # Loop customizado de treino com tf.GradientTape
│       └── visualizator.py      # Inspeção visual de lotes e augmentations
├── pyproject.toml               # Configuração do projeto e dependências (uv/pip)
├── requirements.txt             # Dependências diretas do projeto
├── treinamento_colab.ipynb      # Notebook pronto para executar no Google Colab
└── README.md                    # Documentação do projeto
```

---

## ⚙️ Pré-requisitos & Instalação

### Opção 1: Usando `uv` (Recomendado)
```bash
# Sincroniza o ambiente virtual e dependências
uv sync
```

### Opção 2: Usando `pip` tradicional
```bash
python -m venv .venv
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🗂️ Preparação do Dataset (Cultura: Milho)

O classificador foi projetado especificamente para a cultura do **milho** (escopo do semestre), organizado nas 4 classes oficiais:

| Classe | Por que entra | Dataset alvo (pós-limpeza) | Fonte |
| :--- | :--- | :--- | :--- |
| **`lagarta_cartucho`** | Praga nº 1 para quem planta sem semente Bt | ~2.943 amostras | GBIF (larvas) + KaraAgro AI Maize (recortes) |
| **`lagarta_espiga`** | Decisiva para milho verde de feira | ~351 amostras | GBIF (larvas, incl. CC BY-NC para treino local) |
| **`vaquinha`** | Espécie brasileira, dano visual inconfundível | ~458 amostras | GBIF (adultos) |
| **`milho_sadio`** | Classe negativa — permite responder "não tem nada aqui" | ~627 amostras | KaraAgro AI Maize (fotos de campo rotuladas *healthy*) |

### Estrutura de Diretórios Esperada:
```text
dataset/
├── lagarta_cartucho/    # Spodoptera frugiperda
├── lagarta_espiga/      # Helicoverpa zea
├── vaquinha/            # Diabrotica speciosa
├── milho_sadio/         # Folhas e plantas sadias de controle
└── creditos.csv         # Auditoria com autor, licença e URL de cada imagem
```

### Scripts de Coleta e Preparação:

#### 1. Download de Fotos de Campo via GBIF (`01_baixar_gbif.py`)
Baixa ocorrências públicas com licença declarada e estágio larval filtrado:
```bash
# Teste rápido (baixa 5 fotos de cada praga):
python 01_baixar_gbif.py --teste

# Download completo para as pastas de pragas em dataset/:
python 01_baixar_gbif.py --max-por-classe 500 --saida dataset --modo subpastas --incluir-nc
```

#### 2. Recorte de Anotações do KaraAgro AI Maize (`04_recortar_karaagro.py`)
Recorta as caixas delimitadoras PASCAL VOC do dataset de Harvard Dataverse:
```bash
# Mapeia caixas de FAW para 'lagarta_cartucho' e amostras sadias para 'milho_sadio':
python 04_recortar_karaagro.py --dir caminho_para_pasta_extraida/ --saida dataset/
```

---

## 🚀 Como Executar

### 1. Visualizar o Lote de Dados e Augmentations
Para inspecionar uma grade 3x3 do lote de treinamento e conferir o efeito do Data Augmentation:
```bash
python src/corn_pest_classifier/visualizator.py
```
*(Gera os arquivos `preview_batch.png` e `preview_augmentation.png`)*

### 2. Iniciar o Treinamento Localmente
```bash
python src/corn_pest_classifier/main.py
```
Ao término das épocas, o arquivo otimizado `milho_pragas_model.tflite` será gerado na raiz.

---

## ☁️ Treinamento no Google Colab

Para acelerar o treinamento usando GPUs gratuitas (ex.: T4 GPU) no Google Colab:

1. Abra o arquivo [`treinamento_colab.ipynb`](treinamento_colab.ipynb) no Google Colab.
2. No menu do Colab, selecione **Ambiente de Execução > Alterar tipo de ambiente de execução** e escolha **T4 GPU**.
3. Envie seus dados ou clone este repositório no runtime do Colab.
4. Execute as células sequencialmente para treinar o modelo e baixar o `.tflite` gerado.

### Conexão via Servidor MCP do Colab (`colab-mcp`)
Este repositório já inclui as configurações no arquivo `.agents/mcp_config.json`. Para conectar seu assistente/agente local ao Colab:
```bash
uvx --from git+https://github.com/googlecolab/colab-mcp colab-mcp
```
Faça a autenticação OAuth inicial solicitada pelo navegador para permitir o controle remoto de notebooks no seu Google Drive.

---

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.
