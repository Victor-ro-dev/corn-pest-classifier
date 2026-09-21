import os
import time

import pandas as pd
import requests

# Configurações de diretório
ARQUIVO_GBIF = "dataset/0007043-260916113435855/multimedia.txt" # O arquivo extraído do zip
DIRETORIO_SAIDA = "dataset/praga/"
LIMITE_IMAGENS = 1000  # Limite para não lotar o HD no primeiro teste

def baixar_imagens_gbif():
    # Cria a pasta caso não exista
    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

    print(f"Lendo o arquivo {ARQUIVO_GBIF}...")
    # O GBIF usa tabulação (\t) como separador em vez de vírgula
    df = pd.read_csv(ARQUIVO_GBIF, sep="\t", usecols=["identifier", "format"])

    # Filtrar apenas as linhas que são imagens (JPEG)
    df_imagens = df[df["format"].str.contains("image/jpeg", na=False, case=False)]
    urls = df_imagens["identifier"].dropna().tolist()

    print(f"Encontradas {len(urls)} URLs de imagens. Iniciando download de até {LIMITE_IMAGENS} fotos...")

    sucessos = 0
    for i, url in enumerate(urls):
        if sucessos >= LIMITE_IMAGENS:
            break

        try:
            # Faz o download da imagem com um timeout de 10 segundos
            resposta = requests.get(url, timeout=10)

            # Se o link estiver quebrado (Erro 404, 403, etc), pula para o próximo
            if resposta.status_code != 200:
                continue

            # Salva o arquivo em disco com um nome único (ex: img_0001.jpg)
            nome_arquivo = os.path.join(DIRETORIO_SAIDA, f"img_{sucessos:04d}.jpg")
            with open(nome_arquivo, "wb") as f:
                f.write(resposta.content)

            sucessos += 1
            if sucessos % 50 == 0:
                print(f"[{sucessos}/{LIMITE_IMAGENS}] imagens baixadas...")

            # Pequena pausa para não derrubar o servidor de origem
            time.sleep(0.1)

        except Exception:
            # Ignora erros de conexão ou timeout e continua
            pass

    print(f"Download concluído! {sucessos} imagens salvas em {DIRETORIO_SAIDA}.")

if __name__ == "__main__":
    baixar_imagens_gbif()
