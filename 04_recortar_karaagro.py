#!/usr/bin/env python3
"""
04_recortar_karaagro.py — Processa e recorta anotações PASCAL VOC do KaraAgro AI Maize.

Contexto e Mapeamento de Classes (Cultura: Milho):
  1. lagarta_cartucho (Spodoptera frugiperda):
     Recorta caixas anotadas como:
       - fall-armyworm-larva (a lagarta)
       - fall-armyworm-larval-damage (cartucho raspado/perfurado)
       - fall-armyworm-frass (excremento no cartucho)
       - fall-armyworm-egg (postura de ovos)

  2. milho_sadio (Controle negativo):
     Recorta/extrai amostras de plantas e folhas rotuladas como:
       - healthy / healthy-leaf / maize-healthy

Origem do Dataset:
  - KaraAgro AI Maize (Harvard Dataverse, DOI: 10.7910/DVN/CXUMDS, Licença: CC BY)
  - Autor: KaraAgro AI Foundation

Uso:
    python 04_recortar_karaagro.py --dir pasta_com_arquivos_extraidos/
    python 04_recortar_karaagro.py --dir pasta_com_arquivos_extraidos/ --saida dataset/
"""

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

FONTE = "KaraAgro AI Maize (Harvard Dataverse, DOI 10.7910/DVN/CXUMDS)"
LICENCA = "CC BY"
AUTOR = "KaraAgro AI Foundation"
URL_FONTE = "https://doi.org/10.7910/DVN/CXUMDS"

# Caixa muito pequena (< 40px) fica ilegível no resize para 224px da MobileNetV2
LADO_MINIMO_PX = 40

# Margem de contexto de 15% ao redor do objeto para preservar o fundo foliar
MARGEM = 0.15

# Mapeamento oficial dos objetos PASCAL VOC para as classes do classificador
OBJETOS_PARA_CLASSE = {
    # 1. lagarta_cartucho
    "fall-armyworm-larva": "lagarta_cartucho",
    "fall-armyworm-larval-damage": "lagarta_cartucho",
    "fall-armyworm-frass": "lagarta_cartucho",
    "fall-armyworm-egg": "lagarta_cartucho",
    "faw-larva": "lagarta_cartucho",
    "faw-damage": "lagarta_cartucho",
    # 2. milho_sadio
    "healthy": "milho_sadio",
    "healthy-leaf": "milho_sadio",
    "healthy_leaf": "milho_sadio",
    "maize-healthy": "milho_sadio",
    "healthy-maize": "milho_sadio",
}


def caixas_do_xml(caminho_xml: Path):
    """
    Lê anotações XML PASCAL VOC.
    Normaliza limites (algumas caixas vêm com xmin > xmax ou ymin > ymax no KaraAgro).
    """
    try:
        tree = ET.parse(caminho_xml)
        for obj in tree.findall(".//object"):
            nome = obj.findtext("name", "").strip().lower()
            box = obj.find("bndbox")
            if box is None:
                continue
            try:
                x1 = int(float(box.findtext("xmin", "0")))
                y1 = int(float(box.findtext("ymin", "0")))
                x2 = int(float(box.findtext("xmax", "0")))
                y2 = int(float(box.findtext("ymax", "0")))
                yield nome, (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            except (ValueError, TypeError):
                continue
    except Exception as e:
        print(f"  [AVISO] Erro ao ler XML '{caminho_xml.name}': {e}")


def recortar_com_margem(img: Image.Image, caixa: tuple[int, int, int, int]):
    """
    Aplica margem percentual ao redor da caixa anotada e recorta a imagem com segurança.
    """
    xmin, ymin, xmax, ymax = caixa
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(img.width, xmax), min(img.height, ymax)

    if xmax <= xmin or ymax <= ymin:
        return None

    w, h = xmax - xmin, ymax - ymin
    mx, my = int(w * MARGEM), int(h * MARGEM)
    esq = max(0, xmin - mx)
    topo = max(0, ymin - my)
    dir_ = min(img.width, xmax + mx)
    fundo = min(img.height, ymax + my)

    return img.crop((esq, topo, dir_, fundo))


def main():
    parser = argparse.ArgumentParser(
        description="Recorta objetos do dataset KaraAgro AI Maize e mapeia para as classes do projeto"
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Pasta raiz contendo os arquivos extraídos (.xml e imagens .jpeg/.jpg)",
    )
    parser.add_argument(
        "--saida",
        default="dataset",
        help="Diretório de saída para as classes (padrão: 'dataset', gerando dataset/<classe>/)",
    )
    parser.add_argument(
        "--classe",
        default="auto",
        help=(
            "'auto': mapeia automaticamente pragas para 'lagarta_cartucho' e sadias para 'milho_sadio'. "
            "Ou especifique uma classe única para forçar (ex.: 'lagarta_cartucho' ou 'milho_sadio')."
        ),
    )
    parser.add_argument(
        "--creditos",
        default=None,
        help="Caminho do arquivo creditos.csv (padrão: <saida>/creditos.csv)",
    )

    args = parser.parse_args()

    # Compatibilidade de encoding de saída no Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    raiz_dados = Path(args.dir)
    if not raiz_dados.exists():
        print(f"[ERRO] Diretório de entrada não encontrado: {raiz_dados.resolve()}")
        sys.exit(1)

    raiz_saida = Path(args.saida)
    raiz_saida.mkdir(parents=True, exist_ok=True)

    creditos_path = Path(args.creditos) if args.creditos else raiz_saida / "creditos.csv"
    creditos_path.parent.mkdir(parents=True, exist_ok=True)
    novo_csv = not creditos_path.exists()

    xmls = sorted(raiz_dados.rglob("*.xml"))
    print("=" * 65)
    print("🌽 PROCESSADOR KARAAGRO AI MAIZE — RECORTE E MAPEAMENTO")
    print(f"Diretório de origem: {raiz_dados.resolve()}")
    print(f"Diretório de saída: {raiz_saida.resolve()}")
    print(f"Modo de mapeamento: {args.classe}")
    print(f"Anotações XML localizadas: {len(xmls)}")
    print("=" * 65)

    contagem_classes = {}
    contagem_objetos = {}
    total_recortado = 0
    fotos_sem_imagem = 0

    with creditos_path.open("a", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        if novo_csv:
            w.writerow([
                "arquivo",
                "classe",
                "fonte",
                "gbif_occurrence_key",
                "url_original",
                "licenca",
                "autor",
            ])

        iterable = xmls
        bar = tqdm(xmls, desc="Processando XMLs") if tqdm else None

        for caminho_xml in iterable:
            # Tenta encontrar imagem associada (.jpeg, .jpg, maiúsculas)
            caminho_img = None
            for ext in [".jpeg", ".jpg", ".JPEG", ".JPG", ".png"]:
                tentativa = caminho_xml.with_suffix(ext)
                if tentativa.exists():
                    caminho_img = tentativa
                    break

            if caminho_img is None:
                fotos_sem_imagem += 1
                if bar:
                    bar.update(1)
                continue

            try:
                img = Image.open(caminho_img).convert("RGB")
            except Exception:
                fotos_sem_imagem += 1
                if bar:
                    bar.update(1)
                continue

            # Sanitiza a chave de ocorrência substituindo '_' por '-' para evitar data leakage
            base = caminho_img.stem.replace("_", "-")
            idx = 0

            caixas = list(caixas_do_xml(caminho_xml))

            # Caso especial: Imagem em pasta 'healthy' sem caixas de pragas anotadas
            is_healthy_context = "healthy" in str(caminho_xml).lower() or "healthy" in base.lower()
            if not caixas and is_healthy_context:
                classe_destino = "milho_sadio"
                pasta_destino = raiz_saida / classe_destino
                pasta_destino.mkdir(parents=True, exist_ok=True)
                destino = pasta_destino / f"{base}_full.jpg"
                img.save(destino, "JPEG", quality=90)
                w.writerow([
                    str(destino.as_posix()),
                    classe_destino,
                    FONTE,
                    base,
                    URL_FONTE,
                    LICENCA,
                    AUTOR,
                ])
                contagem_classes[classe_destino] = contagem_classes.get(classe_destino, 0) + 1
                total_recortado += 1
                if bar:
                    bar.update(1)
                continue

            for nome_obj, caixa in caixas:
                if args.classe == "auto":
                    classe_destino = OBJETOS_PARA_CLASSE.get(nome_obj)
                    if not classe_destino:
                        continue
                else:
                    classe_destino = args.classe

                xmin, ymin, xmax, ymax = caixa
                if (xmax - xmin) < LADO_MINIMO_PX or (ymax - ymin) < LADO_MINIMO_PX:
                    continue

                recorte = recortar_com_margem(img, caixa)
                if recorte is None:
                    continue

                pasta_destino = raiz_saida / classe_destino
                pasta_destino.mkdir(parents=True, exist_ok=True)

                nome_arquivo = f"{base}_{idx}.jpg"
                destino = pasta_destino / nome_arquivo
                recorte.save(destino, "JPEG", quality=90)

                w.writerow([
                    str(destino.as_posix()),
                    classe_destino,
                    FONTE,
                    base,
                    URL_FONTE,
                    LICENCA,
                    AUTOR,
                ])

                contagem_objetos[nome_obj] = contagem_objetos.get(nome_obj, 0) + 1
                contagem_classes[classe_destino] = contagem_classes.get(classe_destino, 0) + 1
                total_recortado += 1
                idx += 1

            if bar:
                bar.update(1)

        if bar:
            bar.close()

    print("\n" + "=" * 65)
    print(f"[OK] Processamento concluído! Total de amostras recortadas: {total_recortado}")
    print("\nDetalhamento por Classe do Classificador:")
    for cl, n in sorted(contagem_classes.items()):
        print(f"  • {cl}: {n} recortes")

    print("\nDetalhamento por Objeto do KaraAgro:")
    for obj, n in sorted(contagem_objetos.items()):
        print(f"  • {obj}: {n}")

    if fotos_sem_imagem:
        print(f"\n[AVISO] {fotos_sem_imagem} anotações XML sem imagem correspondente (ignoradas).")

    print(f"\nRegistro de proveniência atualizado em: {creditos_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
