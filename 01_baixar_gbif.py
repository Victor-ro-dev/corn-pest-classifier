#!/usr/bin/env python3
"""
01_baixar_gbif.py — Coleta e download de imagens de campo de pragas do milho via API GBIF.

Características:
  - Filtro por licenças abertas (CC0, CC-BY e opcionalmente CC-BY-NC).
  - Filtro exclusivo de fotos de campo (HUMAN_OBSERVATION), descartando espécimes de museu.
  - Filtro por estágio larval para lagartas (evita mariposas adultas) e adultos para vaquinha.
  - Validação de integridade real das imagens baixadas via Pillow (evita arquivos corrompidos).
  - Rastreamento completo de proveniência com exportação para creditos.csv.
  - Nomenclatura com chave de ocorrência para permitir split treino/validação sem data leakage.
  - Suporte tanto a organização por subpastas (multiclasse) quanto direto para dataset/praga (binário).
"""

import argparse
import csv
import io
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

GBIF = "https://api.gbif.org/v1"

# Mapeamento das pragas do milho coletadas via GBIF
CLASSES_DEFAULT = {
    "lagarta_cartucho": "Spodoptera frugiperda",
    "lagarta_espiga": "Helicoverpa zea",
    "vaquinha": "Diabrotica speciosa",
}

# Estágios de vida: lagartas são larvas; vaquinha é o besouro adulto
LIFE_STAGE_LARVA = {"lagarta_cartucho", "lagarta_espiga"}
LIFE_STAGES_ACEITOS = ["Larva", "Caterpillar", "Immature"]

# Licenças padrão permitidas para redistribuição comercial/aberta
LICENCAS_PADRAO = ["CC0_1_0", "CC_BY_4_0"]

HEADERS = {
    "User-Agent": "UPX-Facens-CornPestClassifier/1.0 (projeto academico; contato: upx@facens.br)"
}


def resolver_taxon_key(nome_cientifico: str) -> int | None:
    """Consulta a API do GBIF para resolver a chave taxonômica oficial do nome científico."""
    try:
        r = requests.get(
            f"{GBIF}/species/match",
            params={"name": nome_cientifico, "strict": "false"},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()
        key = d.get("usageKey")
        if not key:
            print(f"  [AVISO] Não foi possível resolver '{nome_cientifico}' no GBIF.")
            return None
        print(f"  -> taxonKey={key} ({d.get('scientificName')}, match={d.get('matchType')})")
        return key
    except Exception as e:
        print(f"  [ERRO] Falha ao consultar espécie '{nome_cientifico}': {e}")
        return None


def buscar_ocorrencias(
    taxon_key: int,
    limite: int,
    life_stages: list[str] | None = None,
    licencas: list[str] | None = None,
) -> list[tuple[int | str, str, str, str]]:
    """
    Pagina a busca de ocorrências no GBIF com filtros de imagem e licença.
    Retorna lista de tuplas: (occ_key, url_foto, licenca, autor)
    """
    achados = []
    offset = 0
    page = 300  # Máximo por requisição no GBIF
    lics = licencas or LICENCAS_PADRAO

    while len(achados) < limite:
        params: list[tuple[str, str | int]] = [
            ("taxonKey", taxon_key),
            ("mediaType", "StillImage"),
            ("basisOfRecord", "HUMAN_OBSERVATION"),  # Descarta museus/fundo branco
            ("limit", page),
            ("offset", offset),
        ]
        for l in lics:
            params.append(("license", l))

        if life_stages:
            for s in life_stages:
                params.append(("lifeStage", s))

        try:
            r = requests.get(
                f"{GBIF}/occurrence/search",
                params=params,
                headers=HEADERS,
                timeout=60,
            )
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            print(f"  [ERRO] Falha na busca de ocorrências (offset {offset}): {e}")
            break

        results = d.get("results", [])
        if not results:
            break

        for occ in results:
            occ_key = occ.get("key", "desconhecido")
            for m in occ.get("media", []):
                url = m.get("identifier")
                if not url:
                    continue
                lic = m.get("license") or occ.get("license") or ""
                autor = m.get("rightsHolder") or occ.get("recordedBy") or ""
                achados.append((occ_key, url, lic, autor))

        offset += page
        if d.get("endOfRecords"):
            break
        time.sleep(0.3)  # Intervalo de cortesia para a API pública

    return achados[:limite]


def baixar_e_validar_imagem(url: str, destino: Path, timeout: int = 40) -> bool:
    """
    Baixa uma imagem e valida sua integridade real usando Pillow.
    Descarta downloads com erro, corrompidos ou com menos de 8 KB.
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        r.raise_for_status()

        ct = r.headers.get("Content-Type", "")
        # Checa se o content-type retornado é de imagem (quando disponível)
        if ct and "image" not in ct and "octet-stream" not in ct:
            return False

        conteudo = r.content
        if len(conteudo) < 8000:
            return False

        # Validação da integridade real dos bytes da imagem usando Pillow
        try:
            with Image.open(io.BytesIO(conteudo)) as img:
                img.verify()
        except Exception:
            return False

        destino.write_bytes(conteudo)
        return True
    except Exception:
        if destino.exists():
            try:
                destino.unlink()
            except OSError:
                pass
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download de fotos de campo de pragas do milho a partir do GBIF"
    )
    parser.add_argument(
        "--max-por-classe",
        type=int,
        default=500,
        help="Quantidade máxima de imagens a serem baixadas por classe (padrão: 500)",
    )
    parser.add_argument(
        "--saida",
        default="dataset",
        help="Diretório onde salvar as imagens (padrão: 'dataset', gerando dataset/<classe>/)",
    )
    parser.add_argument(
        "--modo",
        choices=["subpastas", "binario"],
        default="subpastas",
        help=(
            "'subpastas': cria subpastas para cada classe (ex.: dataset/lagarta_cartucho/), "
            "ideal para o classificador multiclasse. 'binario': salva tudo na pasta informada com prefixo."
        ),
    )
    parser.add_argument(
        "--teste",
        action="store_true",
        help="Modo teste: baixa apenas 5 fotos por classe para validação rápida",
    )
    parser.add_argument(
        "--incluir-nc",
        action="store_true",
        help="Inclui licença CC_BY_NC_4_0 (não comercial). Use apenas para treino local e não publique o dataset.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(CLASSES_DEFAULT.keys()),
        help=f"Classes específicas a baixar. Opções disponíveis: {list(CLASSES_DEFAULT.keys())}",
    )
    parser.add_argument(
        "--creditos",
        default=None,
        help="Caminho do arquivo creditos.csv (padrão: <saida>/creditos.csv ou dataset/creditos.csv)",
    )

    args = parser.parse_args()

    licencas = LICENCAS_PADRAO + (["CC_BY_NC_4_0"] if args.incluir_nc else [])
    limite = 5 if args.teste else args.max_por_classe

    raiz_saida = Path(args.saida)
    raiz_saida.mkdir(parents=True, exist_ok=True)

    # Define localização do creditos.csv
    if args.creditos:
        creditos_path = Path(args.creditos)
    else:
        # Se saída for dataset/praga, salva em dataset/creditos.csv para ficar no topo
        if raiz_saida.name == "praga":
            creditos_path = raiz_saida.parent / "creditos.csv"
        else:
            creditos_path = raiz_saida / "creditos.csv"

    creditos_path.parent.mkdir(parents=True, exist_ok=True)
    novo_csv = not creditos_path.exists()

    urls_ja_registradas = set()
    if creditos_path.exists():
        try:
            with creditos_path.open("r", encoding="utf-8") as f_leitura:
                leitor = csv.reader(f_leitura)
                for linha in leitor:
                    if len(linha) > 4:
                        urls_ja_registradas.add(linha[4])  # url_original
        except Exception:
            pass

    # Garante compatibilidade de encoding no console do Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 60)
    print("[*] COLETOR DE IMAGENS GBIF - PRAGAS DO MILHO")
    print(f"Diretorio de saida: {raiz_saida.resolve()}")
    print(f"Modo de organizacao: {args.modo}")
    print(f"Limite por classe: {limite} fotos")
    print(f"Licencas permitidas: {licencas}")
    print(f"Registro de autoria: {creditos_path.resolve()}")
    print("=" * 60)

    total_geral_baixadas = 0

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

        for classe in args.classes:
            nome_sci = CLASSES_DEFAULT.get(classe)
            if not nome_sci:
                print(f"\n[AVISO] Classe desconhecida '{classe}'. Pulando...")
                continue

            print(f"\n=== {classe.upper()} ({nome_sci}) ===")
            key = resolver_taxon_key(nome_sci)
            if not key:
                continue

            life_stages = LIFE_STAGES_ACEITOS if classe in LIFE_STAGE_LARVA else None
            if life_stages:
                print(f"  Filtro de estágio de vida: {life_stages} (foco na larva/lagarta)")
            else:
                print("  Sem filtro de estágio larval (espécie identificada pelo adulto)")

            print("  Buscando ocorrências públicas no GBIF...")
            ocorrencias = buscar_ocorrencias(key, limite, life_stages, licencas)
            print(f"  {len(ocorrencias)} mídias candidatas encontradas.")

            if args.modo == "subpastas":
                pasta_destino = raiz_saida / classe
                pasta_destino.mkdir(parents=True, exist_ok=True)
            else:
                pasta_destino = raiz_saida

            ok = 0
            iterable = enumerate(ocorrencias)
            bar = None
            if tqdm:
                bar = tqdm(total=min(limite, len(ocorrencias)), desc=f"Baixando {classe}")

            for i, (occ_key, url, lic, autor) in iterable:
                if ok >= limite:
                    break

                if url in urls_ja_registradas:
                    continue

                if args.modo == "subpastas":
                    nome_arquivo = f"{occ_key}_{i}.jpg"
                else:
                    nome_arquivo = f"{classe}_{occ_key}_{i}.jpg"

                destino = pasta_destino / nome_arquivo

                if destino.exists() and destino.stat().st_size > 8000:
                    ok += 1
                    if bar:
                        bar.update(1)
                    continue

                if baixar_e_validar_imagem(url, destino):
                    ok += 1
                    total_geral_baixadas += 1
                    w.writerow([
                        str(destino.as_posix()),
                        classe,
                        "GBIF",
                        str(occ_key),
                        url,
                        lic,
                        autor,
                    ])
                    fcsv.flush()
                    urls_ja_registradas.add(url)
                    if bar:
                        bar.update(1)
                elif not bar and (i + 1) % 10 == 0:
                    print(f"    Processadas {i + 1}/{len(ocorrencias)}... ({ok} fotos válidas salvas)")

                time.sleep(0.2)  # Respeito à taxa de requisições

            if bar:
                bar.close()

            print(f"  -> Concluído: {ok} imagens salvas para '{classe}'.")

    print("\n" + "=" * 60)
    print(f"[OK] Processamento finalizado! Total de novas imagens baixadas: {total_geral_baixadas}")
    print(f"[INFO] Arquivo de auditoria salvo em: {creditos_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
