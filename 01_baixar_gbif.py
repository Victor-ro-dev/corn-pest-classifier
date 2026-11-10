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
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import os
import sys
import threading
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

GBIF = "https://api.gbif.org/v1"

# Mapeamento das classes do projeto coletadas via GBIF
CLASSES_DEFAULT = {
    "lagarta_cartucho": "Spodoptera frugiperda",
    "lagarta_espiga": "Helicoverpa zea",
    "vaquinha": "Diabrotica speciosa",
    "milho_sadio": "Zea mays",
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


_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Retorna uma sessão HTTP reutilizável por thread (Keep-Alive / Connection Pool)."""
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(HEADERS)
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=2)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _thread_local.session = sess
    return sess


def baixar_e_validar_imagem(
    url: str,
    destino: Path,
    timeout: int = 25,
    max_dim: int = 320,
) -> bool:
    """
    Baixa uma imagem em paralelo, valida sua integridade real usando Pillow e
    redimensiona para no máximo `max_dim`x`max_dim` (JPEG otimizado), mantendo
    margem ideal para a MobileNetV2 (224x224) e reduzindo o peso em ~80x para caber no Git.
    """
    try:
        sess = _get_session()
        r = sess.get(url, timeout=timeout)
        r.raise_for_status()

        ct = r.headers.get("Content-Type", "")
        if ct and "image" not in ct and "octet-stream" not in ct:
            return False

        conteudo = r.content
        if len(conteudo) < 4000:
            return False

        # Validação real e redimensionamento inteligente via Pillow
        try:
            with Image.open(io.BytesIO(conteudo)) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                if img.width < 96 or img.height < 96:
                    return False
                if max_dim and (img.width > max_dim or img.height > max_dim):
                    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85, optimize=True)
                dados_otimizados = buf.getvalue()
        except Exception:
            return False

        destino.write_bytes(dados_otimizados)
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
        description="Download paralelo de fotos de campo de pragas do milho a partir do GBIF"
    )
    parser.add_argument(
        "--max-por-classe",
        type=int,
        default=600,
        help="Quantidade máxima de imagens a serem baixadas por classe (padrão: 600)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Número de threads paralelas para download acelerado (padrão: 24)",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=320,
        help="Dimensão máxima (px) para redimensionar as imagens salvas (padrão: 320, 0 = original)",
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
    print("[*] COLETOR PARALELO DE IMAGENS GBIF - PRAGAS DO MILHO")
    print(f"Diretorio de saida: {raiz_saida.resolve()}")
    print(f"Modo de organizacao: {args.modo}")
    print(f"Limite por classe: {limite} fotos | Workers paralelos: {args.workers}")
    print(f"Redimensionamento automatico: {args.max_dim}x{args.max_dim}px (JPEG 85%)")
    print(f"Licencas permitidas: {licencas}")
    print(f"Registro de autoria: {creditos_path.resolve()}")
    print("=" * 60)

    total_geral_baixadas = 0
    csv_lock = threading.Lock()

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
            ocorrencias = buscar_ocorrencias(key, int(limite * 1.15) + 20, life_stages, licencas)
            print(f"  {len(ocorrencias)} mídias candidatas encontradas.")

            if args.modo == "subpastas":
                pasta_destino = raiz_saida / classe
                pasta_destino.mkdir(parents=True, exist_ok=True)
            else:
                pasta_destino = raiz_saida

            # Conta arquivos já existentes no diretório
            existentes = [
                f for f in pasta_destino.glob("*.jpg") if f.is_file() and f.stat().st_size > 2000
            ]
            ok = len(existentes)
            if ok >= limite:
                print(f"  -> Já existem {ok} imagens válidas em '{pasta_destino}'. Pulando download!")
                continue

            faltam = limite - ok
            tarefas = []
            for i, (occ_key, url, lic, autor) in enumerate(ocorrencias):
                if url in urls_ja_registradas:
                    continue
                if args.modo == "subpastas":
                    nome_arquivo = f"{occ_key}_{i}.jpg"
                else:
                    nome_arquivo = f"{classe}_{occ_key}_{i}.jpg"
                destino = pasta_destino / nome_arquivo
                if destino.exists() and destino.stat().st_size > 2000:
                    continue
                tarefas.append((occ_key, url, lic, autor, destino))

            alvo_barra = min(faltam, len(tarefas))
            bar = tqdm(total=alvo_barra, desc=f"Baixando {classe} ({args.workers} threads)") if tqdm else None
            novas_classe = 0

            def _worker(item):
                occ_key_w, url_w, lic_w, autor_w, destino_w = item
                sucesso = baixar_e_validar_imagem(url_w, destino_w, max_dim=args.max_dim)
                return sucesso, occ_key_w, url_w, lic_w, autor_w, destino_w

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                # Submete em lotes controlados para parar assim que atingir o limite
                futures = [pool.submit(_worker, t) for t in tarefas[: int(faltam * 1.25) + 15]]
                for fut in as_completed(futures):
                    if ok >= limite:
                        for pendente in futures:
                            pendente.cancel()
                        break
                    sucesso, occ_key_r, url_r, lic_r, autor_r, destino_r = fut.result()
                    if sucesso:
                        with csv_lock:
                            if ok < limite:
                                ok += 1
                                novas_classe += 1
                                total_geral_baixadas += 1
                                w.writerow([
                                    str(destino_r.as_posix()),
                                    classe,
                                    "GBIF",
                                    str(occ_key_r),
                                    url_r,
                                    lic_r,
                                    autor_r,
                                ])
                                fcsv.flush()
                                urls_ja_registradas.add(url_r)
                                if bar:
                                    bar.update(1)
                            else:
                                # Excedente pós-limite: remove para manter exatamente o limite pedido
                                try:
                                    destino_r.unlink()
                                except OSError:
                                    pass

            if bar:
                bar.close()

            print(f"  -> Concluído: {ok} imagens prontas para '{classe}' (+{novas_classe} novas).")

    print("\n" + "=" * 60)
    print(f"[OK] Processamento finalizado! Total de novas imagens baixadas: {total_geral_baixadas}")
    print(f"[INFO] Arquivo de auditoria salvo em: {creditos_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
