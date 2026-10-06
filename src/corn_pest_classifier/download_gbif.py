#!/usr/bin/env python3
"""
download_gbif.py — Módulo utilitário para download de ocorrências e fotos de pragas do milho do GBIF.
Permite execução direta ou importação de funções auxiliares pelo pipeline do projeto.
"""

import sys
from pathlib import Path

# Adiciona a raiz do projeto ao path caso necessário
_root_dir = str(Path(__file__).resolve().parent.parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

try:
    # Importa a lógica principal de 01_baixar_gbif.py
    import importlib.util
    _script_path = Path(_root_dir) / "01_baixar_gbif.py"
    _spec = importlib.util.spec_from_file_location("gbif_downloader", _script_path)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        CLASSES_DEFAULT = _mod.CLASSES_DEFAULT
        resolver_taxon_key = _mod.resolver_taxon_key
        buscar_ocorrencias = _mod.buscar_ocorrencias
        baixar_e_validar_imagem = _mod.baixar_e_validar_imagem
        main = _mod.main
except Exception as e:
    print(f"[AVISO] Não foi possível carregar 01_baixar_gbif.py dinamicamente: {e}")


if __name__ == "__main__":
    if "main" in locals():
        main()
    else:
        print("Execute diretamente: python 01_baixar_gbif.py")
