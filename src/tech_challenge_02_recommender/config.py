"""Carregamento centralizado das configurações do projeto."""

from pathlib import Path
from typing import Any

import yaml

# Identifica a pasta principal do projeto a partir deste arquivo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define o caminho do arquivo que concentra os parâmetros do projeto.
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params() -> dict[str, Any]:
    """Carrega aqui as configurações definidas no arquivo params.yaml.

    Returns:
        Dicionário contendo as configurações do projeto.

    Raises:
        FileNotFoundError: Quando o arquivo params.yaml não é encontrado.
        ValueError: Quando o arquivo não contém uma estrutura YAML válida.
    """
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"arquivo de configuração não encontrado: {PARAMS_PATH}"
        )

    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    if not isinstance(params, dict):
        raise ValueError(
            "O arquivo params.yaml deve conter uma estrutura válida."
        )

    return params