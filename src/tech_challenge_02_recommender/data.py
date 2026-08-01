"""Geração e preparação dos dados do sistema de recomendação."""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tech_challenge_02_recommender.config import (
    PROJECT_ROOT,
    load_params,
)

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configura os logs exibidos durante a execução do projeto."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def set_seed(seed: int) -> None:
    """Fixa as sementes aleatórias para permitir reprodutibilidade.

    Args:
        seed: Número utilizado como semente pelo Python e pelo NumPy.
    """
    random.seed(seed)
    np.random.seed(seed)


def generate_interactions() -> pd.DataFrame:
    """Gera interações sintéticas entre usuários e produtos.

    Cada usuário recebe interações positivas com produtos pertencentes
    às suas categorias preferidas. Também são geradas interações
    negativas com produtos não selecionados.

    Returns:
        DataFrame com usuários, produtos, categorias e interações.
    """
    params = load_params()

    seed = int(params["project"]["random_seed"])
    num_users = int(params["data"]["num_users"])
    num_items = int(params["data"]["num_items"])
    interactions_per_user = int(
        params["data"]["interactions_per_user"]
    )

    set_seed(seed)

    # Associa cada produto a uma entre doze categorias disponíveis.
    item_categories = np.random.randint(
        low=0,
        high=12,
        size=num_items,
    )

    records: list[dict[str, int]] = []

    for user_id in range(num_users):
        # Cada usuário possui três categorias de maior preferência.
        preferred_categories = np.random.choice(
            12,
            size=3,
            replace=False,
        )

        positive_candidates = [
            item_id
            for item_id in range(num_items)
            if item_categories[item_id] in preferred_categories
        ]

        positive_count = min(
            interactions_per_user,
            len(positive_candidates),
        )

        positive_items = np.random.choice(
            positive_candidates,
            size=positive_count,
            replace=False,
        )

        negative_candidates = list(
            set(range(num_items)) - set(positive_items)
        )

        negative_items = np.random.choice(
            negative_candidates,
            size=positive_count,
            replace=False,
        )

        for item_id in positive_items:
            records.append(
                {
                    "user_id": user_id,
                    "item_id": int(item_id),
                    "category_id": int(
                        item_categories[item_id]
                    ),
                    "interaction": 1,
                }
            )

        for item_id in negative_items:
            records.append(
                {
                    "user_id": user_id,
                    "item_id": int(item_id),
                    "category_id": int(
                        item_categories[item_id]
                    ),
                    "interaction": 0,
                }
            )

    return pd.DataFrame(records)


def generate_raw_data() -> Path:
    """Gera o dataset bruto e salva o resultado em um arquivo CSV.

    Returns:
        Caminho do arquivo CSV gerado.
    """
    params = load_params()

    output_path = (
        PROJECT_ROOT / params["data"]["raw_path"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = generate_interactions()
    dataframe.to_csv(output_path, index=False)

    LOGGER.info(
        "Dataset bruto salvo | caminho=%s | linhas=%s "
        "| usuários=%s | produtos=%s",
        output_path,
        len(dataframe),
        dataframe["user_id"].nunique(),
        dataframe["item_id"].nunique(),
    )

    return output_path


def prepare_data() -> tuple[Path, Path]:
    """Separa o dataset bruto em conjuntos de treino e teste.

    A divisão utiliza estratificação para manter a proporção entre
    interações positivas e negativas nos dois conjuntos.

    Returns:
        Caminhos dos arquivos CSV de treino e teste.

    Raises:
        FileNotFoundError: Quando o dataset bruto ainda não foi gerado.
    """
    params = load_params()

    raw_path = (
        PROJECT_ROOT / params["data"]["raw_path"]
    )
    train_path = (
        PROJECT_ROOT / params["data"]["train_path"]
    )
    test_path = (
        PROJECT_ROOT / params["data"]["test_path"]
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset bruto não encontrado: {raw_path}. "
            "Execute primeiro o comando generate-data."
        )

    dataframe = pd.read_csv(raw_path)

    train_data, test_data = train_test_split(
        dataframe,
        test_size=float(params["data"]["test_size"]),
        random_state=int(
            params["project"]["random_seed"]
        ),
        stratify=dataframe["interaction"],
    )

    train_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    test_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    LOGGER.info(
        "Dados preparados | linhas de treino=%s "
        "| linhas de teste=%s",
        len(train_data),
        len(test_data),
    )

    return train_path, test_path


def main_generate() -> None:
    """Executa a geração do dataset bruto pelo terminal."""
    configure_logging()
    generate_raw_data()


def main_prepare() -> None:
    """Executa a preparação dos dados pelo terminal."""
    configure_logging()
    prepare_data()


if __name__ == "__main__":
    configure_logging()
    generate_raw_data()
    prepare_data()