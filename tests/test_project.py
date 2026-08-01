"""Testes básicos do projeto."""


import pandas as pd
import torch

from tech_challenge_02_recommender.config import PROJECT_ROOT, load_params
from tech_challenge_02_recommender.factory import ModelFactory


def test_load_params() -> None:
    """Valida o carregamento das configurações."""
    params = load_params()

    assert params["project"]["random_seed"] == 42
    assert params["data"]["num_users"] == 500


def test_processed_data_schema() -> None:
    """Valida as colunas e o conteúdo do dataset processado."""
    train_path = PROJECT_ROOT / "data/processed/train.csv"

    assert train_path.exists()

    dataframe = pd.read_csv(train_path)

    assert set(dataframe.columns) == {
        "user_id",
        "item_id",
        "category_id",
        "interaction",
    }
    assert set(dataframe["interaction"].unique()).issubset({0, 1})


def test_model_forward() -> None:
    """Valida uma passagem simples pela rede neural."""
    model = ModelFactory.create_neural_model(
        num_users=500,
        num_items=200,
        num_categories=12,
        embedding_dim=32,
        hidden_dim=64,
        dropout=0.1,
    )

    output = model(
        torch.tensor([0, 1]),
        torch.tensor([10, 20]),
        torch.tensor([2, 4]),
    )

    assert output.shape == torch.Size([2])