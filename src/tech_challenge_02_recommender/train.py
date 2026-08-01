"""Treinamento da rede neural utilizada no sistema de recomendação."""

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from tech_challenge_02_recommender.config import PROJECT_ROOT, load_params
from tech_challenge_02_recommender.data import configure_logging, set_seed
from tech_challenge_02_recommender.factory import ModelFactory
from tech_challenge_02_recommender.models import NeuralRecommender

LOGGER = logging.getLogger(__name__)


class RecommendationDataset(Dataset):
    """Representa as interações utilizadas pelo PyTorch."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Converte as colunas do DataFrame em tensores.

        Args:
            dataframe: Dados contendo usuários, produtos e interações.
        """
        self.users = torch.tensor(
            dataframe["user_id"].to_numpy(),
            dtype=torch.long,
        )
        self.items = torch.tensor(
            dataframe["item_id"].to_numpy(),
            dtype=torch.long,
        )
        self.categories = torch.tensor(
            dataframe["category_id"].to_numpy(),
            dtype=torch.long,
        )
        self.targets = torch.tensor(
            dataframe["interaction"].to_numpy(),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        """Retorna a quantidade de interações disponíveis."""
        return len(self.targets)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna uma interação individual."""
        return (
            self.users[index],
            self.items[index],
            self.categories[index],
            self.targets[index],
        )


def create_dataloader(
    dataframe: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Cria o carregador de dados utilizado no treinamento.

    Args:
        dataframe: Dados que serão carregados.
        batch_size: Quantidade de registros por lote.
        shuffle: Indica se os dados serão embaralhados.

    Returns:
        DataLoader preparado para o PyTorch.
    """
    dataset = RecommendationDataset(dataframe)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def run_epoch(
    model: NeuralRecommender,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam | None = None,
) -> float:
    """Executa uma época de treino ou validação.

    Quando o otimizador é informado, os pesos da rede são atualizados.
    Sem otimizador, a função apenas calcula a perda de validação.

    Args:
        model: Rede neural utilizada.
        dataloader: Lotes de dados.
        criterion: Função de perda.
        optimizer: Otimizador utilizado no treinamento.

    Returns:
        Perda média da época.
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    for user_ids, item_ids, category_ids, targets in dataloader:
        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            predictions = model(user_ids, item_ids, category_ids,)
            loss = criterion(predictions, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(targets)

    return total_loss / len(dataloader.dataset)


def calculate_metrics(
    model: NeuralRecommender,
    dataloader: DataLoader,
) -> dict[str, float]:
    """Calcula as métricas do modelo no conjunto de teste.

    Args:
        model: Modelo treinado.
        dataloader: Dados usados para avaliação.

    Returns:
        Dicionário contendo as métricas de classificação.
    """
    model.eval()

    probabilities: list[float] = []
    targets: list[int] = []

    with torch.no_grad():
        for user_ids, item_ids, category_ids, batch_targets in dataloader:

            logits = model(
                user_ids,
                item_ids,
                category_ids,
            )

            batch_probabilities = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
            )

            probabilities.extend(batch_probabilities.tolist())
            targets.extend(
                batch_targets.int().cpu().numpy().tolist()
            )

    predicted_classes = [
        1 if probability >= 0.5 else 0
        for probability in probabilities
    ]

    return {
                "accuracy": float(
            accuracy_score(targets, predicted_classes)
        ),
        "precision": float(
            precision_score(
                targets,
                predicted_classes,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                targets,
                predicted_classes,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                targets,
                predicted_classes,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(targets, probabilities)
        ),
    }


def save_artifacts(
    model: NeuralRecommender,
    metrics: dict[str, float],
    params: dict[str, Any],
) -> tuple[Path, Path]:
    """Salva o modelo treinado e os seus metadados.

    Args:
        model: Modelo treinado.
        metrics: Métricas obtidas na avaliação.
        params: Configurações utilizadas.

    Returns:
        Caminhos do modelo e do arquivo de metadados.
    """
    model_path = (
        PROJECT_ROOT / params["artifacts"]["model_path"]
    )
    metadata_path = (
        PROJECT_ROOT / params["artifacts"]["metadata_path"]
    )
    metrics_path = (
        PROJECT_ROOT / params["artifacts"]["metrics_path"]
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)

    metadata = {
        "model_type": params["model"]["type"],
        "num_users": int(params["data"]["num_users"]),
        "num_items": int(params["data"]["num_items"]),
        "embedding_dim": int(params["model"]["embedding_dim"]),
        "hidden_dim": int(params["model"]["hidden_dim"]),
        "dropout": float(params["model"]["dropout"]),
        "metrics": metrics,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    return model_path, metadata_path


def train_model() -> dict[str, float]:
    """Treina, avalia e salva a rede neural de recomendação.

    Returns:
        Métricas finais calculadas no conjunto de teste.
    """
    params = load_params()

    seed = int(params["project"]["random_seed"])
    set_seed(seed)
    torch.manual_seed(seed)

    train_path = (
        PROJECT_ROOT / params["data"]["train_path"]
    )
    test_path = (
        PROJECT_ROOT / params["data"]["test_path"]
    )

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    batch_size = int(params["training"]["batch_size"])

    train_loader = create_dataloader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = create_dataloader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    model = ModelFactory.create_neural_model(
    num_users=int(params["data"]["num_users"]),
    num_items=int(params["data"]["num_items"]),
    num_categories=int(params["data"]["num_categories"]),
    embedding_dim=int(params["model"]["embedding_dim"]),
    hidden_dim=int(params["model"]["hidden_dim"]),
    dropout=float(params["model"]["dropout"]),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=float(params["training"]["learning_rate"]),
    )

    epochs = int(params["training"]["epochs"])
    patience = int(params["training"]["patience"])

    best_validation_loss = np.inf
    best_model_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    # Usa um banco SQLite local para armazenar os experimentos do MLflow.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(
        params["mlflow"]["experiment_name"]
    )

    with mlflow.start_run():
        mlflow.log_params(

            {
                "embedding_dim": int(
                    params["model"]["embedding_dim"]
                ),
                "hidden_dim": int(
                    params["model"]["hidden_dim"]
                ),
                "dropout": float(params["model"]["dropout"]),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": float(
                    params["training"]["learning_rate"]
                ),
                "patience": patience,
                "random_seed": seed,
            }
        )

        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
            )

            validation_loss = run_epoch(
                model=model,
                dataloader=test_loader,
                criterion=criterion,
            )

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )
            mlflow.log_metric(
                "validation_loss",
                validation_loss,
                step=epoch,
            )

            LOGGER.info(
                "Época %s/%s | perda treino=%.4f "
                "| perda validação=%.4f",
                epoch,
                epochs,
                train_loss,
                validation_loss,
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                LOGGER.info(
                    "Early stopping ativado na época %s.",
                    epoch,
                )
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        metrics = calculate_metrics(model, test_loader)
        model_path, metadata_path = save_artifacts(
            model=model,
            metrics=metrics,
            params=params,
        )

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metadata_path))

    LOGGER.info("Treinamento concluído | métricas=%s", metrics)

    return metrics


def main_train() -> None:
    """Executa o treinamento por meio do terminal."""
    configure_logging()

    metrics = train_model()

    print("\nMétricas finais:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main_train()