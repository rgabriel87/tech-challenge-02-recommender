"""Modelos utilizados pelo sistema de recomendação."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Armazena as configurações da rede neural."""

    num_users: int
    num_items: int
    num_categories: int
    embedding_dim: int
    hidden_dim: int
    dropout: float


class NeuralRecommender(nn.Module):
    """Rede neural de recomendação baseada em embeddings.

    O modelo aprende representações para usuários, produtos e categorias.
    Essas informações são combinadas para estimar a probabilidade de uma
    interação positiva.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Inicializa as camadas da rede neural.

        Args:
            config: Configurações utilizadas para criar o modelo.
        """
        super().__init__()

        self.user_embedding = nn.Embedding(
            config.num_users,
            config.embedding_dim,
        )

        self.item_embedding = nn.Embedding(
            config.num_items,
            config.embedding_dim,
        )

        self.category_embedding = nn.Embedding(
            config.num_categories,
            config.embedding_dim,
        )

        self.network = nn.Sequential(
            nn.Linear(
                config.embedding_dim * 3,
                config.hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Executa a passagem dos dados pela rede.

        Args:
            user_ids: Identificadores dos usuários.
            item_ids: Identificadores dos produtos.
            category_ids: Identificadores das categorias.

        Returns:
            Logits produzidos pela rede neural.
        """
        user_vector = self.user_embedding(user_ids)
        item_vector = self.item_embedding(item_ids)
        category_vector = self.category_embedding(category_ids)

        features = torch.cat(
            [
                user_vector,
                item_vector,
                category_vector,
            ],
            dim=1,
        )

        return self.network(features).squeeze(1)