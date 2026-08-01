"""Factory responsável pela criação dos modelos do projeto."""

from tech_challenge_02_recommender.models import (
    ModelConfig,
    NeuralRecommender,
)


class ModelFactory:
    """Centraliza a criação dos modelos utilizados no projeto."""

    @staticmethod
    def create_neural_model(
        num_users: int,
        num_items: int,
        num_categories: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> NeuralRecommender:
        """Cria a rede neural de recomendação.

        Args:
            num_users: Quantidade de usuários.
            num_items: Quantidade de produtos.
            num_categories: Quantidade de categorias.
            embedding_dim: Dimensão dos embeddings.
            hidden_dim: Quantidade de neurônios da camada oculta.
            dropout: Taxa de dropout.

        Returns:
            Modelo neural configurado.
        """
        config = ModelConfig(
            num_users=num_users,
            num_items=num_items,
            num_categories=num_categories,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        return NeuralRecommender(config)