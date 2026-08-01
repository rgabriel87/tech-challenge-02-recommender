# Tech Challenge 02 — Sistema de Recomendação para E-commerce

Projeto desenvolvido para o Tech Challenge da Fase 2 da Pós Tech FIAP em Machine Learning Engineering.

A solução implementa um pipeline reproduzível para treinamento de um sistema de recomendação com rede neural em PyTorch, utilizando `uv`, DVC, MLflow, Docker, testes automatizados e boas práticas de engenharia de software.

## 1. Objetivo

O objetivo é estimar a probabilidade de interação entre um usuário e um produto em um cenário de e-commerce.

O modelo utiliza informações de:

- usuário;
- produto;
- categoria do produto;
- histórico de interações positivas e negativas.

## 2. Arquitetura da solução

```text
Geração de dados
       ↓
Preparação e divisão estratificada
       ↓
Dataset e DataLoader PyTorch
       ↓
Rede neural com embeddings
       ↓
Treinamento com early stopping
       ↓
Avaliação e métricas
       ↓
MLflow e artefatos
```

O pipeline é orquestrado pelo DVC:

```text
generate → prepare → train
```

## 3. Tecnologias

- Python 3.11
- PyTorch
- Scikit-Learn
- Pandas e NumPy
- MLflow
- DVC
- uv
- Docker
- Pytest
- Ruff

## 4. Estrutura do projeto

```text
tech-challenge-02-recommender/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── reports/
├── src/
│   └── tech_challenge_02_recommender/
│       ├── config.py
│       ├── data.py
│       ├── factory.py
│       ├── models.py
│       └── train.py
├── tests/
│   └── test_project.py
├── .dvc/
├── .dockerignore
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

## 5. Organização do código

O projeto utiliza separação de responsabilidades:

- `config.py`: leitura centralizada das configurações;
- `data.py`: geração e preparação dos dados;
- `models.py`: definição da rede neural;
- `factory.py`: criação do modelo por meio do Factory Pattern;
- `train.py`: treinamento, avaliação, MLflow e salvamento dos artefatos.

Os parâmetros ficam centralizados em `params.yaml`.

## 6. Configuração do ambiente

Com o `uv` instalado, execute:

```bash
uv sync --frozen
```

O arquivo `uv.lock` mantém as versões das dependências utilizadas no projeto.

## 7. Execução manual

Gerar o dataset:

```bash
uv run generate-data
```

Preparar os conjuntos de treino e teste:

```bash
uv run prepare-data
```

Treinar o modelo:

```bash
uv run train-model
```

## 8. Pipeline DVC

Para reproduzir todas as etapas:

```bash
uv run dvc repro
```

Para visualizar o grafo:

```bash
uv run dvc dag
```

O DVC executa apenas as etapas impactadas por alterações nos dados, no código ou nos parâmetros.

## 9. MLflow

Os experimentos são armazenados em um banco SQLite local:

```text
mlflow.db
```

Para abrir a interface:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Acesse:

```text
http://127.0.0.1:5000
```

O treinamento registra:

- hiperparâmetros;
- perdas de treino e validação;
- métricas finais;
- modelo treinado;
- metadados do modelo.

## 10. Rede neural

O modelo utiliza embeddings para representar:

- usuários;
- produtos;
- categorias.

Os vetores são concatenados e enviados para uma rede neural com:

- camada oculta;
- ativação ReLU;
- dropout;
- saída binária;
- `BCEWithLogitsLoss`;
- otimizador Adam;
- treinamento em batches;
- early stopping.

A criação do modelo foi centralizada em uma Factory para reduzir acoplamento e facilitar futuras extensões.

## 11. Resultados

O treinamento foi interrompido pelo early stopping na época 23.

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.8190 |
| Precision | 0.7965 |
| Recall | 0.8570 |
| F1-score | 0.8256 |
| ROC-AUC | 0.8748 |

O ROC-AUC de aproximadamente `0.87` indica boa capacidade de separar interações positivas e negativas no conjunto avaliado.

## 12. Testes e qualidade

Executar os testes:

```bash
uv run pytest
```

Executar o lint:

```bash
uv run ruff check .
```

Foram incluídos testes para:

- leitura das configurações;
- schema dos dados processados;
- passagem de dados pela rede neural.

## 13. Docker

Construir a imagem:

```bash
docker build -t tech-challenge-02-recommender:1.0.0 .
```

Executar o treinamento no container:

```bash
docker run --rm tech-challenge-02-recommender:1.0.0
```

O Dockerfile utiliza build multi-stage e usuário sem privilégios administrativos.

## 14. Reprodutibilidade

A solução combina:

```text
Git        → código
uv         → dependências e ambiente
DVC        → dados e pipeline
MLflow     → experimentos e métricas
Docker     → ambiente de execução
params.yaml → parâmetros
seeds      → aleatoriedade controlada
```

## 15. Limitações

- O dataset utilizado é sintético e representa um cenário acadêmico.
- O modelo não foi avaliado com métricas específicas de ranking, como Precision@K e Recall@K.
- O MLflow está configurado localmente.
- Não foi realizado deploy público.

## 16. Próximos passos

- utilizar dados reais de e-commerce;
- implementar recomendações Top-K;
- adicionar Precision@K, Recall@K e NDCG;
- registrar e promover modelos no Model Registry;
- configurar armazenamento remoto do DVC;
- automatizar o pipeline com CI/CD;
- realizar deploy em nuvem.