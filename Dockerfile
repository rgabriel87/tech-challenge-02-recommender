# Etapa 1: instalação das dependências
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev


# Etapa 2: imagem final de execução
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN groupadd --system appgroup \
    && useradd --system --gid appgroup appuser

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY params.yaml ./
COPY data ./data

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN mkdir -p models reports \
    && chown -R appuser:appgroup /app

USER appuser

CMD ["train-model"]