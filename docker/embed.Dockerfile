# docker/embed.Dockerfile
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling (avoids manylinux/ABI hiccups)
RUN python -m pip install --upgrade pip setuptools wheel

# Copy code & IO dirs (volumes will overwrite at runtime)
COPY tools/ tools/
COPY data/ data/
COPY artifacts/ artifacts/

# 1) Install Torch from the PyTorch CPU index
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# 2) Install the rest from the default PyPI index
RUN pip install --no-cache-dir \
    sentence-transformers \
    numpy \
    tqdm \
    orjson

# Default command (overridden by compose)
CMD ["python", "tools/embed_movies.py", "data/movies.ndjson", "artifacts"]