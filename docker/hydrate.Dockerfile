FROM python:3.10-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY tools/ tools/
COPY data/ data/
RUN pip install --no-cache-dir aiohttp orjson tqdm python-dotenv