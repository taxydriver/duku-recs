FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential gfortran libopenblas-dev   && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir .

COPY serve/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY serve/ ./serve/

ENV MODEL_DIR=/app/artifacts
ENV TOPK_DEFAULT=20

EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
