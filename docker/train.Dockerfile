FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential gfortran libopenblas-dev   && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir .

COPY train/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY train/ ./train/

CMD ["python", "train/train_als.py", "--config", "configs/train.yaml"]
