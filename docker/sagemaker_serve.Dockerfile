# docker/sagemaker_serve.Dockerfile
FROM --platform=linux/amd64 python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

# Python deps
COPY serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference app
COPY serve/ /opt/program/serve/

# Add SageMaker-compatible entrypoint
RUN echo '#!/bin/bash\nexec uvicorn serve.app_sagemaker:app --host 0.0.0.0 --port 8080' \
    > /usr/bin/serve && chmod +x /usr/bin/serve

# Port for SageMaker
ENV PORT=8080
EXPOSE 8080

# SageMaker will invoke /usr/bin/serve by default