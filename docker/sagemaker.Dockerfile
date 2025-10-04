# docker/sagemaker.Dockerfile  (TRAIN IMAGE)
FROM --platform=linux/amd64 python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran libopenblas-dev libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV SM_MODEL_DIR=/opt/ml/model
ENV SM_CHANNEL_TRAIN=/opt/ml/input/data/train
WORKDIR /opt/program

COPY train/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train/train_sagemaker_lightfm.py /opt/program/train_sagemaker_lightfm.py

# 👇 This is the important bit
ENTRYPOINT ["python","/opt/program/train_sagemaker_lightfm.py"]