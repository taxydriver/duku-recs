FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# OS deps (tiny)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY serve/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Bake ONLY torch weights (no ONNX)
RUN python - <<'PY'
from transformers import AutoModel, AutoTokenizer
name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tok = AutoTokenizer.from_pretrained(name)
mdl = AutoModel.from_pretrained(name)
tok.save_pretrained("/app/minilm")
mdl.save_pretrained("/app/minilm")
print("Saved MiniLM torch weights to /app/minilm")
PY

# Clean caches + optional heavy deps
RUN rm -rf /root/.cache/huggingface /root/.cache/torch /usr/local/share/.cache \
 && python - <<'PY'
import sys, subprocess
for pkg in ("sympy","networkx"):
    try:
        __import__(pkg)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", pkg], check=False)
    except ImportError:
        pass
PY

COPY serve/ ./serve/

ENV EMBED_MODEL=/app/minilm \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]