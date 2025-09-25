#!/usr/bin/env bash
# duku.sh — one-shot build → train → serve for Duku Recs
# Usage: ./duku.sh [--rebuild] [--ratings data/ml-25m/ratings.csv] [--artifacts artifacts/als/0.0.1]

set -euo pipefail

# ---------- configurable bits ----------
SERVE_SERVICE="serve"         # name of the serving service in docker-compose.yml
TRAIN_SERVICE="train"         # name of the training service in docker-compose.yml
PY="python3"                  # python inside the train container
RATINGS="${RATINGS:-data/ml-25m/ratings.csv}"
ARTIFACTS_OUT="${ARTIFACTS_OUT:-artifacts/als/}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"  # change if your API uses a different port/path
# --------------------------------------

REBUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild) REBUILD=1; shift ;;
    --ratings) RATINGS="$2"; shift 2 ;;
    --artifacts) ARTIFACTS_OUT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# pick docker compose CLI
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif docker-compose version >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  echo "❌ Docker Compose not found. Install Docker Desktop or docker-compose."
  exit 1
fi

echo "🧩 Using: $COMPOSE"
echo "📄 Ratings: $RATINGS"
echo "📦 Artifacts out: $ARTIFACTS_OUT"
echo

# 1) Build images
if [[ $REBUILD -eq 1 ]]; then
  echo "🔨 Rebuilding images (no cache)…"
  $COMPOSE build --no-cache
else
  echo "🔨 Building images…"
  $COMPOSE build
fi
echo

# 2) Ensure services that need to be up (e.g., DB) are running
#    If you only have serve/train, this will just start what's defined.
echo "🚀 Bringing up dependencies (detached)…"
$COMPOSE up -d
echo

# 3) Train model (runs once and exits)
echo "🎓 Training ALS on MovieLens…"
# Adjust the module/args to match your train entrypoint if different
echo "🎓 Training ALS using config…"
$COMPOSE run --rm "$TRAIN_SERVICE" \
  $PY train/train_als.py \
    --config configs/als.yaml
echo "✅ Training finished. Artifacts at: $ARTIFACTS_OUT"
echo

# 4) Start/Restart serving API with the new artifacts
echo "🌐 Starting serving API…"
$COMPOSE up -d "$SERVE_SERVICE"

# 5) Optional health wait
echo "⏳ Waiting for health: $HEALTH_URL"
for i in {1..30}; do
  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "✅ Serve is healthy."
    break
  fi
  sleep 1
  if [[ $i -eq 30 ]]; then
    echo "⚠️  Could not confirm health; check logs below."
  fi
done
echo

# 6) Show where to look & tail logs
echo "📜 Recent logs (Ctrl+C to stop tailing):"
$COMPOSE logs -f --tail=200 "$SERVE_SERVICE"