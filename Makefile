PY=python3
DC=docker compose

install:
	$(PY) -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e .

# Build docker images
build:
	$(DC) build

# Run training job inside container
train:
	$(DC) run --rm trainer

# Launch serving API (after training artifacts exist)
serve:
	$(DC) up api

# Stop serving API
down:
	$(DC) down

# Tail logs for API
logs:
	$(DC) logs -f api
