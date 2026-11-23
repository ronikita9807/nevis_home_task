.DEFAULT_GOAL := help
.PHONY: help install fmt lint typecheck test check up down logs experiment eval

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install runtime and dev dependencies
	uv sync

fmt: ## Format code and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

typecheck: ## Run pyright and mypy
	uv run pyright
	uv run mypy

test: ## Run the test suite
	uv run pytest

check: lint typecheck test ## Run everything CI runs

up: ## Build and start db, migrations and api (http://localhost:8000/docs)
	docker compose up --build -d --wait

down: ## Stop containers (keeps the database volume)
	docker compose down

logs: ## Follow api logs
	docker compose logs -f api

experiment: ## Compare embedding models on the golden set (add ARGS=--extended for large models)
	uv run python scripts/embedding_experiment.py $(ARGS)

eval: ## Search quality on the golden set through the whole pipeline (recall@5, MRR)
	uv run pytest tests/search/test_golden_set.py -q -s -k recall
