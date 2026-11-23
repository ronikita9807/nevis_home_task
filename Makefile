.DEFAULT_GOAL := help
.PHONY: help install fmt lint typecheck test check

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
