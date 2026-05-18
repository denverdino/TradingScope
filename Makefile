# Makefile for TradingScope

# Default target
.PHONY: help
help:
	@echo "TradingScope Makefile"
	@echo "Usage:"
	@echo "  make install     - Install dependencies with uv"
	@echo "  make lint        - Run linting with ruff"
	@echo "  make lint-fix    - Run linting and resolve violations with ruff"
	@echo "  make format      - Format code with ruff"
	@echo "  make format-check - Check formatting with ruff"
	@echo "  make imports     - Order imports with ruff"
	@echo "  make test        - Run tests with pytest"
	@echo "  make test-cov    - Run tests with coverage report"
	@echo "  make evaluate    - Run post-market analysis evaluation"
	@echo "  make clean       - Clean build artifacts"

# Install dependencies
.PHONY: install
install:
	uv sync --extra dev

# Linting
.PHONY: lint
lint:
	uv run ruff check .

# Strict linting (with auto-fix)
.PHONY: lint-fix
lint-fix:
	uv run ruff check . --fix

# Import ordering
.PHONY: imports
imports:
	uv run ruff check . --select I --fix

# Formatting
.PHONY: format
format:
	uv run ruff format .

# Check formatting
.PHONY: format-check
format-check:
	uv run ruff format --check .

# Testing
.PHONY: test
test:
	uv run pytest

# Testing with coverage
.PHONY: test-cov
test-cov:
	uv run pytest --cov=tradingscope --cov-report=html --cov-report=term

# Post-market evaluation
.PHONY: evaluate
evaluate:
	uv run python -m tradingscope.agents.evaluation.cli

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete