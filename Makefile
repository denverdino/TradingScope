# Makefile for TradingScope

# Variables
PYTHON := python3
PIP := pip3

# Default target
.PHONY: help
help:
	@echo "TradingScope Makefile"
	@echo "Usage:"
	@echo "  make install     - Install dependencies"
	@echo "  make lint        - Run linting with ruff"
	@echo "  make lint-fix    - Run linting and resolve violations with ruff"
	@echo "  make format      - Format code with ruff"
	@echo "  make format-check - Check formatting with ruff"
	@echo "  make imports     - Order imports with ruff"
	@echo "  make test        - Run tests with pytest"
	@echo "  make test-cov    - Run tests with coverage report"
	@echo "  make clean       - Clean build artifacts"

# Install dependencies
.PHONY: install
install:
	$(PIP) install -e ".[dev]"

# Linting
.PHONY: lint
lint:
	ruff check .

# Strict linting (with auto-fix)
.PHONY: lint-fix
lint-fix:
	ruff check . --fix

# Import ordering
.PHONY: imports
imports:
	ruff check . --select I --fix

# Formatting
.PHONY: format
format:
	ruff format .

# Check formatting
.PHONY: format-check
format-check:
	ruff format --check .

# Testing
.PHONY: test
test:
	pytest

# Testing with coverage
.PHONY: test-cov
test-cov:
	pytest --cov=tradingscope --cov-report=html --cov-report=term

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