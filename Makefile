.PHONY: help install install-dev test lint format clean build publish

help:
	@echo "Available commands:"
	@echo "  make install       - Install package"
	@echo "  make install-dev   - Install with dev dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run linting (ruff, black, mypy)"
	@echo "  make format        - Format code with black and ruff"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make build         - Build distribution packages"
	@echo "  make publish       - Publish to PyPI"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,all]"

test:
	pytest

lint:
	ruff check .
	black --check .
	mypy .

format:
	ruff check --fix .
	black .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	twine upload dist/*
