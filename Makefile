# Local development shortcuts.

VENV ?= .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
VENV_PRECOMMIT := $(VENV)/bin/pre-commit

ifeq ($(wildcard $(VENV_PY)),)
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PRECOMMIT ?= pre-commit
else
PYTHON ?= $(VENV_PY)
PIP ?= $(VENV_PIP)
ifneq ($(wildcard $(VENV_PRECOMMIT)),)
PRECOMMIT ?= $(VENV_PRECOMMIT)
else
PRECOMMIT ?= pre-commit
endif
endif

define warn_if_no_venv
@if [ ! -f "$(VENV_PY)" ]; then \
  echo "[warn] Virtual environment $(VENV) not found; using system tools."; \
fi
endef

.DEFAULT_GOAL := help

.PHONY: help setup-venv install install-dev test lint format mypy check precommit-install precommit clean

help:
	@echo "Available targets:"
	@echo "  setup-venv        Create virtual environment in $(VENV)"
	@echo "  install           Install package"
	@echo "  install-dev       Install package with dev dependencies"
	@echo "  test              Run test suite"
	@echo "  lint              Run Ruff checks"
	@echo "  format            Apply Ruff fixes and formatting"
	@echo "  mypy              Run static type checks"
	@echo "  check             Run lint + mypy + tests"
	@echo "  precommit-install Install git pre-commit hooks"
	@echo "  precommit         Run all pre-commit hooks"
	@echo "  clean             Remove local caches and artifacts"

setup-venv:
	python3 -m venv $(VENV)

install:
	$(call warn_if_no_venv)
	$(PIP) install -e .

install-dev:
	$(call warn_if_no_venv)
	$(PIP) install -e ".[dev]"

test:
	$(call warn_if_no_venv)
	$(PYTHON) -m pytest -q

lint:
	$(call warn_if_no_venv)
	$(PYTHON) -m ruff check carapace tests

format:
	$(call warn_if_no_venv)
	$(PYTHON) -m ruff check --fix carapace tests
	$(PYTHON) -m ruff format carapace tests

mypy:
	$(call warn_if_no_venv)
	$(PYTHON) -m mypy --config-file pyproject.toml carapace

check: lint mypy test

precommit-install:
	$(call warn_if_no_venv)
	$(PRECOMMIT) install

precommit:
	$(call warn_if_no_venv)
	$(PRECOMMIT) run --all-files

clean:
	@rm -rf build dist .pytest_cache .mypy_cache .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
