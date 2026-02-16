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
GIT_REMOTE_REPO := $(shell git config --get remote.origin.url 2>/dev/null | sed -e 's|^git@github.com:||' -e 's|^https://github.com/||' -e 's|\.git$$||')
REPO ?= $(if $(GIT_REMOTE_REPO),$(GIT_REMOTE_REPO),owner/repo)
REPO_PATH ?= $(CURDIR)
REPO_NAME ?= $(notdir $(REPO))
OUTPUT_DIR ?= $(REPO_PATH)/carapace-out/$(REPO_NAME)
MAX_PRS ?= 0
MAX_ISSUES ?= 0
ENRICH_MODE ?= minimal
ENRICH_WORKERS ?= 8
UI_HOST ?= 127.0.0.1
UI_PORT ?= 8765

.PHONY: help bootstrap setup-venv install install-dev install-ui test lint format mypy check precommit-install precommit ingest process enrich audit triage serve serve-reload serve-ui serve-ui-reload clean

help:
	@echo "Available targets:"
	@echo "  setup-venv        Create virtual environment in $(VENV)"
	@echo "  bootstrap         Setup venv + dev/ui deps + pre-commit hooks"
	@echo "  install           Install package"
	@echo "  install-dev       Install package with dev dependencies"
	@echo "  install-ui        Install package with dev + UI dependencies"
	@echo "  test              Run test suite"
	@echo "  lint              Run Ruff checks"
	@echo "  format            Apply Ruff fixes and formatting"
	@echo "  mypy              Run static type checks"
	@echo "  check             Run lint + mypy + tests"
	@echo "  precommit-install Install git pre-commit hooks"
	@echo "  precommit         Run all pre-commit hooks"
	@echo "  ingest            Resumable GitHub ingest into SQLite"
	@echo "  enrich            Enrich missing PR details in SQLite"
	@echo "  process           Process stored entities and emit triage report bundle"
	@echo "  triage            One-shot GitHub triage run (ingest + process)"
	@echo "  audit             Print SQLite ingest audit summary"
	@echo "  serve             Run FastAPI UI (repo=$(REPO), path=$(REPO_PATH))"
	@echo "  serve-reload      Run FastAPI UI with hot reload"
	@echo "  serve-ui          Compatibility alias for serve"
	@echo "  serve-ui-reload   Compatibility alias for serve-reload"
	@echo "  clean             Remove local caches and artifacts"

setup-venv:
	python3 -m venv $(VENV)

bootstrap: setup-venv install-ui precommit-install

install:
	$(call warn_if_no_venv)
	$(PIP) install -e .

install-dev:
	$(call warn_if_no_venv)
	$(PIP) install -e ".[dev]"

install-ui:
	$(call warn_if_no_venv)
	$(PIP) install -e ".[dev,ui]"

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

ingest:
	$(call warn_if_no_venv)
	$(PYTHON) -m carapace.cli ingest \
		--repo $(REPO) \
		--repo-path $(REPO_PATH) \
		--skip-repo-path-check \
		--max-prs $(MAX_PRS) \
		--max-issues $(MAX_ISSUES)

enrich:
	$(call warn_if_no_venv)
	$(PYTHON) -m carapace.cli enrich \
		--repo $(REPO) \
		--repo-path $(REPO_PATH) \
		--skip-repo-path-check \
		--enrich-mode $(ENRICH_MODE) \
		--enrich-workers $(ENRICH_WORKERS)

process:
	$(call warn_if_no_venv)
	$(PYTHON) -m carapace.cli process \
		--repo $(REPO) \
		--repo-path $(REPO_PATH) \
		--skip-repo-path-check \
		--output-dir $(OUTPUT_DIR) \
		--enrich-missing \
		--enrich-mode $(ENRICH_MODE) \
		--enrich-workers $(ENRICH_WORKERS)

triage:
	$(call warn_if_no_venv)
	$(PYTHON) -m carapace.cli triage \
		--repo $(REPO) \
		--repo-path $(REPO_PATH) \
		--skip-repo-path-check \
		--output-dir $(OUTPUT_DIR) \
		--max-prs $(MAX_PRS) \
		--max-issues $(MAX_ISSUES)

audit:
	$(call warn_if_no_venv)
	$(PYTHON) -m carapace.cli audit \
		--repo $(REPO) \
		--repo-path $(REPO_PATH) \
		--skip-repo-path-check

serve:
	$(call warn_if_no_venv)
	$(PYTHON) -m carapace.cli serve \
		--repo $(REPO) \
		--repo-path $(REPO_PATH) \
		--skip-repo-path-check \
		--host $(UI_HOST) \
		--port $(UI_PORT)

serve-reload:
	$(call warn_if_no_venv)
	$(PYTHON) -m carapace.cli serve \
		--repo $(REPO) \
		--repo-path $(REPO_PATH) \
		--skip-repo-path-check \
		--host $(UI_HOST) \
		--port $(UI_PORT) \
		--reload

serve-ui: serve

serve-ui-reload: serve-reload

clean:
	@rm -rf build dist .pytest_cache .mypy_cache .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
