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
REPO ?= openclaw/openclaw
REPO_PATH ?= /Users/vincentkoc/GIT/_Perso/openclaw
UI_HOST ?= 127.0.0.1
UI_PORT ?= 8765
SQLITE ?= sqlite3
REPORT_DB ?= .carapace/carapace.db
REPORT_DIR ?= carapace-out/reports
REPORT_LIMIT ?= 50
REPORT_MIN_TOTAL ?= 20
REPORT_MIN_CLOSED14 ?= 10
REPORT_MIN_STALE14 ?= 10

.PHONY: help setup-venv install install-dev install-ui test lint format mypy check precommit-install precommit serve-ui serve-ui-reload reports-sqlite reports-authors-overview reports-authors-closed-stale clean

help:
	@echo "Available targets:"
	@echo "  setup-venv        Create virtual environment in $(VENV)"
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
	@echo "  serve-ui          Run FastAPI UI (repo=$(REPO), path=$(REPO_PATH))"
	@echo "  serve-ui-reload   Run FastAPI UI with hot reload"
	@echo "  reports-sqlite    Generate SQLite author analysis reports in $(REPORT_DIR)"
	@echo "  reports-authors-overview     Author totals + 14d/7d activity with links"
	@echo "  reports-authors-closed-stale High-volume authors with closed/stale signals"
	@echo "  clean             Remove local caches and artifacts"

setup-venv:
	python3 -m venv $(VENV)

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

serve-ui:
	$(call warn_if_no_venv)
	$(PYTHON) -m carapace.cli serve-ui \
		--repo $(REPO) \
		--repo-path $(REPO_PATH) \
		--skip-repo-path-check \
		--host $(UI_HOST) \
		--port $(UI_PORT)

serve-ui-reload:
	$(call warn_if_no_venv)
	$(PYTHON) -m carapace.cli serve-ui \
		--repo $(REPO) \
		--repo-path $(REPO_PATH) \
		--skip-repo-path-check \
		--host $(UI_HOST) \
		--port $(UI_PORT) \
		--reload

reports-sqlite: reports-authors-overview reports-authors-closed-stale

reports-authors-overview:
	@if [ ! -f "$(REPORT_DB)" ]; then echo "[error] Missing SQLite DB: $(REPORT_DB)"; exit 1; fi
	@mkdir -p "$(REPORT_DIR)"
	@$(SQLITE) -separator $$'\t' "$(REPORT_DB)" "WITH bounds AS ( \
		SELECT date(MAX(json_extract(payload_json,'$$.created_at'))) AS max_d, \
		       date(MAX(json_extract(payload_json,'$$.created_at')), '-13 day') AS start_14d \
		FROM ingest_entities \
		WHERE repo='$(REPO)' \
	), base AS ( \
		SELECT lower(json_extract(payload_json,'$$.author')) AS author_lc, \
		       MIN(json_extract(payload_json,'$$.author')) AS author, \
		       COUNT(*) AS total_all, \
		       SUM(CASE WHEN kind='pr' THEN 1 ELSE 0 END) AS prs_all, \
		       SUM(CASE WHEN kind='issue' THEN 1 ELSE 0 END) AS issues_all \
		FROM ingest_entities \
		WHERE repo='$(REPO)' \
		  AND json_extract(payload_json,'$$.author') IS NOT NULL \
		GROUP BY lower(json_extract(payload_json,'$$.author')) \
	), d14 AS ( \
		SELECT lower(json_extract(payload_json,'$$.author')) AS author_lc, \
		       COUNT(*) AS total_14d, \
		       SUM(CASE WHEN kind='pr' THEN 1 ELSE 0 END) AS prs_14d, \
		       SUM(CASE WHEN kind='issue' THEN 1 ELSE 0 END) AS issues_14d \
		FROM ingest_entities \
		WHERE repo='$(REPO)' \
		  AND json_extract(payload_json,'$$.author') IS NOT NULL \
		  AND date(json_extract(payload_json,'$$.created_at')) >= (SELECT start_14d FROM bounds) \
		GROUP BY lower(json_extract(payload_json,'$$.author')) \
	), d7 AS ( \
		SELECT lower(json_extract(payload_json,'$$.author')) AS author_lc, \
		       COUNT(*) AS total_7d \
		FROM ingest_entities \
		WHERE repo='$(REPO)' \
		  AND json_extract(payload_json,'$$.author') IS NOT NULL \
		  AND date(json_extract(payload_json,'$$.created_at')) >= date((SELECT max_d FROM bounds), '-6 day') \
		GROUP BY lower(json_extract(payload_json,'$$.author')) \
	) \
	SELECT b.author, \
	       b.total_all, \
	       b.prs_all, \
	       b.issues_all, \
	       COALESCE(d14.total_14d,0) AS total_14d, \
	       COALESCE(d14.prs_14d,0) AS prs_14d, \
	       COALESCE(d14.issues_14d,0) AS issues_14d, \
	       COALESCE(d7.total_7d,0) AS total_7d, \
	       printf('https://github.com/$(REPO)/issues?q=author%%3A%s+sort%%3Acreated-desc', b.author) AS all_link, \
	       printf('https://github.com/$(REPO)/issues?q=author%%3A%s+created%%3A%s..%s+sort%%3Acreated-desc', b.author, (SELECT start_14d FROM bounds), (SELECT max_d FROM bounds)) AS last14_link \
	FROM base b \
	LEFT JOIN d14 ON d14.author_lc=b.author_lc \
	LEFT JOIN d7 ON d7.author_lc=b.author_lc \
	ORDER BY b.total_all DESC, total_14d DESC, b.author \
	LIMIT $(REPORT_LIMIT);" > "$(REPORT_DIR)/authors_overview.tsv"
	@awk -F'\t' 'BEGIN { \
		print "| Author | Total | PRs | Issues | Last 14d | PRs 14d | Issues 14d | Last 7d | All | All 2w |"; \
		print "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"; \
	} { \
		printf "| %s | %s | %s | %s | %s | %s | %s | %s | [All](%s) | [All 2w](%s) |\n", $$1, $$2, $$3, $$4, $$5, $$6, $$7, $$8, $$9, $$10; \
	}' "$(REPORT_DIR)/authors_overview.tsv" > "$(REPORT_DIR)/authors_overview.md"
	@echo "[ok] Wrote $(REPORT_DIR)/authors_overview.tsv"
	@echo "[ok] Wrote $(REPORT_DIR)/authors_overview.md"

reports-authors-closed-stale:
	@if [ ! -f "$(REPORT_DB)" ]; then echo "[error] Missing SQLite DB: $(REPORT_DB)"; exit 1; fi
	@mkdir -p "$(REPORT_DIR)"
	@$(SQLITE) -separator $$'\t' "$(REPORT_DB)" "WITH rows AS ( \
		SELECT lower(json_extract(payload_json,'$$.author')) AS author_lc, \
		       json_extract(payload_json,'$$.author') AS author, \
		       state, \
		       date(json_extract(payload_json,'$$.created_at')) AS cdate, \
		       date(json_extract(payload_json,'$$.updated_at')) AS udate \
		FROM ingest_entities \
		WHERE repo='$(REPO)' \
		  AND json_extract(payload_json,'$$.author') IS NOT NULL \
	), maxd AS ( \
		SELECT MAX(cdate) AS max_created, MAX(udate) AS max_updated FROM rows \
	), agg AS ( \
		SELECT author_lc, \
		       MIN(author) AS author, \
		       COUNT(*) AS total, \
		       SUM(CASE WHEN state='closed' THEN 1 ELSE 0 END) AS closed_total, \
		       SUM(CASE WHEN state='closed' AND udate >= date((SELECT max_updated FROM maxd), '-13 day') THEN 1 ELSE 0 END) AS closed_14d, \
		       SUM(CASE WHEN state='open' THEN 1 ELSE 0 END) AS open_total, \
		       SUM(CASE WHEN state='open' AND cdate <= date((SELECT max_created FROM maxd), '-14 day') THEN 1 ELSE 0 END) AS stale_open_age14, \
		       SUM(CASE WHEN state='open' AND cdate <= date((SELECT max_created FROM maxd), '-30 day') THEN 1 ELSE 0 END) AS stale_open_age30 \
		FROM rows \
		GROUP BY author_lc \
	) \
	SELECT author, \
	       total, \
	       closed_14d, \
	       closed_total, \
	       open_total, \
	       stale_open_age14, \
	       stale_open_age30, \
	       printf('https://github.com/$(REPO)/issues?q=author%%3A%s+sort%%3Acreated-desc', author) AS all_link, \
	       printf('https://github.com/$(REPO)/issues?q=author%%3A%s+is%%3Aclosed+sort%%3Aupdated-desc', author) AS closed_link, \
	       printf('https://github.com/$(REPO)/issues?q=author%%3A%s+is%%3Aopen+created%%3A..%s+sort%%3Acreated-asc', author, (SELECT date(max_created, '-14 day') FROM maxd)) AS stale_link \
	FROM agg \
	WHERE total >= $(REPORT_MIN_TOTAL) \
	  AND (closed_14d >= $(REPORT_MIN_CLOSED14) OR stale_open_age14 >= $(REPORT_MIN_STALE14)) \
	ORDER BY closed_14d DESC, stale_open_age14 DESC, total DESC \
	LIMIT $(REPORT_LIMIT);" > "$(REPORT_DIR)/authors_closed_stale.tsv"
	@awk -F'\t' 'BEGIN { \
		print "| Author | Total | Closed 14d | Closed Total | Open Total | Stale Open (14d age) | Stale Open (30d age) | All | Closed | Stale Open |"; \
		print "|---|---:|---:|---:|---:|---:|---:|---|---|---|"; \
	} { \
		printf "| %s | %s | %s | %s | %s | %s | %s | [All](%s) | [Closed](%s) | [Stale](%s) |\n", $$1, $$2, $$3, $$4, $$5, $$6, $$7, $$8, $$9, $$10; \
	}' "$(REPORT_DIR)/authors_closed_stale.tsv" > "$(REPORT_DIR)/authors_closed_stale.md"
	@echo "[ok] Wrote $(REPORT_DIR)/authors_closed_stale.tsv"
	@echo "[ok] Wrote $(REPORT_DIR)/authors_closed_stale.md"

clean:
	@rm -rf build dist .pytest_cache .mypy_cache .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
