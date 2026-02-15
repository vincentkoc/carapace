# carapace

`carapace` brings order to high-volume repositories with PR/issue fingerprinting, similarity clustering, canonical candidate selection, and low-pass noise filtering.

## Current Capabilities
- Repo-level configuration via `.carapace.yaml`.
- Typed Python core built on Pydantic models.
- SQLite-first persistence (`.carapace/carapace.db`) with adapter boundary for PostgreSQL replacement.
- Offline scan pipeline with:
  - low-pass filtering (`pass` / `suppress` / `skip`)
  - fingerprinting + embeddings
  - similarity edges + clustering
  - canonical ranking in each cluster
  - routing decisions (canonical, duplicate, related, suppressed)
- Report bundle output:
  - `triage_report.md`
  - `clusters.json`
  - `labels_to_apply.json`

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

Run an offline scan:
```bash
carapace scan --input /path/to/entities.json --repo-path /path/to/repo --output-dir ./carapace-out
```

Run directly against GitHub using `gh`:
```bash
carapace scan-github \
  --repo openclaw/openclaw \
  --repo-path /path/to/local/repo \
  --max-prs 300 \
  --include-issues \
  --max-issues 200 \
  --output-dir ./carapace-out \
  --save-input-json
```

Optional routing application:
```bash
# dry run (default)
carapace scan-github --repo openclaw/openclaw --apply-routing

# live GitHub writes (labels/comments)
carapace scan-github --repo openclaw/openclaw --apply-routing --live-actions
```

## Config
Carapace loads configuration from repository root `.carapace.yaml` with precedence:
1. runtime override
2. repo `.carapace.yaml`
3. org defaults
4. system defaults
