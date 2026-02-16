# carapace

`carapace` brings order to high-volume repositories with PR/issue fingerprinting, similarity clustering, canonical candidate selection, and low-pass noise filtering.

Command API note: legacy command names (`scan-github`, `ingest-github`, `process-stored`, `enrich-stored`, `db-audit`, `serve-ui`) remain supported as aliases.

## Current Capabilities
- Repo-level configuration via `.carapace.yaml`.
- Typed Python core built on Pydantic models.
- SQLite-first persistence (`.carapace/carapace.db`) with adapter boundary for PostgreSQL replacement.
- Offline scan pipeline with:
  - low-pass filtering (`pass` / `suppress` / `skip`)
  - fingerprinting + embeddings
  - similarity edges + clustering
  - advanced similarity algorithms:
    - MinHash + LSH candidate retrieval
    - SimHash candidate retrieval/similarity
    - Winnowing fingerprints
  - canonical ranking in each cluster
  - routing decisions (canonical, duplicate, related, suppressed)
- Split ingest/process workflow for large repos:
  - `ingest` loads entities into SQLite with resumable state
  - `process` processes from persisted ingest snapshot
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
carapace triage \
  --repo openclaw/openclaw \
  --repo-path /path/to/local/repo \
  --max-prs 300 \
  --include-issues \
  --max-issues 200 \
  --output-dir ./carapace-out \
  --save-input-json
```

Stateful large-repo workflow (recommended for 3k+ PRs):
```bash
# 1) Ingest into SQLite (resumable)
carapace ingest \
  --repo openclaw/openclaw \
  --repo-path /path/to/local/openclaw \
  --max-prs 0 \
  --max-issues 0

# 2) Process from stored ingest snapshot
carapace process \
  --repo openclaw/openclaw \
  --repo-path /path/to/local/openclaw \
  --entity-kind pr \
  --output-dir ./carapace-out/openclaw

# 3) Audit ingest DB quality/integrity quickly
carapace audit \
  --repo openclaw/openclaw \
  --repo-path /path/to/local/openclaw
```

Fast enrichment during processing:
```bash
carapace --log-level INFO process \
  --repo openclaw/openclaw \
  --repo-path /path/to/local/openclaw \
  --output-dir ./carapace-out/openclaw \
  --entity-kind pr \
  --enrich-missing \
  --enrich-mode minimal \
  --enrich-workers 8 \
  --enrich-progress-every 100 \
  --enrich-flush-every 50 \
  --enrich-heartbeat-seconds 10
```

Notes:
- Enrichment now logs progress at INFO with rate and ETA.
- Enrichment writes back to SQLite in batches (`--enrich-flush-every`) so interrupted runs can resume without losing all progress.
- `minimal` enrichment uses a fast files-only API path and records watermarks by PR `updated_at`.
- Processing now reuses a warm fingerprint cache in SQLite (keyed by repo/entity/model/updated_at), so repeat runs avoid recomputing unchanged embeddings/fingerprints.

Enrich-only workflow (split from scan):
```bash
carapace --log-level INFO enrich \
  --repo openclaw/openclaw \
  --repo-path /path/to/local/openclaw \
  --entity-kind pr \
  --enrich-mode minimal \
  --enrich-workers 8
```

Enrichment watermark behavior:
- Enrichment is tracked per PR by `updated_at` revision.
- Re-processing skips PR enrichment when the stored watermark already matches current `updated_at`.
- Full ingest does not erase enriched payloads when PR revision has not changed.

Lightweight graph UI (FastAPI + HTMX + Alpine + Cytoscape):
```bash
pip install -e .[dev,ui]
carapace serve \
  --repo openclaw/openclaw \
  --repo-path /path/to/local/openclaw \
  --skip-repo-path-check \
  --host 127.0.0.1 \
  --port 8765
```
Then open `http://127.0.0.1:8765`.

Optional routing application:
```bash
# dry run (default)
carapace triage --repo openclaw/openclaw --apply-routing

# live GitHub writes (labels/comments)
carapace triage --repo openclaw/openclaw --apply-routing --live-actions
```

## Config
Carapace loads configuration from repository root `.carapace.yaml` with precedence:
1. runtime override
2. repo `.carapace.yaml`
3. org defaults
4. system defaults

Ingest controls live under `ingest`:
- `include_closed`
- `include_drafts`
- `include_issues`
- `page_size`
- `resume`
- `enrich_pr_details`
- `enrich_issue_comments`

Low-pass controls include:
- `skip_closed`
- `skip_drafts`
- `ignore_recent_pr_hours` (e.g. `4` to suppress very fresh PRs from first-pass analysis)

Repo safety:
- By default, GitHub commands validate that `--repo-path` git origin matches `--repo`.
- Use `--skip-repo-path-check` only when you intentionally want to bypass this.

Debug logging:
```bash
carapace --log-level DEBUG ingest --repo openclaw/openclaw --repo-path /path/to/openclaw
```

Machine-readable DB audit:
```bash
carapace audit --repo openclaw/openclaw --repo-path /path/to/openclaw --json
```
