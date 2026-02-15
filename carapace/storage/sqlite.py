"""SQLite storage backend with vector-ready schema."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from carapace.models import EngineReport, Fingerprint, SourceEntity


class SQLiteStorage:
    """SQLite-backed persistence with clear adapter boundary for PostgreSQL swap."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  generated_at TEXT NOT NULL,
                  processed_entities INTEGER NOT NULL,
                  active_entities INTEGER NOT NULL,
                  suppressed_entities INTEGER NOT NULL,
                  skipped_entities INTEGER NOT NULL,
                  report_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS entities (
                  run_id INTEGER NOT NULL,
                  entity_id TEXT NOT NULL,
                  provider TEXT NOT NULL,
                  repo TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  number INTEGER,
                  title TEXT NOT NULL,
                  labels_json TEXT NOT NULL,
                  author TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  PRIMARY KEY (run_id, entity_id),
                  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                  run_id INTEGER NOT NULL,
                  entity_id TEXT NOT NULL,
                  model_id TEXT NOT NULL,
                  dimensions INTEGER NOT NULL,
                  vector_json TEXT NOT NULL,
                  PRIMARY KEY (run_id, entity_id),
                  FOREIGN KEY (run_id, entity_id) REFERENCES entities(run_id, entity_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS fingerprints (
                  run_id INTEGER NOT NULL,
                  entity_id TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  PRIMARY KEY (run_id, entity_id),
                  FOREIGN KEY (run_id, entity_id) REFERENCES entities(run_id, entity_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_entities_repo_kind ON entities(repo, kind);
                CREATE INDEX IF NOT EXISTS idx_entities_number ON entities(number);

                CREATE TABLE IF NOT EXISTS ingest_entities (
                  repo TEXT NOT NULL,
                  entity_id TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  number INTEGER,
                  state TEXT NOT NULL,
                  is_draft INTEGER NOT NULL,
                  updated_at TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  fetched_at TEXT NOT NULL,
                  enriched_for_updated_at TEXT,
                  last_enriched_at TEXT,
                  enrich_level TEXT,
                  PRIMARY KEY (repo, entity_id)
                );

                CREATE TABLE IF NOT EXISTS ingest_state (
                  repo TEXT PRIMARY KEY,
                  pr_next_page INTEGER NOT NULL DEFAULT 1,
                  issue_next_page INTEGER NOT NULL DEFAULT 1,
                  phase TEXT NOT NULL DEFAULT 'prs',
                  last_sync_at TEXT,
                  completed INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_ingest_entities_repo_state ON ingest_entities(repo, state, is_draft);
                CREATE INDEX IF NOT EXISTS idx_ingest_entities_repo_kind_state_num ON ingest_entities(repo, kind, state, number);
                """
            )
            self._ensure_ingest_columns(conn)

    @staticmethod
    def _ensure_ingest_columns(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(ingest_entities)").fetchall()
        columns = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in rows}

        to_add = []
        if "enriched_for_updated_at" not in columns:
            to_add.append("ALTER TABLE ingest_entities ADD COLUMN enriched_for_updated_at TEXT")
        if "last_enriched_at" not in columns:
            to_add.append("ALTER TABLE ingest_entities ADD COLUMN last_enriched_at TEXT")
        if "enrich_level" not in columns:
            to_add.append("ALTER TABLE ingest_entities ADD COLUMN enrich_level TEXT")

        for statement in to_add:
            conn.execute(statement)

    def save_run(
        self,
        entities: list[SourceEntity],
        fingerprints: dict[str, Fingerprint],
        report: EngineReport,
        embedding_model: str,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs (
                  generated_at,
                  processed_entities,
                  active_entities,
                  suppressed_entities,
                  skipped_entities,
                  report_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report.generated_at.isoformat(),
                    report.processed_entities,
                    report.active_entities,
                    report.suppressed_entities,
                    report.skipped_entities,
                    report.model_dump_json(),
                ),
            )
            run_id = int(cursor.lastrowid)

            for entity in entities:
                conn.execute(
                    """
                    INSERT INTO entities (
                      run_id, entity_id, provider, repo, kind, number, title,
                      labels_json, author, updated_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        entity.id,
                        entity.provider,
                        entity.repo,
                        entity.kind.value,
                        entity.number,
                        entity.title,
                        json.dumps(entity.labels),
                        entity.author,
                        entity.updated_at.isoformat(),
                        json.dumps(entity.metadata),
                    ),
                )

                fp = fingerprints.get(entity.id)
                if fp is None:
                    continue

                conn.execute(
                    """
                    INSERT INTO fingerprints (run_id, entity_id, payload_json)
                    VALUES (?, ?, ?)
                    """,
                    (run_id, entity.id, fp.model_dump_json()),
                )

                conn.execute(
                    """
                    INSERT INTO embeddings (run_id, entity_id, model_id, dimensions, vector_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        entity.id,
                        embedding_model,
                        len(fp.embedding),
                        json.dumps(fp.embedding),
                    ),
                )

            conn.commit()
            return run_id

    def list_runs(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, generated_at, processed_entities, active_entities, suppressed_entities, skipped_entities
                FROM runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_ingest_entities(
        self,
        repo: str,
        entities: list[SourceEntity],
        *,
        source: str = "ingest",
        enrich_level: str | None = None,
    ) -> int:
        if not entities:
            return 0
        now = datetime.now(UTC).isoformat()
        rows = [
            (
                repo,
                entity.id,
                entity.kind.value,
                entity.number,
                entity.state,
                1 if entity.is_draft else 0,
                entity.updated_at.isoformat(),
                entity.model_dump_json(),
                now,
                entity.updated_at.isoformat() if source == "enrichment" else None,
                now if source == "enrichment" else None,
                enrich_level if source == "enrichment" else None,
            )
            for entity in entities
        ]
        with self._connect() as conn:
            if source == "enrichment":
                conn.executemany(
                    """
                    INSERT INTO ingest_entities (
                      repo, entity_id, kind, number, state, is_draft, updated_at, payload_json, fetched_at,
                      enriched_for_updated_at, last_enriched_at, enrich_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(repo, entity_id) DO UPDATE SET
                      kind=excluded.kind,
                      number=excluded.number,
                      state=excluded.state,
                      is_draft=excluded.is_draft,
                      updated_at=excluded.updated_at,
                      payload_json=excluded.payload_json,
                      fetched_at=excluded.fetched_at,
                      enriched_for_updated_at=excluded.enriched_for_updated_at,
                      last_enriched_at=excluded.last_enriched_at,
                      enrich_level=excluded.enrich_level
                    """,
                    rows,
                )
            else:
                conn.executemany(
                    """
                    INSERT INTO ingest_entities (
                      repo, entity_id, kind, number, state, is_draft, updated_at, payload_json, fetched_at,
                      enriched_for_updated_at, last_enriched_at, enrich_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(repo, entity_id) DO UPDATE SET
                      kind=excluded.kind,
                      number=excluded.number,
                      state=excluded.state,
                      is_draft=excluded.is_draft,
                      updated_at=excluded.updated_at,
                      payload_json=CASE
                        WHEN excluded.updated_at = ingest_entities.updated_at
                          AND COALESCE(json_array_length(json_extract(excluded.payload_json, '$.changed_files')), 0) = 0
                          AND COALESCE(json_array_length(json_extract(ingest_entities.payload_json, '$.changed_files')), 0) > 0
                        THEN ingest_entities.payload_json
                        ELSE excluded.payload_json
                      END,
                      fetched_at=excluded.fetched_at,
                      enriched_for_updated_at=CASE
                        WHEN excluded.updated_at = ingest_entities.updated_at
                          AND COALESCE(json_array_length(json_extract(excluded.payload_json, '$.changed_files')), 0) = 0
                          AND COALESCE(json_array_length(json_extract(ingest_entities.payload_json, '$.changed_files')), 0) > 0
                        THEN ingest_entities.enriched_for_updated_at
                        ELSE NULL
                      END,
                      last_enriched_at=CASE
                        WHEN excluded.updated_at = ingest_entities.updated_at
                          AND COALESCE(json_array_length(json_extract(excluded.payload_json, '$.changed_files')), 0) = 0
                          AND COALESCE(json_array_length(json_extract(ingest_entities.payload_json, '$.changed_files')), 0) > 0
                        THEN ingest_entities.last_enriched_at
                        ELSE NULL
                      END,
                      enrich_level=CASE
                        WHEN excluded.updated_at = ingest_entities.updated_at
                          AND COALESCE(json_array_length(json_extract(excluded.payload_json, '$.changed_files')), 0) = 0
                          AND COALESCE(json_array_length(json_extract(ingest_entities.payload_json, '$.changed_files')), 0) > 0
                        THEN ingest_entities.enrich_level
                        ELSE NULL
                      END
                    """,
                    rows,
                )
            conn.commit()
        return len(entities)

    def load_ingested_entities(
        self,
        repo: str,
        include_closed: bool = False,
        include_drafts: bool = False,
    ) -> list[SourceEntity]:
        predicates = ["repo = ?"]
        params: list[object] = [repo]
        if not include_closed:
            predicates.append("state = 'open'")
        if not include_drafts:
            predicates.append("is_draft = 0")

        sql = (
            "SELECT payload_json FROM ingest_entities WHERE "
            + " AND ".join(predicates)
            + " ORDER BY kind, number"
        )

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [SourceEntity.model_validate_json(row["payload_json"]) for row in rows]

    def get_ingest_state(self, repo: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT repo, pr_next_page, issue_next_page, phase, last_sync_at, completed
                FROM ingest_state
                WHERE repo = ?
                """,
                (repo,),
            ).fetchone()
        if row is None:
            return {
                "repo": repo,
                "pr_next_page": 1,
                "issue_next_page": 1,
                "phase": "prs",
                "last_sync_at": None,
                "completed": 0,
            }
        return dict(row)

    def save_ingest_state(
        self,
        repo: str,
        *,
        pr_next_page: int,
        issue_next_page: int,
        phase: str,
        completed: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingest_state (
                  repo, pr_next_page, issue_next_page, phase, last_sync_at, completed
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo) DO UPDATE SET
                  pr_next_page=excluded.pr_next_page,
                  issue_next_page=excluded.issue_next_page,
                  phase=excluded.phase,
                  last_sync_at=excluded.last_sync_at,
                  completed=excluded.completed
                """,
                (
                    repo,
                    pr_next_page,
                    issue_next_page,
                    phase,
                    datetime.now(UTC).isoformat(),
                    1 if completed else 0,
                ),
            )
            conn.commit()

    def ingest_quality_stats(self, repo: str, kind: str | None = None) -> dict[str, int]:
        predicates = ["repo = ?"]
        params: list[object] = [repo]
        if kind is not None:
            predicates.append("kind = ?")
            params.append(kind)

        sql = (
            """
                SELECT
                  COUNT(*) AS total,
                  SUM(CASE WHEN COALESCE(json_array_length(json_extract(payload_json, '$.changed_files')), 0) = 0 THEN 1 ELSE 0 END) AS missing_changed_files,
                  SUM(CASE WHEN COALESCE(json_array_length(json_extract(payload_json, '$.diff_hunks')), 0) = 0 THEN 1 ELSE 0 END) AS missing_diff_hunks,
                  SUM(CASE WHEN COALESCE(json_extract(payload_json, '$.ci_status'), 'unknown') = 'unknown' THEN 1 ELSE 0 END) AS ci_unknown,
                  SUM(CASE WHEN enriched_for_updated_at IS NOT NULL THEN 1 ELSE 0 END) AS enriched_rows
                FROM ingest_entities
                WHERE """
            + " AND ".join(predicates)
        )
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return {
                "total": 0,
                "missing_changed_files": 0,
                "missing_diff_hunks": 0,
                "ci_unknown": 0,
                "enriched_rows": 0,
            }
        return {
            "total": int(row["total"] or 0),
            "missing_changed_files": int(row["missing_changed_files"] or 0),
            "missing_diff_hunks": int(row["missing_diff_hunks"] or 0),
            "ci_unknown": int(row["ci_unknown"] or 0),
            "enriched_rows": int(row["enriched_rows"] or 0),
        }

    def get_enrichment_watermarks(self, repo: str, kind: str | None = None) -> dict[str, dict[str, str | None]]:
        predicates = ["repo = ?"]
        params: list[object] = [repo]
        if kind is not None:
            predicates.append("kind = ?")
            params.append(kind)

        sql = (
            """
                SELECT entity_id, enriched_for_updated_at, enrich_level
                FROM ingest_entities
                WHERE """
            + " AND ".join(predicates)
        )
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return {
            row["entity_id"]: {
                "enriched_for_updated_at": row["enriched_for_updated_at"],
                "enrich_level": row["enrich_level"],
            }
            for row in rows
        }

    def mark_entities_closed_except(self, repo: str, *, kind: str, seen_entity_ids: set[str]) -> int:
        with self._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS temp._seen_ids")
            conn.execute("CREATE TEMP TABLE _seen_ids (entity_id TEXT PRIMARY KEY)")
            if seen_entity_ids:
                conn.executemany(
                    "INSERT INTO _seen_ids(entity_id) VALUES (?)",
                    [(entity_id,) for entity_id in sorted(seen_entity_ids)],
                )

            cursor = conn.execute(
                """
                UPDATE ingest_entities
                SET
                  state = 'closed',
                  payload_json = json_set(payload_json, '$.state', 'closed')
                WHERE repo = ?
                  AND kind = ?
                  AND state = 'open'
                  AND entity_id NOT IN (SELECT entity_id FROM _seen_ids)
                """,
                (repo, kind),
            )
            conn.commit()
            return int(cursor.rowcount if cursor.rowcount is not None else 0)

    def ingest_audit_summary(self, repo: str) -> dict:
        with self._connect() as conn:
            total = int(
                conn.execute(
                    "SELECT COUNT(*) FROM ingest_entities WHERE repo = ?",
                    (repo,),
                ).fetchone()[0]
            )

            by_kind_rows = conn.execute(
                """
                SELECT kind, COUNT(*) AS c
                FROM ingest_entities
                WHERE repo = ?
                GROUP BY kind
                """,
                (repo,),
            ).fetchall()
            by_kind = {row["kind"]: int(row["c"]) for row in by_kind_rows}

            by_kind_state_rows = conn.execute(
                """
                SELECT kind, state, COUNT(*) AS c
                FROM ingest_entities
                WHERE repo = ?
                GROUP BY kind, state
                """,
                (repo,),
            ).fetchall()
            by_kind_state: dict[str, dict[str, int]] = {}
            for row in by_kind_state_rows:
                kind = row["kind"]
                state = row["state"]
                by_kind_state.setdefault(kind, {})[state] = int(row["c"])

            draft_prs = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM ingest_entities
                    WHERE repo = ? AND kind = 'pr' AND is_draft = 1
                    """,
                    (repo,),
                ).fetchone()[0]
            )

            enriched_prs = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM ingest_entities
                    WHERE repo = ? AND kind = 'pr' AND enriched_for_updated_at IS NOT NULL
                    """,
                    (repo,),
                ).fetchone()[0]
            )

            enrich_level_rows = conn.execute(
                """
                SELECT COALESCE(enrich_level, 'none') AS level, COUNT(*) AS c
                FROM ingest_entities
                WHERE repo = ? AND kind = 'pr'
                GROUP BY COALESCE(enrich_level, 'none')
                """,
                (repo,),
            ).fetchall()
            enrich_levels = {row["level"]: int(row["c"]) for row in enrich_level_rows}

            integrity = {
                "kind_id_prefix_mismatch": int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM ingest_entities
                        WHERE repo = ?
                          AND (
                            (kind = 'pr' AND entity_id NOT LIKE 'pr:%')
                            OR (kind = 'issue' AND entity_id NOT LIKE 'issue:%')
                          )
                        """,
                        (repo,),
                    ).fetchone()[0]
                ),
                "kind_payload_mismatch": int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM ingest_entities
                        WHERE repo = ?
                          AND COALESCE(json_extract(payload_json, '$.kind'), '') != kind
                        """,
                        (repo,),
                    ).fetchone()[0]
                ),
                "repo_payload_mismatch": int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM ingest_entities
                        WHERE repo = ?
                          AND COALESCE(json_extract(payload_json, '$.repo'), '') != repo
                        """,
                        (repo,),
                    ).fetchone()[0]
                ),
                "entity_number_mismatch": int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM ingest_entities
                        WHERE repo = ?
                          AND number IS NOT NULL
                          AND (
                            (kind = 'pr' AND entity_id != ('pr:' || number))
                            OR (kind = 'issue' AND entity_id != ('issue:' || number))
                          )
                        """,
                        (repo,),
                    ).fetchone()[0]
                ),
            }

        return {
            "total": total,
            "by_kind": by_kind,
            "by_kind_state": by_kind_state,
            "draft_prs": draft_prs,
            "enriched_prs": enriched_prs,
            "enrich_levels": enrich_levels,
            "integrity": integrity,
        }
