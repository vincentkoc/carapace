"""SQLite storage backend with vector-ready schema."""

from __future__ import annotations

import json
import sqlite3
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
        return conn

    def init_schema(self) -> None:
        with self._connect() as conn:
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
                """
            )

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
