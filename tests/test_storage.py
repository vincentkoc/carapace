from datetime import timedelta
from pathlib import Path

from carapace.config import CarapaceConfig, StorageConfig
from carapace.models import EntityKind, Fingerprint, SourceEntity
from carapace.pipeline import CarapaceEngine
from carapace.storage import SQLiteStorage


def test_sqlite_storage_persists_run(tmp_path: Path) -> None:
    db_path = tmp_path / "carapace.db"
    storage = SQLiteStorage(db_path)

    config = CarapaceConfig(storage=StorageConfig(persist_runs=True, sqlite_path=str(db_path)))
    entity = SourceEntity.model_validate(
        {
            "id": "pr:1",
            "repo": "acme/repo",
            "kind": EntityKind.PR,
            "title": "Fix issue",
            "body": "Fixes #12",
            "author": "alice",
            "number": 1,
            "changed_files": ["src/a.py"],
        }
    )

    engine = CarapaceEngine(config=config, storage=storage)
    report = engine.scan_entities([entity])

    assert report.processed_entities == 1
    runs = storage.list_runs()
    assert len(runs) == 1
    assert runs[0]["processed_entities"] == 1


def test_sqlite_ingest_upsert_and_filtering(tmp_path: Path) -> None:
    db_path = tmp_path / "carapace.db"
    storage = SQLiteStorage(db_path)

    open_pr = SourceEntity.model_validate(
        {
            "id": "pr:1",
            "repo": "acme/repo",
            "kind": EntityKind.PR,
            "state": "open",
            "is_draft": False,
            "title": "Open PR",
            "author": "alice",
            "number": 1,
        }
    )
    closed_pr = SourceEntity.model_validate(
        {
            "id": "pr:2",
            "repo": "acme/repo",
            "kind": EntityKind.PR,
            "state": "closed",
            "is_draft": False,
            "title": "Closed PR",
            "author": "bob",
            "number": 2,
        }
    )
    draft_pr = SourceEntity.model_validate(
        {
            "id": "pr:3",
            "repo": "acme/repo",
            "kind": EntityKind.PR,
            "state": "open",
            "is_draft": True,
            "title": "Draft PR",
            "author": "carl",
            "number": 3,
        }
    )

    storage.upsert_ingest_entities("acme/repo", [open_pr, closed_pr, draft_pr])

    default_loaded = storage.load_ingested_entities("acme/repo")
    assert [entity.id for entity in default_loaded] == ["pr:1"]

    with_closed = storage.load_ingested_entities("acme/repo", include_closed=True)
    assert {entity.id for entity in with_closed} == {"pr:1", "pr:2"}

    with_drafts = storage.load_ingested_entities("acme/repo", include_drafts=True)
    assert {entity.id for entity in with_drafts} == {"pr:1", "pr:3"}

    all_entities = storage.load_ingested_entities("acme/repo", include_closed=True, include_drafts=True)
    assert {entity.id for entity in all_entities} == {"pr:1", "pr:2", "pr:3"}

    only_prs = storage.load_ingested_entities("acme/repo", include_closed=True, include_drafts=True, kind="pr")
    assert {entity.id for entity in only_prs} == {"pr:1", "pr:2", "pr:3"}

    issue = SourceEntity.model_validate(
        {
            "id": "issue:10",
            "repo": "acme/repo",
            "kind": EntityKind.ISSUE,
            "state": "open",
            "title": "Issue",
            "author": "dora",
            "number": 10,
        }
    )
    storage.upsert_ingest_entities("acme/repo", [issue])
    only_issues = storage.load_ingested_entities("acme/repo", include_closed=True, include_drafts=True, kind="issue")
    assert [entity.id for entity in only_issues] == ["issue:10"]


def test_mark_entities_closed_except(tmp_path: Path) -> None:
    db_path = tmp_path / "carapace.db"
    storage = SQLiteStorage(db_path)

    entities = [
        SourceEntity.model_validate(
            {
                "id": "pr:1",
                "repo": "acme/repo",
                "kind": EntityKind.PR,
                "state": "open",
                "title": "One",
                "author": "a",
            }
        ),
        SourceEntity.model_validate(
            {
                "id": "pr:2",
                "repo": "acme/repo",
                "kind": EntityKind.PR,
                "state": "open",
                "title": "Two",
                "author": "b",
            }
        ),
    ]
    storage.upsert_ingest_entities("acme/repo", entities)
    changed = storage.mark_entities_closed_except("acme/repo", kind="pr", seen_entity_ids={"pr:1"})
    assert changed == 1

    loaded = storage.load_ingested_entities("acme/repo", include_closed=True, include_drafts=True)
    state_by_id = {entity.id: entity.state for entity in loaded}
    assert state_by_id["pr:1"] == "open"
    assert state_by_id["pr:2"] == "closed"


def test_ingest_upsert_preserves_enriched_payload_on_same_updated_at(tmp_path: Path) -> None:
    db_path = tmp_path / "carapace.db"
    storage = SQLiteStorage(db_path)

    enriched = SourceEntity.model_validate(
        {
            "id": "pr:10",
            "repo": "acme/repo",
            "kind": EntityKind.PR,
            "state": "open",
            "title": "PR",
            "author": "a",
            "updated_at": "2026-02-15T00:00:00Z",
            "changed_files": ["src/core.py"],
        }
    )
    storage.upsert_ingest_entities("acme/repo", [enriched], source="enrichment", enrich_level="minimal")

    # Lightweight ingest payload for same revision should not erase enriched details.
    lightweight = SourceEntity.model_validate(
        {
            "id": "pr:10",
            "repo": "acme/repo",
            "kind": EntityKind.PR,
            "state": "open",
            "title": "PR",
            "author": "a",
            "updated_at": "2026-02-15T00:00:00Z",
            "changed_files": [],
        }
    )
    storage.upsert_ingest_entities("acme/repo", [lightweight], source="ingest")
    loaded = storage.load_ingested_entities("acme/repo", include_closed=True, include_drafts=True)
    assert loaded[0].changed_files == ["src/core.py"]

    watermarks = storage.get_enrichment_watermarks("acme/repo")
    assert watermarks["pr:10"]["enriched_for_updated_at"] == "2026-02-15T00:00:00+00:00"


def test_get_enrichment_watermarks_can_filter_by_kind(tmp_path: Path) -> None:
    db_path = tmp_path / "carapace.db"
    storage = SQLiteStorage(db_path)

    pr_entity = SourceEntity.model_validate(
        {
            "id": "pr:10",
            "repo": "acme/repo",
            "kind": EntityKind.PR,
            "state": "open",
            "title": "PR",
            "author": "a",
            "updated_at": "2026-02-15T00:00:00Z",
            "changed_files": ["src/core.py"],
        }
    )
    issue_entity = SourceEntity.model_validate(
        {
            "id": "issue:20",
            "repo": "acme/repo",
            "kind": EntityKind.ISSUE,
            "state": "open",
            "title": "Issue",
            "author": "b",
            "updated_at": "2026-02-15T00:00:00Z",
        }
    )
    storage.upsert_ingest_entities("acme/repo", [pr_entity], source="enrichment", enrich_level="minimal")
    storage.upsert_ingest_entities("acme/repo", [issue_entity], source="enrichment", enrich_level="minimal")

    pr_watermarks = storage.get_enrichment_watermarks("acme/repo", kind="pr")
    assert set(pr_watermarks.keys()) == {"pr:10"}


def test_ingest_quality_stats_can_filter_by_kind(tmp_path: Path) -> None:
    db_path = tmp_path / "carapace.db"
    storage = SQLiteStorage(db_path)
    storage.upsert_ingest_entities(
        "acme/repo",
        [
            SourceEntity.model_validate(
                {
                    "id": "pr:1",
                    "repo": "acme/repo",
                    "kind": EntityKind.PR,
                    "title": "PR",
                    "author": "alice",
                    "state": "open",
                    "changed_files": [],
                }
            ),
            SourceEntity.model_validate(
                {
                    "id": "issue:2",
                    "repo": "acme/repo",
                    "kind": EntityKind.ISSUE,
                    "title": "Issue",
                    "author": "bob",
                    "state": "open",
                }
            ),
        ],
    )

    all_quality = storage.ingest_quality_stats("acme/repo")
    pr_quality = storage.ingest_quality_stats("acme/repo", kind="pr")
    issue_quality = storage.ingest_quality_stats("acme/repo", kind="issue")
    assert all_quality["total"] == 2
    assert pr_quality["total"] == 1
    assert issue_quality["total"] == 1


def test_ingest_audit_summary_reports_integrity(tmp_path: Path) -> None:
    db_path = tmp_path / "carapace.db"
    storage = SQLiteStorage(db_path)
    storage.upsert_ingest_entities(
        "acme/repo",
        [
            SourceEntity.model_validate(
                {
                    "id": "pr:1",
                    "repo": "acme/repo",
                    "kind": EntityKind.PR,
                    "number": 1,
                    "title": "PR",
                    "author": "alice",
                    "state": "open",
                    "changed_files": [],
                }
            ),
            SourceEntity.model_validate(
                {
                    "id": "issue:2",
                    "repo": "acme/repo",
                    "kind": EntityKind.ISSUE,
                    "number": 2,
                    "title": "Issue",
                    "author": "bob",
                    "state": "open",
                }
            ),
        ],
    )

    # Inject one malformed row to ensure integrity checks catch it.
    with storage._connect() as conn:  # noqa: SLF001 - test-only direct DB mutation
        conn.execute(
            """
            INSERT OR REPLACE INTO ingest_entities (
              repo, entity_id, kind, number, state, is_draft, updated_at, payload_json, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "acme/repo",
                "issue:999",
                "pr",
                3,
                "open",
                0,
                "2026-02-15T00:00:00+00:00",
                '{"id":"issue:999","repo":"wrong/repo","kind":"issue","number":999,"state":"open","title":"bad","author":"x"}',
                "2026-02-15T00:00:00+00:00",
            ),
        )
        conn.commit()

    summary = storage.ingest_audit_summary("acme/repo")
    assert summary["total"] == 3
    assert summary["by_kind"]["pr"] == 2
    assert summary["integrity"]["kind_id_prefix_mismatch"] == 1
    assert summary["integrity"]["kind_payload_mismatch"] == 1
    assert summary["integrity"]["repo_payload_mismatch"] == 1
    assert summary["integrity"]["entity_number_mismatch"] == 1


def test_fingerprint_cache_respects_updated_at(tmp_path: Path) -> None:
    db_path = tmp_path / "carapace.db"
    storage = SQLiteStorage(db_path)
    entity = SourceEntity.model_validate(
        {
            "id": "pr:1",
            "repo": "acme/repo",
            "kind": EntityKind.PR,
            "state": "open",
            "title": "PR",
            "author": "alice",
            "updated_at": "2026-02-15T00:00:00Z",
        }
    )
    fp = Fingerprint(
        entity_id="pr:1",
        tokens=["a"],
        embedding=[0.1, 0.2],
    )
    written = storage.upsert_fingerprint_cache("acme/repo", [entity], {"pr:1": fp}, model_id="test-model")
    assert written == 1

    hits = storage.load_fingerprint_cache("acme/repo", [entity], model_id="test-model")
    assert "pr:1" in hits

    newer_entity = entity.model_copy(update={"updated_at": entity.updated_at + timedelta(days=1)})
    misses = storage.load_fingerprint_cache("acme/repo", [newer_entity], model_id="test-model")
    assert misses == {}

    # Same updated_at but changed fingerprint inputs should invalidate cache.
    changed_entity = entity.model_copy(update={"changed_files": ["src/new.py"]})
    misses_same_timestamp = storage.load_fingerprint_cache("acme/repo", [changed_entity], model_id="test-model")
    assert misses_same_timestamp == {}


def test_storage_latest_report_and_entity_lookup(tmp_path: Path) -> None:
    db_path = tmp_path / "carapace.db"
    storage = SQLiteStorage(db_path)

    config = CarapaceConfig(storage=StorageConfig(persist_runs=True, sqlite_path=str(db_path)))
    entities = [
        SourceEntity.model_validate(
            {
                "id": "pr:1",
                "repo": "acme/repo",
                "kind": EntityKind.PR,
                "state": "open",
                "title": "A",
                "author": "alice",
                "number": 1,
            }
        ),
        SourceEntity.model_validate(
            {
                "id": "issue:2",
                "repo": "acme/repo",
                "kind": EntityKind.ISSUE,
                "state": "open",
                "title": "B",
                "author": "bob",
                "number": 2,
            }
        ),
    ]
    storage.upsert_ingest_entities("acme/repo", entities)

    engine = CarapaceEngine(config=config, storage=storage)
    report = engine.scan_entities(entities)
    assert report.processed_entities == 2

    latest = storage.get_latest_run_report("acme/repo")
    assert latest is not None
    assert latest.processed_entities == 2

    looked_up = storage.load_ingested_entities_by_ids("acme/repo", ["pr:1", "issue:2"])
    assert set(looked_up.keys()) == {"pr:1", "issue:2"}
    assert looked_up["pr:1"].author == "alice"

    repos = storage.list_ingested_repos()
    assert repos == ["acme/repo"]


def test_author_metrics_cache_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "carapace.db"
    storage = SQLiteStorage(db_path)

    written = storage.upsert_author_metrics_cache(
        "acme/repo",
        {
            "alice": (42, 0.93),
            "bob": (3, 0.45),
        },
    )
    assert written == 2

    cache = storage.get_author_metrics_cache("acme/repo", ["alice", "bob", "carol"])
    assert cache["alice"]["merged_pr_count"] == 42
    assert float(cache["alice"]["trust_score"]) == 0.93
    assert cache["bob"]["merged_pr_count"] == 3
    assert "carol" not in cache


def test_load_latest_run_embeddings_returns_vectors(tmp_path: Path) -> None:
    db_path = tmp_path / "carapace.db"
    storage = SQLiteStorage(db_path)
    config = CarapaceConfig(storage=StorageConfig(persist_runs=True, sqlite_path=str(db_path)))
    entities = [
        SourceEntity.model_validate(
            {
                "id": "pr:1",
                "repo": "acme/repo",
                "kind": EntityKind.PR,
                "state": "open",
                "title": "Fix bug",
                "body": "Fixes #12",
                "author": "alice",
                "number": 1,
                "changed_files": ["src/a.py"],
            }
        ),
        SourceEntity.model_validate(
            {
                "id": "issue:12",
                "repo": "acme/repo",
                "kind": EntityKind.ISSUE,
                "state": "open",
                "title": "Bug",
                "body": "Something broke",
                "author": "bob",
                "number": 12,
            }
        ),
    ]
    storage.upsert_ingest_entities("acme/repo", entities)
    engine = CarapaceEngine(config=config, storage=storage)
    _ = engine.scan_entities(entities)

    run_id, vectors = storage.load_latest_run_embeddings("acme/repo")
    assert run_id is not None
    assert "pr:1" in vectors
    assert len(vectors["pr:1"]) > 0

    _, filtered_vectors = storage.load_latest_run_embeddings("acme/repo", entity_ids=["pr:1"])
    assert set(filtered_vectors.keys()) == {"pr:1"}


def test_json_cache_roundtrip_with_signature(tmp_path: Path) -> None:
    db_path = tmp_path / "carapace.db"
    storage = SQLiteStorage(db_path)

    assert storage.load_json_cache("acme/repo", "atlas:default", source_signature="sig-a") is None

    payload = {"node_count": 2, "edge_count": 1, "elements": {"nodes": [], "edges": []}}
    storage.upsert_json_cache(
        "acme/repo",
        "atlas:default",
        source_signature="sig-a",
        payload=payload,
    )

    cached = storage.load_json_cache("acme/repo", "atlas:default", source_signature="sig-a")
    assert cached == payload
    assert storage.load_json_cache("acme/repo", "atlas:default", source_signature="sig-b") is None
