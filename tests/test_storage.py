from pathlib import Path

from carapace.config import CarapaceConfig, StorageConfig
from carapace.models import EntityKind, SourceEntity
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
