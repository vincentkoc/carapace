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
