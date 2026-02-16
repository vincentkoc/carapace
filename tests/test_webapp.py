from pathlib import Path

import pytest

from carapace.config import CarapaceConfig, StorageConfig
from carapace.models import EntityKind, SourceEntity
from carapace.pipeline import CarapaceEngine
from carapace.storage import SQLiteStorage


def test_webapp_repo_path_routes_accept_owner_repo(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from carapace.webapp import create_app

    db_path = tmp_path / "carapace.db"
    repo = "openclaw/openclaw"

    storage = SQLiteStorage(db_path)
    config = CarapaceConfig(storage=StorageConfig(persist_runs=True, sqlite_path=str(db_path)))

    entities = [
        SourceEntity.model_validate(
            {
                "id": "issue:1",
                "repo": repo,
                "kind": EntityKind.ISSUE,
                "state": "open",
                "number": 1,
                "title": "Bug",
                "body": "Fixes #1",
                "author": "alice",
            }
        ),
        SourceEntity.model_validate(
            {
                "id": "pr:2",
                "repo": repo,
                "kind": EntityKind.PR,
                "state": "open",
                "number": 2,
                "title": "Fix bug",
                "body": "Fixes #1",
                "author": "bob",
                "changed_files": ["src/app.py"],
            }
        ),
    ]
    storage.upsert_ingest_entities(repo, entities)

    engine = CarapaceEngine(config=config, storage=storage)
    report = engine.scan_entities(entities)
    assert report.processed_entities == 2

    app = create_app(config)
    client = TestClient(app)

    graph = client.get(f"/api/repos/{repo}/graph")
    assert graph.status_code == 200

    clusters = client.get(f"/api/repos/{repo}/clusters")
    assert clusters.status_code == 200

    authors = client.get(f"/api/repos/{repo}/authors")
    assert authors.status_code == 200


def test_webapp_graph_falls_back_to_ingest_when_run_has_no_multi_clusters(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from carapace.webapp import create_app

    db_path = tmp_path / "carapace.db"
    repo = "openclaw/openclaw"

    storage = SQLiteStorage(db_path)
    config = CarapaceConfig(storage=StorageConfig(persist_runs=True, sqlite_path=str(db_path)))

    singleton = SourceEntity.model_validate(
        {
            "id": "issue:1",
            "repo": repo,
            "kind": EntityKind.ISSUE,
            "state": "open",
            "number": 1,
            "title": "Only one",
            "body": "",
            "author": "alice",
        }
    )
    storage.upsert_ingest_entities(repo, [singleton])
    engine = CarapaceEngine(config=config, storage=storage)
    _ = engine.scan_entities([singleton])

    app = create_app(config)
    client = TestClient(app)
    graph = client.get(f"/api/repos/{repo}/graph")
    assert graph.status_code == 200
    payload = graph.json()
    assert payload["mode"] == "ingest_fallback"
    assert payload["node_count"] >= 2  # issue + author node
