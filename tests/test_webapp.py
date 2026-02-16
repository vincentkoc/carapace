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

    atlas = client.get(f"/api/repos/{repo}/graph/atlas")
    assert atlas.status_code == 200
    atlas_payload = atlas.json()
    assert atlas_payload["mode"] == "embedding_atlas"
    assert atlas_payload["node_count"] >= 1

    cluster_map = client.get(f"/api/repos/{repo}/graph/cluster-map")
    assert cluster_map.status_code == 200

    clusters = client.get(f"/api/repos/{repo}/clusters?min_members=1")
    assert clusters.status_code == 200
    cluster_rows = clusters.json().get("clusters", [])
    assert cluster_rows
    assert "canonical_title" in cluster_rows[0]

    detail = client.get(f"/api/repos/{repo}/clusters/{cluster_rows[0]['cluster_id']}/detail")
    assert detail.status_code == 200

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
    assert payload["node_count"] >= 1

    graph_with_authors = client.get(f"/api/repos/{repo}/graph?include_authors=true")
    assert graph_with_authors.status_code == 200
    payload_with_authors = graph_with_authors.json()
    assert payload_with_authors["node_count"] >= 2  # issue + author node

    atlas = client.get(f"/api/repos/{repo}/graph/atlas")
    assert atlas.status_code == 200
    atlas_payload = atlas.json()
    assert atlas_payload["mode"] == "embedding_atlas"
    assert atlas_payload["node_count"] == 1

    cluster_map = client.get(f"/api/repos/{repo}/graph/cluster-map")
    assert cluster_map.status_code == 200
    cluster_map_payload = cluster_map.json()
    assert cluster_map_payload["mode"] == "cluster_map_ingest"
    assert cluster_map_payload["node_count"] == 0


def test_webapp_factory_from_env_loads_repo_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from carapace.webapp import create_app_from_env

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    db_path = repo_root / ".carapace" / "carapace.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    (repo_root / ".carapace.yaml").write_text(
        f"""
storage:
  backend: sqlite
  sqlite_path: {db_path}
  persist_runs: true
"""
    )
    storage = SQLiteStorage(db_path)
    storage.upsert_ingest_entities(
        "openclaw/openclaw",
        [
            SourceEntity.model_validate(
                {
                    "id": "issue:1",
                    "repo": "openclaw/openclaw",
                    "kind": EntityKind.ISSUE,
                    "state": "open",
                    "number": 1,
                    "title": "A",
                    "author": "alice",
                }
            )
        ],
    )

    monkeypatch.setenv("CARAPACE_REPO_PATH", str(repo_root))
    app = create_app_from_env()
    client = TestClient(app)
    res = client.get("/api/repos/openclaw/openclaw/graph")
    assert res.status_code == 200


def test_webapp_cluster_detail_from_ingest_without_runs(tmp_path: Path) -> None:
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
                "id": "issue:10",
                "repo": repo,
                "kind": EntityKind.ISSUE,
                "state": "open",
                "number": 10,
                "title": "Bug report",
                "author": "alice",
            }
        ),
        SourceEntity.model_validate(
            {
                "id": "pr:11",
                "repo": repo,
                "kind": EntityKind.PR,
                "state": "open",
                "number": 11,
                "title": "Fix bug",
                "body": "Fixes #10",
                "author": "bob",
                "linked_issues": ["10"],
            }
        ),
    ]
    storage.upsert_ingest_entities(repo, entities)

    app = create_app(config)
    client = TestClient(app)

    clusters = client.get(f"/api/repos/{repo}/clusters?min_members=2")
    assert clusters.status_code == 200
    rows = clusters.json()["clusters"]
    assert rows
    assert rows[0]["canonical_title"] is not None
    cid = rows[0]["cluster_id"]
    assert cid.startswith("ingest-cluster-")

    detail = client.get(f"/api/repos/{repo}/clusters/{cid}/detail")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["mode"] == "cluster_detail_ingest"
    assert payload["node_count"] >= 2
