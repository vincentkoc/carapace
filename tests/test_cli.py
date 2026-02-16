import json
from datetime import UTC, datetime
from pathlib import Path

from carapace import cli
from carapace.connectors.github_gh import GithubRateLimitError
from carapace.models import CIStatus, EntityKind, SourceEntity


def test_cli_parser_supports_simple_command_aliases() -> None:
    parser = cli.build_parser()
    parsed = parser.parse_args(["ingest", "--repo", "acme/repo"])
    assert parsed.command == "ingest"

    parsed_process = parser.parse_args(["process", "--repo", "acme/repo"])
    assert parsed_process.command == "process"

    parsed_serve = parser.parse_args(["serve", "--repo-path", "."])
    assert parsed_serve.command == "serve"


def test_default_enrich_workers_honors_env(monkeypatch) -> None:
    monkeypatch.setenv("CARAPACE_ENRICH_WORKERS", "11")
    assert cli._default_enrich_workers() == 11  # noqa: SLF001 - module helper test

    monkeypatch.setenv("CARAPACE_ENRICH_WORKERS", "bad")
    value = cli._default_enrich_workers()  # noqa: SLF001 - module helper test
    assert value >= 4


def test_scan_command_writes_reports(tmp_path: Path) -> None:
    input_path = tmp_path / "entities.json"
    out_dir = tmp_path / "out"
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    entities = [
        {
            "id": "pr:1",
            "repo": "acme/repo",
            "kind": "pr",
            "number": 1,
            "title": "Fix parser",
            "body": "Fixes #1",
            "author": "alice",
            "changed_files": ["src/parser.py"],
            "ci_status": "pass",
        }
    ]
    input_path.write_text(json.dumps(entities))

    exit_code = cli.main(
        [
            "scan",
            "--input",
            str(input_path),
            "--output-dir",
            str(out_dir),
            "--repo-path",
            str(repo_path),
        ]
    )
    assert exit_code == 0
    assert (out_dir / "triage_report.md").exists()


def test_scan_command_can_include_singleton_clusters_in_report(tmp_path: Path) -> None:
    input_path = tmp_path / "entities.json"
    out_dir = tmp_path / "out"
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    entities = [
        {
            "id": "pr:1",
            "repo": "acme/repo",
            "kind": "pr",
            "number": 1,
            "title": "Fix parser",
            "body": "Fixes #1",
            "author": "alice",
            "changed_files": ["src/parser.py"],
            "ci_status": "pass",
        }
    ]
    input_path.write_text(json.dumps(entities))

    exit_code = cli.main(
        [
            "scan",
            "--input",
            str(input_path),
            "--output-dir",
            str(out_dir),
            "--repo-path",
            str(repo_path),
            "--report-include-singletons",
        ]
    )
    assert exit_code == 0
    triage = (out_dir / "triage_report.md").read_text()
    assert "## cluster-1" in triage


def test_scan_github_command_uses_connector_and_can_save_input(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    class FakeSource:
        def __init__(self, repo: str, gh_bin: str = "gh", **kwargs: object) -> None:
            _ = (repo, gh_bin, kwargs)

        def fetch_open_entities(
            self,
            max_prs: int,
            include_issues: bool,
            max_issues: int,
            include_drafts: bool,
            include_closed: bool,
            enrich_pr_details: bool,
            enrich_issue_comments: bool,
        ):
            _ = (
                max_prs,
                include_issues,
                max_issues,
                include_drafts,
                include_closed,
                enrich_pr_details,
                enrich_issue_comments,
            )
            return [
                SourceEntity(
                    id="pr:9",
                    repo="acme/repo",
                    kind=EntityKind.PR,
                    number=9,
                    title="Title",
                    body="Body",
                    author="alice",
                    changed_files=["src/a.py"],
                    ci_status=CIStatus.PASS,
                )
            ]

    monkeypatch.setattr(cli, "GithubGhSourceConnector", FakeSource)

    exit_code = cli.main(
        [
            "scan-github",
            "--repo",
            "acme/repo",
            "--output-dir",
            str(out_dir),
            "--repo-path",
            str(repo_path),
            "--save-input-json",
            "--no-include-issues",
            "--skip-repo-path-check",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "entities.json").exists()
    assert (out_dir / "labels_to_apply.json").exists()


def test_ingest_github_command_invokes_loader(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    class FakeSource:
        def __init__(self, repo: str, gh_bin: str = "gh", **kwargs: object) -> None:
            _ = (repo, gh_bin, kwargs)

    class FakeResult:
        repo = "acme/repo"
        prs_ingested = 2
        issues_ingested = 1
        pr_pages = 1
        issue_pages = 1

    called = {"loader": 0}

    def fake_loader(connector, storage, *, repo, ingest_cfg, max_prs, max_issues):
        _ = (connector, storage, repo, ingest_cfg, max_prs, max_issues)
        called["loader"] += 1
        return FakeResult()

    monkeypatch.setattr(cli, "GithubGhSourceConnector", FakeSource)
    monkeypatch.setattr(cli, "ingest_github_to_sqlite", fake_loader)

    exit_code = cli.main(
        [
            "ingest-github",
            "--repo",
            "acme/repo",
            "--repo-path",
            str(repo_path),
            "--skip-repo-path-check",
        ]
    )
    assert exit_code == 0
    assert called["loader"] == 1


def test_process_stored_command_reads_sqlite_and_writes_output(tmp_path: Path) -> None:
    from carapace.storage import SQLiteStorage

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    out_dir = tmp_path / "out"
    db_path = repo_path / ".carapace" / "carapace.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    storage = SQLiteStorage(db_path)
    storage.upsert_ingest_entities(
        "acme/repo",
        [
            SourceEntity(
                id="pr:1",
                repo="acme/repo",
                kind=EntityKind.PR,
                number=1,
                state="open",
                title="Fix parser",
                body="Fixes #1",
                author="alice",
                changed_files=["src/parser.py"],
                ci_status=CIStatus.PASS,
            )
        ],
    )

    (repo_path / ".carapace.yaml").write_text(
        f"""
storage:
  backend: sqlite
  sqlite_path: {db_path}
  persist_runs: false
"""
    )

    exit_code = cli.main(
        [
            "process-stored",
            "--repo",
            "acme/repo",
            "--repo-path",
            str(repo_path),
            "--output-dir",
            str(out_dir),
            "--skip-repo-path-check",
        ]
    )
    assert exit_code == 0
    assert (out_dir / "triage_report.md").exists()


def test_process_stored_can_filter_entity_kind_issue(tmp_path: Path) -> None:
    from carapace.storage import SQLiteStorage

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    out_dir = tmp_path / "out"
    db_path = repo_path / ".carapace" / "carapace.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    storage = SQLiteStorage(db_path)
    storage.upsert_ingest_entities(
        "acme/repo",
        [
            SourceEntity(
                id="pr:1",
                repo="acme/repo",
                kind=EntityKind.PR,
                number=1,
                state="open",
                title="PR",
                body="",
                author="alice",
            ),
            SourceEntity(
                id="issue:2",
                repo="acme/repo",
                kind=EntityKind.ISSUE,
                number=2,
                state="open",
                title="Issue",
                body="",
                author="bob",
            ),
        ],
    )

    (repo_path / ".carapace.yaml").write_text(
        f"""
storage:
  backend: sqlite
  sqlite_path: {db_path}
  persist_runs: false
"""
    )

    exit_code = cli.main(
        [
            "process-stored",
            "--repo",
            "acme/repo",
            "--repo-path",
            str(repo_path),
            "--output-dir",
            str(out_dir),
            "--skip-repo-path-check",
            "--entity-kind",
            "issue",
        ]
    )
    assert exit_code == 0


def test_process_stored_enrich_failure_does_not_set_watermark(tmp_path: Path, monkeypatch) -> None:
    from carapace.storage import SQLiteStorage

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    out_dir = tmp_path / "out"
    db_path = repo_path / ".carapace" / "carapace.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    storage = SQLiteStorage(db_path)
    storage.upsert_ingest_entities(
        "acme/repo",
        [
            SourceEntity(
                id="pr:1",
                repo="acme/repo",
                kind=EntityKind.PR,
                number=1,
                state="open",
                title="Needs enrich",
                body="",
                author="alice",
                changed_files=[],
                ci_status=CIStatus.UNKNOWN,
            )
        ],
    )

    (repo_path / ".carapace.yaml").write_text(
        f"""
storage:
  backend: sqlite
  sqlite_path: {db_path}
  persist_runs: false
"""
    )

    class FailingSource:
        def __init__(self, repo: str, gh_bin: str = "gh", **kwargs: object) -> None:
            _ = (repo, gh_bin, kwargs)

        def enrich_entity(
            self,
            entity: SourceEntity,
            include_comments: bool = False,
            mode: str = "minimal",
            include_simple_scores: bool = False,
        ) -> SourceEntity:
            _ = (include_comments, mode, include_simple_scores)
            raise RuntimeError("boom")

    monkeypatch.setattr(cli, "GithubGhSourceConnector", FailingSource)

    exit_code = cli.main(
        [
            "process-stored",
            "--repo",
            "acme/repo",
            "--repo-path",
            str(repo_path),
            "--output-dir",
            str(out_dir),
            "--skip-repo-path-check",
            "--enrich-missing",
            "--enrich-mode",
            "minimal",
            "--enrich-workers",
            "1",
            "--enrich-flush-every",
            "1",
            "--enrich-progress-every",
            "1",
            "--enrich-heartbeat-seconds",
            "1",
        ]
    )
    assert exit_code == 0

    watermarks = storage.get_enrichment_watermarks("acme/repo", kind="pr")
    assert watermarks["pr:1"]["enriched_for_updated_at"] is None


def test_process_stored_enrich_rate_limit_stops_gracefully(tmp_path: Path, monkeypatch) -> None:
    from carapace.storage import SQLiteStorage

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    out_dir = tmp_path / "out"
    db_path = repo_path / ".carapace" / "carapace.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    storage = SQLiteStorage(db_path)
    storage.upsert_ingest_entities(
        "acme/repo",
        [
            SourceEntity(
                id="pr:1",
                repo="acme/repo",
                kind=EntityKind.PR,
                number=1,
                state="open",
                title="Needs enrich 1",
                body="",
                author="alice",
                changed_files=[],
                ci_status=CIStatus.UNKNOWN,
            ),
            SourceEntity(
                id="pr:2",
                repo="acme/repo",
                kind=EntityKind.PR,
                number=2,
                state="open",
                title="Needs enrich 2",
                body="",
                author="bob",
                changed_files=[],
                ci_status=CIStatus.UNKNOWN,
            ),
        ],
    )

    (repo_path / ".carapace.yaml").write_text(
        f"""
storage:
  backend: sqlite
  sqlite_path: {db_path}
  persist_runs: false
"""
    )

    class RateLimitedSource:
        def __init__(self, repo: str, gh_bin: str = "gh", **kwargs: object) -> None:
            _ = (repo, gh_bin, kwargs)

        def enrich_entity(
            self,
            entity: SourceEntity,
            include_comments: bool = False,
            mode: str = "minimal",
            include_simple_scores: bool = False,
        ) -> SourceEntity:
            _ = (include_comments, mode, include_simple_scores)
            raise GithubRateLimitError(
                "API rate limit exceeded",
                reset_at=datetime(2026, 2, 16, 1, 0, tzinfo=UTC),
            )

    monkeypatch.setattr(cli, "GithubGhSourceConnector", RateLimitedSource)

    exit_code = cli.main(
        [
            "process-stored",
            "--repo",
            "acme/repo",
            "--repo-path",
            str(repo_path),
            "--output-dir",
            str(out_dir),
            "--skip-repo-path-check",
            "--enrich-missing",
            "--enrich-mode",
            "minimal",
            "--enrich-workers",
            "1",
            "--enrich-flush-every",
            "1",
            "--enrich-progress-every",
            "1",
            "--enrich-heartbeat-seconds",
            "1",
        ]
    )
    assert exit_code == 0

    watermarks = storage.get_enrichment_watermarks("acme/repo", kind="pr")
    assert watermarks["pr:1"]["enriched_for_updated_at"] is None
    assert watermarks["pr:2"]["enriched_for_updated_at"] is None


def test_db_audit_command_outputs_summary(tmp_path: Path, capsys) -> None:
    from carapace.storage import SQLiteStorage

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    db_path = repo_path / ".carapace" / "carapace.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    storage = SQLiteStorage(db_path)
    storage.upsert_ingest_entities(
        "acme/repo",
        [
            SourceEntity(
                id="pr:1",
                repo="acme/repo",
                kind=EntityKind.PR,
                number=1,
                state="open",
                title="Needs enrich",
                body="",
                author="alice",
                changed_files=[],
                ci_status=CIStatus.UNKNOWN,
            )
        ],
    )

    (repo_path / ".carapace.yaml").write_text(
        f"""
storage:
  backend: sqlite
  sqlite_path: {db_path}
  persist_runs: false
"""
    )

    exit_code = cli.main(
        [
            "db-audit",
            "--repo",
            "acme/repo",
            "--repo-path",
            str(repo_path),
            "--skip-repo-path-check",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Repo: acme/repo" in out
    assert "Entities: total=1 prs=1 issues=0" in out
    assert "Integrity:" in out


def test_db_audit_command_can_emit_json(tmp_path: Path, capsys) -> None:
    from carapace.storage import SQLiteStorage

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    db_path = repo_path / ".carapace" / "carapace.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = SQLiteStorage(db_path)
    storage.upsert_ingest_entities(
        "acme/repo",
        [
            SourceEntity(
                id="issue:2",
                repo="acme/repo",
                kind=EntityKind.ISSUE,
                number=2,
                state="open",
                title="Issue",
                body="",
                author="bob",
            )
        ],
    )

    (repo_path / ".carapace.yaml").write_text(
        f"""
storage:
  backend: sqlite
  sqlite_path: {db_path}
  persist_runs: false
"""
    )

    exit_code = cli.main(
        [
            "db-audit",
            "--repo",
            "acme/repo",
            "--repo-path",
            str(repo_path),
            "--skip-repo-path-check",
            "--json",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["repo"] == "acme/repo"
    assert payload["summary"]["total"] == 1


def test_enrich_stored_command_enriches_prs(tmp_path: Path, monkeypatch) -> None:
    from carapace.storage import SQLiteStorage

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    db_path = repo_path / ".carapace" / "carapace.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = SQLiteStorage(db_path)
    storage.upsert_ingest_entities(
        "acme/repo",
        [
            SourceEntity(
                id="pr:1",
                repo="acme/repo",
                kind=EntityKind.PR,
                number=1,
                state="open",
                title="PR",
                body="",
                author="alice",
                changed_files=[],
            )
        ],
    )

    (repo_path / ".carapace.yaml").write_text(
        f"""
storage:
  backend: sqlite
  sqlite_path: {db_path}
  persist_runs: false
"""
    )

    class FakeSource:
        def __init__(self, repo: str, gh_bin: str = "gh", **kwargs: object) -> None:
            _ = (repo, gh_bin, kwargs)

        def enrich_entity(
            self,
            entity: SourceEntity,
            include_comments: bool = False,
            mode: str = "minimal",
            include_simple_scores: bool = False,
        ) -> SourceEntity:
            _ = (include_comments, mode, include_simple_scores)
            return entity.model_copy(update={"changed_files": ["src/a.py"]})

    monkeypatch.setattr(cli, "GithubGhSourceConnector", FakeSource)

    exit_code = cli.main(
        [
            "enrich-stored",
            "--repo",
            "acme/repo",
            "--repo-path",
            str(repo_path),
            "--skip-repo-path-check",
            "--enrich-workers",
            "1",
            "--enrich-flush-every",
            "1",
        ]
    )
    assert exit_code == 0
    loaded = storage.load_ingested_entities("acme/repo", include_closed=True, include_drafts=True)
    assert loaded[0].changed_files == ["src/a.py"]
