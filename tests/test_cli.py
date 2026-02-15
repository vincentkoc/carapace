import json
from pathlib import Path

from carapace import cli
from carapace.models import CIStatus, EntityKind, SourceEntity


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


def test_scan_github_command_uses_connector_and_can_save_input(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    class FakeSource:
        def __init__(self, repo: str, gh_bin: str = "gh") -> None:
            _ = (repo, gh_bin)

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
        def __init__(self, repo: str, gh_bin: str = "gh") -> None:
            _ = (repo, gh_bin)

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
        """
storage:
  backend: sqlite
  sqlite_path: .carapace/carapace.db
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
        """
storage:
  backend: sqlite
  sqlite_path: .carapace/carapace.db
  persist_runs: false
"""
    )

    class FailingSource:
        def __init__(self, repo: str, gh_bin: str = "gh") -> None:
            _ = (repo, gh_bin)

        def enrich_entity(self, entity: SourceEntity, include_comments: bool = False, mode: str = "minimal") -> SourceEntity:
            _ = (include_comments, mode)
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
