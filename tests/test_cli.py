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

        def fetch_open_entities(self, max_prs: int, include_issues: bool, max_issues: int):
            _ = (max_prs, include_issues, max_issues)
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
        ]
    )

    assert exit_code == 0
    assert (out_dir / "entities.json").exists()
    assert (out_dir / "labels_to_apply.json").exists()
