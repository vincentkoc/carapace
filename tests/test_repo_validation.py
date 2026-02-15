from pathlib import Path

import pytest

from carapace.repo_validation import validate_repo_path_matches


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()

    import subprocess

    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", "https://github.com/openclaw/openclaw.git"],
        check=True,
        capture_output=True,
    )
    return repo


def test_repo_validation_passes_on_matching_slug(git_repo: Path) -> None:
    validate_repo_path_matches(git_repo, "openclaw/openclaw")


def test_repo_validation_fails_on_mismatch(git_repo: Path) -> None:
    with pytest.raises(ValueError):
        validate_repo_path_matches(git_repo, "acme/other")


def test_repo_validation_fails_without_origin(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    import subprocess

    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)

    with pytest.raises(ValueError):
        validate_repo_path_matches(repo, "openclaw/openclaw")
