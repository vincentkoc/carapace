"""Repository path validation utilities."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _extract_repo_slug(remote_url: str) -> str | None:
    url = remote_url.strip()

    m = re.search(r"github\.com[:/](?P<slug>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?$", url)
    if not m:
        return None
    return m.group("slug")


def validate_repo_path_matches(repo_path: str | Path, expected_repo: str) -> None:
    path = Path(repo_path)
    if not path.exists():
        raise ValueError(f"repo_path does not exist: {path}")

    proc = subprocess.run(
        ["git", "-C", str(path), "remote", "get-url", "origin"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Could not resolve git origin for repo_path={path}. "
            "Use a local clone of the target repository or pass --skip-repo-path-check."
        )

    detected = _extract_repo_slug(proc.stdout)
    if not detected:
        raise ValueError(f"Unable to parse GitHub repo slug from origin URL: {proc.stdout.strip()}")

    if detected.lower() != expected_repo.lower():
        raise ValueError(
            f"Repo mismatch: repo argument is '{expected_repo}' but repo_path origin is '{detected}'. "
            "Fix repo_path or pass --skip-repo-path-check."
        )
