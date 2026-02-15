from pathlib import Path

from carapace.config import load_effective_config


def test_config_precedence(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / ".carapace.yaml").write_text(
        """
low_pass:
  stale_days: 7
labels:
  canonical: triage/canon-repo
"""
    )

    system = {
        "low_pass": {"stale_days": 90},
        "labels": {"canonical": "triage/canon-system"},
    }
    org = {
        "low_pass": {"stale_days": 30},
        "labels": {"canonical": "triage/canon-org"},
    }
    runtime = {
        "labels": {"canonical": "triage/canon-runtime"},
    }

    cfg = load_effective_config(repo, org_defaults=org, system_defaults=system, runtime_override=runtime)

    assert cfg.low_pass.stale_days == 7
    assert cfg.labels.canonical == "triage/canon-runtime"
