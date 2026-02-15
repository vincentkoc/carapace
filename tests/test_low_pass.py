from datetime import UTC, datetime, timedelta

from carapace.config import LowPassConfig
from carapace.low_pass import apply_low_pass
from carapace.models import CIStatus, EntityKind, FilterState, SourceEntity


def _entity(**kwargs):
    base = {
        "id": "1",
        "repo": "acme/repo",
        "kind": EntityKind.PR,
        "title": "Fix bug",
        "body": "",
        "author": "alice",
    }
    base.update(kwargs)
    return SourceEntity.model_validate(base)


def test_hard_skip_label() -> None:
    e = _entity(labels=["invalid"])
    out = apply_low_pass(e, LowPassConfig())
    assert out.state == FilterState.SKIP


def test_docs_only_suppression() -> None:
    e = _entity(changed_files=["docs/readme.md"], ci_status=CIStatus.UNKNOWN)
    out = apply_low_pass(e, LowPassConfig())
    assert out.state == FilterState.SUPPRESS


def test_stale_skips() -> None:
    e = _entity(updated_at=datetime.now(UTC) - timedelta(days=200))
    out = apply_low_pass(e, LowPassConfig(stale_days=90))
    assert out.state == FilterState.SKIP


def test_boost_priority() -> None:
    e = _entity(labels=["security"])
    out = apply_low_pass(e, LowPassConfig())
    assert out.state == FilterState.PASS
    assert out.priority_weight > 1.0
