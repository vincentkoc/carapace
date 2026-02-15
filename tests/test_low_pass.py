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


def test_closed_state_skips() -> None:
    e = _entity(state="closed")
    out = apply_low_pass(e, LowPassConfig(skip_closed=True))
    assert out.state == FilterState.SKIP
    assert "CLOSED_STATE" in out.reason_codes


def test_draft_skips() -> None:
    e = _entity(is_draft=True)
    out = apply_low_pass(e, LowPassConfig(skip_drafts=True))
    assert out.state == FilterState.SKIP
    assert "DRAFT_PR" in out.reason_codes


def test_docs_only_suppression() -> None:
    e = _entity(changed_files=["docs/readme.md"], ci_status=CIStatus.UNKNOWN)
    out = apply_low_pass(e, LowPassConfig())
    assert out.state == FilterState.SUPPRESS


def test_recent_pr_suppressed() -> None:
    e = _entity(updated_at=datetime.now(UTC))
    out = apply_low_pass(e, LowPassConfig(ignore_recent_pr_hours=4))
    assert out.state == FilterState.SUPPRESS
    assert "RECENT_PR" in out.reason_codes


def test_stale_skips() -> None:
    e = _entity(updated_at=datetime.now(UTC) - timedelta(days=200))
    out = apply_low_pass(e, LowPassConfig(stale_days=90))
    assert out.state == FilterState.SKIP


def test_boost_priority() -> None:
    e = _entity(labels=["security"])
    out = apply_low_pass(e, LowPassConfig())
    assert out.state == FilterState.PASS
    assert out.priority_weight > 1.0
