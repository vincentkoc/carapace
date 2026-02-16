from datetime import UTC, datetime, timedelta

from carapace.config import LowPassConfig
from carapace.low_pass import apply_low_pass
from carapace.models import CIStatus, EntityKind, FilterState, LowPassAction, SourceEntity


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


def test_issue_template_noise_suppressed() -> None:
    issue = _entity(
        kind=EntityKind.ISSUE,
        body="## Summary\n## Steps to reproduce\n## Expected behavior\n## Actual behavior\nCrash",
    )
    out = apply_low_pass(issue, LowPassConfig())
    assert out.state == FilterState.SUPPRESS
    assert "ISSUE_TEMPLATE_NOISE" in out.reason_codes


def test_issue_one_liner_suppressed() -> None:
    issue = _entity(kind=EntityKind.ISSUE, body="It fails.")
    out = apply_low_pass(issue, LowPassConfig())
    assert out.state == FilterState.SUPPRESS
    assert "ISSUE_ONE_LINER" in out.reason_codes


def test_pr_missing_context_suppressed() -> None:
    pr = _entity(kind=EntityKind.PR, body="", linked_issues=[], soft_linked_issues=[])
    out = apply_low_pass(pr, LowPassConfig(pr_min_body_tokens=8))
    assert out.state == FilterState.SUPPRESS
    assert "PR_MISSING_CONTEXT" in out.reason_codes


def test_pr_large_changeset_skips() -> None:
    pr = _entity(kind=EntityKind.PR, body="Has context", changed_files=[f"src/{i}.py" for i in range(45)], additions=200, deletions=20)
    out = apply_low_pass(pr, LowPassConfig())
    assert out.state == FilterState.SKIP
    assert "PR_LARGE_CHANGESET" in out.reason_codes


def test_pr_unenriched_old_pr_suppressed() -> None:
    pr = _entity(
        kind=EntityKind.PR,
        updated_at=datetime.now(UTC) - timedelta(hours=12),
        changed_files=[],
        diff_hunks=[],
    )
    out = apply_low_pass(pr, LowPassConfig(pr_unenriched_max_age_hours=6))
    assert out.state == FilterState.SUPPRESS
    assert "PR_UNENRICHED" in out.reason_codes


def test_pr_unenriched_can_be_configured_to_close() -> None:
    pr = _entity(
        kind=EntityKind.PR,
        updated_at=datetime.now(UTC) - timedelta(hours=24),
        changed_files=[],
        diff_hunks=[],
    )
    out = apply_low_pass(
        pr,
        LowPassConfig(pr_unenriched_max_age_hours=6, pr_unenriched_action="close"),
    )
    assert out.state == FilterState.SKIP
    assert out.action == LowPassAction.CLOSE


def test_issue_template_can_be_configured_to_close() -> None:
    issue = _entity(
        kind=EntityKind.ISSUE,
        body="## Summary\n## Steps to reproduce\n## Expected behavior\nBroken",
    )
    out = apply_low_pass(issue, LowPassConfig(issue_template_action="close"))
    assert out.state == FilterState.SKIP
    assert out.action == LowPassAction.CLOSE


def test_boost_priority() -> None:
    e = _entity(labels=["security"])
    out = apply_low_pass(e, LowPassConfig())
    assert out.state == FilterState.PASS
    assert out.priority_weight > 1.0
