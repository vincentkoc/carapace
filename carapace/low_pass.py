"""Low-pass filtering for noise suppression and prioritization."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from carapace.config import LowPassConfig
from carapace.models import CIStatus, EntityKind, FilterState, LowPassAction, LowPassDecision, SourceEntity

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_ISSUE_TEMPLATE_PREFIXES = (
    "## summary",
    "## steps to reproduce",
    "## expected behavior",
    "## actual behavior",
    "## proposed solution",
    "## additional context",
)
_ISSUE_TEMPLATE_LINES = {
    "what went wrong?",
    "what did you expect to happen?",
    "what actually happened?",
    "1.",
    "2.",
    "3.",
}


def _is_docs_only(entity: SourceEntity, cfg: LowPassConfig) -> bool:
    if not entity.changed_files:
        return False

    for path in entity.changed_files:
        normalized = path.lower()
        if any(normalized.startswith(prefix.lower()) for prefix in cfg.docs_only_prefixes):
            continue
        if any(normalized.endswith(suffix.lower()) for suffix in cfg.docs_only_suffixes):
            continue
        return False
    return True


def _is_stale(entity: SourceEntity, stale_days: int | None) -> bool:
    if stale_days is None:
        return False
    cutoff = datetime.now(UTC) - timedelta(days=stale_days)
    return entity.updated_at < cutoff


def _is_recent_pr(entity: SourceEntity, ignore_recent_pr_hours: int | None) -> bool:
    if ignore_recent_pr_hours is None:
        return False
    if entity.kind.value != "pr":
        return False
    cutoff = datetime.now(UTC) - timedelta(hours=ignore_recent_pr_hours)
    return entity.updated_at >= cutoff


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


def _issue_template_noise(body: str, cfg: LowPassConfig) -> bool:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return False

    template_lines = 0
    content_lines: list[str] = []
    for line in lines:
        lower = line.lower()
        if any(lower.startswith(prefix) for prefix in _ISSUE_TEMPLATE_PREFIXES) or lower in _ISSUE_TEMPLATE_LINES:
            template_lines += 1
            continue
        content_lines.append(line)

    ratio = template_lines / max(1, len(lines))
    content_tokens = len(_tokens("\n".join(content_lines)))
    return ratio >= cfg.issue_template_match_threshold and content_tokens <= cfg.issue_template_max_content_tokens


def _issue_one_liner_noise(body: str, cfg: LowPassConfig) -> bool:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    content_lines: list[str] = []
    for line in lines:
        lower = line.lower()
        if any(lower.startswith(prefix) for prefix in _ISSUE_TEMPLATE_PREFIXES) or lower in _ISSUE_TEMPLATE_LINES:
            continue
        content_lines.append(line)
    return len(content_lines) <= 1 and len(_tokens("\n".join(content_lines))) <= cfg.issue_one_liner_max_tokens


def _pr_missing_context(entity: SourceEntity, cfg: LowPassConfig) -> bool:
    if entity.kind != EntityKind.PR:
        return False
    if cfg.pr_min_body_tokens <= 0:
        return False
    body_tokens = len(_tokens(entity.body))
    if body_tokens >= cfg.pr_min_body_tokens:
        return False
    # No context and no explicit issue linkage.
    return len(entity.linked_issues) == 0 and len(entity.soft_linked_issues) == 0


def _pr_large(entity: SourceEntity, cfg: LowPassConfig) -> bool:
    if entity.kind != EntityKind.PR:
        return False
    file_count = len(entity.changed_files)
    churn = max(0, entity.additions) + max(0, entity.deletions)
    return file_count > cfg.pr_large_max_files or churn > cfg.pr_large_max_churn


def _pr_unenriched(entity: SourceEntity, cfg: LowPassConfig) -> bool:
    if entity.kind != EntityKind.PR:
        return False
    if cfg.pr_unenriched_max_age_hours is None:
        return False
    if entity.changed_files or entity.diff_hunks:
        return False
    cutoff = datetime.now(UTC) - timedelta(hours=cfg.pr_unenriched_max_age_hours)
    return entity.updated_at < cutoff


def _to_state(action: str, default: FilterState = FilterState.SUPPRESS) -> FilterState:
    normalized = (action or "").strip().lower()
    if normalized in {"close", "skip"}:
        return FilterState.SKIP
    if normalized == "pass":
        return FilterState.PASS
    return default


def _to_action(action: str) -> LowPassAction:
    return LowPassAction.CLOSE if (action or "").strip().lower() == "close" else LowPassAction.IGNORE


def apply_low_pass(entity: SourceEntity, cfg: LowPassConfig) -> LowPassDecision:
    reasons: list[str] = []

    label_set = {label.lower() for label in entity.labels}
    hard_skip = {label.lower() for label in cfg.hard_skip_labels}
    soft_suppress = {label.lower() for label in cfg.soft_suppress_labels}

    if cfg.skip_closed and entity.state.lower() != "open":
        reasons.append("CLOSED_STATE")
        return LowPassDecision(entity_id=entity.id, state=FilterState.SKIP, reason_codes=reasons, priority_weight=0.0)

    if cfg.skip_drafts and entity.is_draft:
        reasons.append("DRAFT_PR")
        return LowPassDecision(entity_id=entity.id, state=FilterState.SKIP, reason_codes=reasons, priority_weight=0.0)

    if label_set & hard_skip:
        reasons.append("HARD_SKIP_LABEL")
        return LowPassDecision(entity_id=entity.id, state=FilterState.SKIP, reason_codes=reasons, priority_weight=0.0)

    if _is_stale(entity, cfg.stale_days):
        reasons.append("STALE")
        return LowPassDecision(
            entity_id=entity.id,
            state=_to_state(cfg.stale_action, default=FilterState.SKIP),
            action=_to_action(cfg.stale_action),
            reason_codes=reasons,
            priority_weight=0.0,
        )

    if _is_recent_pr(entity, cfg.ignore_recent_pr_hours):
        reasons.append("RECENT_PR")
        return LowPassDecision(entity_id=entity.id, state=FilterState.SUPPRESS, reason_codes=reasons, priority_weight=1.0)

    if entity.kind == EntityKind.ISSUE and _issue_template_noise(entity.body or "", cfg):
        reasons.append("ISSUE_TEMPLATE_NOISE")
        action = cfg.issue_template_action
        return LowPassDecision(
            entity_id=entity.id,
            state=_to_state(action),
            action=_to_action(action),
            reason_codes=reasons,
            priority_weight=0.0,
        )

    if entity.kind == EntityKind.ISSUE and _issue_one_liner_noise(entity.body or "", cfg):
        reasons.append("ISSUE_ONE_LINER")
        action = cfg.issue_one_liner_action
        return LowPassDecision(
            entity_id=entity.id,
            state=_to_state(action),
            action=_to_action(action),
            reason_codes=reasons,
            priority_weight=0.0,
        )

    if _pr_missing_context(entity, cfg):
        reasons.append("PR_MISSING_CONTEXT")
        action = cfg.pr_missing_context_action
        return LowPassDecision(
            entity_id=entity.id,
            state=_to_state(action),
            action=_to_action(action),
            reason_codes=reasons,
            priority_weight=0.0,
        )

    if _pr_large(entity, cfg):
        reasons.append("PR_LARGE_CHANGESET")
        action = cfg.pr_large_action
        return LowPassDecision(
            entity_id=entity.id,
            state=_to_state(action, default=FilterState.SKIP),
            action=_to_action(action),
            reason_codes=reasons,
            priority_weight=0.0,
        )

    if _pr_unenriched(entity, cfg):
        reasons.append("PR_UNENRICHED")
        action = cfg.pr_unenriched_action
        return LowPassDecision(
            entity_id=entity.id,
            state=_to_state(action, default=FilterState.SUPPRESS),
            action=_to_action(action),
            reason_codes=reasons,
            priority_weight=0.0,
        )

    if label_set & soft_suppress:
        reasons.append("SOFT_SUPPRESS_LABEL")

    if cfg.suppress_docs_only_if_no_ci and _is_docs_only(entity, cfg) and entity.ci_status == CIStatus.UNKNOWN:
        reasons.append("LOW_SIGNAL_DOCS_ONLY")

    if cfg.suppress_bot_authors:
        author = entity.author.lower()
        if entity.is_bot or any(pattern.lower() in author for pattern in cfg.bot_author_patterns):
            reasons.append("BOT_SUPPRESSED")

    priority_weight = 1.0
    for label, weight in cfg.boost_labels.items():
        if label.lower() in label_set:
            priority_weight = max(priority_weight, weight)
            reasons.append(f"BOOST_{label.upper().replace('-', '_')}")

    if reasons and any(code in {"SOFT_SUPPRESS_LABEL", "LOW_SIGNAL_DOCS_ONLY", "BOT_SUPPRESSED"} for code in reasons):
        state = FilterState.SUPPRESS
    else:
        state = FilterState.PASS

    return LowPassDecision(entity_id=entity.id, state=state, reason_codes=reasons, priority_weight=priority_weight)
