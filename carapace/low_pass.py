"""Low-pass filtering for noise suppression and prioritization."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from carapace.config import LowPassConfig
from carapace.models import CIStatus, FilterState, LowPassDecision, SourceEntity


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
        return LowPassDecision(entity_id=entity.id, state=FilterState.SKIP, reason_codes=reasons, priority_weight=0.0)

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

    if reasons and any(
        code in {"SOFT_SUPPRESS_LABEL", "LOW_SIGNAL_DOCS_ONLY", "BOT_SUPPRESSED"} for code in reasons
    ):
        state = FilterState.SUPPRESS
    else:
        state = FilterState.PASS

    return LowPassDecision(entity_id=entity.id, state=state, reason_codes=reasons, priority_weight=priority_weight)
