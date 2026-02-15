"""Fingerprint generation from source entities."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from carapace.models import EntityKind, Fingerprint, SourceEntity

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def _module_bucket(path: str, depth: int = 2) -> str:
    parts = [part for part in path.split("/") if part]
    if not parts:
        return "root/*"
    return "/".join(parts[:depth]) + "/*"


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


def _normalize_body_for_tokens(entity: SourceEntity) -> str:
    text = entity.body or ""
    if entity.kind != EntityKind.ISSUE:
        return text

    kept: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if not line:
            continue
        if any(lower.startswith(prefix) for prefix in _ISSUE_TEMPLATE_PREFIXES):
            continue
        if lower in _ISSUE_TEMPLATE_LINES:
            continue
        kept.append(line)
    return "\n".join(kept)


def _hunk_signature(file_path: str, context: str, added: Iterable[str], removed: Iterable[str]) -> str:
    added_tokens = _tokens(" ".join(added))
    removed_tokens = _tokens(" ".join(removed))
    material = "|".join(
        [
            file_path,
            context,
            " ".join(added_tokens[:64]),
            " ".join(removed_tokens[:64]),
        ]
    )
    return hashlib.blake2b(material.encode("utf-8"), digest_size=20).hexdigest()


def build_fingerprint(entity: SourceEntity, embedding: list[float]) -> Fingerprint:
    title_tokens = _tokens(entity.title)
    body_tokens = _tokens(_normalize_body_for_tokens(entity))
    modules = sorted({_module_bucket(path) for path in entity.changed_files})

    reviewer_score = 0.0
    if entity.external_reviews:
        reviewer_score = sum(sig.overall_score for sig in entity.external_reviews) / len(entity.external_reviews)

    hunk_signatures = [_hunk_signature(h.file_path, h.context, h.added_lines, h.removed_lines) for h in entity.diff_hunks]

    return Fingerprint(
        entity_id=entity.id,
        tokens=title_tokens + body_tokens,
        module_buckets=modules,
        changed_files=entity.changed_files,
        hunk_signatures=hunk_signatures,
        linked_issues=entity.linked_issues,
        commits=entity.commits,
        patch_ids=entity.patch_ids,
        additions=entity.additions,
        deletions=entity.deletions,
        ci_status=entity.ci_status,
        approvals=entity.approvals,
        reviewer_score=reviewer_score,
        embedding=embedding,
    )
