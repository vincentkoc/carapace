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


def build_diff_text(entity: SourceEntity) -> str:
    parts: list[str] = []
    if entity.changed_files:
        parts.append("files:" + " ".join(entity.changed_files))
    if entity.diff_hunks:
        hunk_texts: list[str] = []
        for hunk in entity.diff_hunks[:200]:
            hunk_texts.append(
                " ".join(
                    [
                        hunk.file_path,
                        hunk.context or "",
                        " ".join(hunk.added_lines[:10]),
                        " ".join(hunk.removed_lines[:10]),
                    ]
                )
            )
        parts.append("hunks:" + " ".join(hunk_texts))
    return "\n".join(parts)


def build_fingerprint(
    entity: SourceEntity,
    text_embedding: list[float],
    diff_embedding: list[float] | None = None,
) -> Fingerprint:
    title_tokens = _tokens(entity.title)
    body_tokens = _tokens(_normalize_body_for_tokens(entity))
    modules = sorted({_module_bucket(path) for path in entity.changed_files})

    reviewer_score = 0.0
    if entity.external_reviews:
        reviewer_score = sum(sig.overall_score for sig in entity.external_reviews) / len(entity.external_reviews)

    hunk_signatures = [_hunk_signature(h.file_path, h.context, h.added_lines, h.removed_lines) for h in entity.diff_hunks]
    linked_issues = set(entity.linked_issues)
    if entity.kind == EntityKind.ISSUE and entity.number is not None:
        linked_issues.add(str(entity.number))

    return Fingerprint(
        entity_id=entity.id,
        title_tokens=title_tokens,
        tokens=title_tokens + body_tokens,
        module_buckets=modules,
        changed_files=entity.changed_files,
        hunk_signatures=hunk_signatures,
        linked_issues=sorted(linked_issues),
        soft_linked_issues=entity.soft_linked_issues,
        commits=entity.commits,
        patch_ids=entity.patch_ids,
        additions=entity.additions,
        deletions=entity.deletions,
        ci_status=entity.ci_status,
        approvals=entity.approvals,
        reviewer_score=reviewer_score,
        text_embedding=text_embedding,
        diff_embedding=diff_embedding or [],
        embedding=text_embedding,
    )
