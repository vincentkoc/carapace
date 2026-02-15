"""Similarity retrieval and pair scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from carapace.config import SimilarityConfig
from carapace.models import EdgeTier, Fingerprint, SimilarityBreakdown, SimilarityEdge


@dataclass
class CandidateIndex:
    by_module: dict[str, set[str]]
    by_issue: dict[str, set[str]]
    by_file: dict[str, set[str]]


def _jaccard(items_a: set[str], items_b: set[str]) -> float:
    if not items_a and not items_b:
        return 0.0
    union = items_a | items_b
    if not union:
        return 0.0
    return len(items_a & items_b) / len(union)


def _overlap_min(items_a: set[str], items_b: set[str]) -> float:
    if not items_a or not items_b:
        return 0.0
    return len(items_a & items_b) / max(1, min(len(items_a), len(items_b)))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def build_candidate_index(fingerprints: dict[str, Fingerprint]) -> CandidateIndex:
    by_module: dict[str, set[str]] = defaultdict(set)
    by_issue: dict[str, set[str]] = defaultdict(set)
    by_file: dict[str, set[str]] = defaultdict(set)

    for entity_id, fp in fingerprints.items():
        for module in fp.module_buckets:
            by_module[module].add(entity_id)
        for issue in fp.linked_issues:
            by_issue[issue].add(entity_id)
        for path in fp.changed_files:
            by_file[path].add(entity_id)

    return CandidateIndex(by_module=by_module, by_issue=by_issue, by_file=by_file)


def retrieve_candidates(entity_id: str, fp: Fingerprint, idx: CandidateIndex, top_k: int) -> list[str]:
    candidates: set[str] = set()

    for module in fp.module_buckets:
        candidates.update(idx.by_module.get(module, set()))
    for issue in fp.linked_issues:
        candidates.update(idx.by_issue.get(issue, set()))
    for path in fp.changed_files:
        candidates.update(idx.by_file.get(path, set()))

    candidates.discard(entity_id)
    return sorted(candidates)[:top_k]


def score_pair(a: Fingerprint, b: Fingerprint, cfg: SimilarityConfig) -> tuple[float, SimilarityBreakdown]:
    patch_a = set(a.patch_ids)
    patch_b = set(b.patch_ids)
    file_a = set(a.changed_files)
    file_b = set(b.changed_files)
    hunk_a = set(a.hunk_signatures)
    hunk_b = set(b.hunk_signatures)

    lineage = max(_jaccard(patch_a, patch_b), _overlap_min(patch_a, patch_b))
    structure = 0.6 * _jaccard(hunk_a, hunk_b) + 0.4 * _jaccard(file_a, file_b)
    semantic = max(0.0, _cosine(a.embedding, b.embedding))

    churn_a = max(1, a.additions + a.deletions)
    churn_b = max(1, b.additions + b.deletions)
    size_ratio = max(churn_a, churn_b) / min(churn_a, churn_b)
    size_penalty = min(1.0, max(0.0, math.log10(size_ratio)))

    total = (
        cfg.weight_lineage * lineage
        + cfg.weight_structure * structure
        + cfg.weight_semantic * semantic
        - cfg.size_penalty_weight * size_penalty
    )

    breakdown = SimilarityBreakdown(
        lineage=lineage,
        structure=structure,
        semantic=semantic,
        size_penalty=size_penalty,
        total=total,
    )
    return total, breakdown


def _edge_tier(score: float, breakdown: SimilarityBreakdown, cfg: SimilarityConfig) -> EdgeTier | None:
    if breakdown.lineage >= cfg.lineage_strong_overlap or score >= cfg.strong_score:
        return EdgeTier.STRONG
    if score >= cfg.min_score and (
        breakdown.structure >= cfg.weak_structure_min or breakdown.semantic >= cfg.weak_semantic_min
    ):
        return EdgeTier.WEAK
    return None


def compute_similarity_edges(fingerprints: dict[str, Fingerprint], cfg: SimilarityConfig) -> list[SimilarityEdge]:
    idx = build_candidate_index(fingerprints)
    pair_seen: set[tuple[str, str]] = set()
    edges: list[SimilarityEdge] = []

    for entity_id, fp in fingerprints.items():
        candidates = retrieve_candidates(entity_id, fp, idx, cfg.top_k_candidates)
        for other_id in candidates:
            a, b = sorted((entity_id, other_id))
            pair = (a, b)
            if pair in pair_seen:
                continue
            pair_seen.add(pair)

            score, breakdown = score_pair(fingerprints[a], fingerprints[b], cfg)
            tier = _edge_tier(score, breakdown, cfg)
            if tier is None:
                continue

            edges.append(
                SimilarityEdge(
                    entity_a=a,
                    entity_b=b,
                    score=score,
                    tier=tier,
                    breakdown=breakdown,
                )
            )

    return edges
