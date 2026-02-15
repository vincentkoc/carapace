"""Similarity retrieval and pair scoring."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass

from carapace.algorithms import (
    minhash_lsh_bands,
    minhash_signature,
    minhash_similarity,
    simhash64,
    simhash_chunks,
    simhash_similarity,
    winnowing_fingerprints,
)
from carapace.config import SimilarityConfig
from carapace.models import EdgeTier, Fingerprint, SimilarityBreakdown, SimilarityEdge

logger = logging.getLogger(__name__)


@dataclass
class CandidateIndex:
    by_module: dict[str, set[str]]
    by_issue: dict[str, set[str]]
    by_soft_issue: dict[str, set[str]]
    by_file: dict[str, set[str]]
    by_lsh_band: dict[tuple[int, tuple[int, ...]], set[str]]
    by_simhash_chunk: dict[tuple[int, int], set[str]]
    by_winnow_hash: dict[int, set[str]]
    minhash_signatures: dict[str, list[int]]
    simhash_values: dict[str, int]
    winnow_sets: dict[str, set[int]]


@dataclass
class SimilarityComputationStats:
    entities_total: int = 0
    candidate_links_generated: int = 0
    unique_pairs_scored: int = 0
    edges_emitted: int = 0


def _jaccard(items_a: set[str] | set[int], items_b: set[str] | set[int]) -> float:
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


def _material_tokens(fp: Fingerprint) -> list[str]:
    tokens: list[str] = []
    tokens.extend(fp.tokens)
    tokens.extend(fp.linked_issues)
    tokens.extend(fp.soft_linked_issues)
    tokens.extend(fp.hunk_signatures)
    return [token.lower() for token in tokens if token]


def build_candidate_index(fingerprints: dict[str, Fingerprint], cfg: SimilarityConfig) -> CandidateIndex:
    by_module: dict[str, set[str]] = defaultdict(set)
    by_issue: dict[str, set[str]] = defaultdict(set)
    by_soft_issue: dict[str, set[str]] = defaultdict(set)
    by_file: dict[str, set[str]] = defaultdict(set)

    by_lsh_band: dict[tuple[int, tuple[int, ...]], set[str]] = defaultdict(set)
    by_simhash_chunk: dict[tuple[int, int], set[str]] = defaultdict(set)
    by_winnow_hash: dict[int, set[str]] = defaultdict(set)

    minhash_signatures: dict[str, list[int]] = {}
    simhash_values: dict[str, int] = {}
    winnow_sets: dict[str, set[int]] = {}

    for entity_id, fp in fingerprints.items():
        for module in fp.module_buckets:
            by_module[module].add(entity_id)
        for issue in fp.linked_issues:
            by_issue[issue].add(entity_id)
        for issue in fp.soft_linked_issues:
            by_soft_issue[issue].add(entity_id)
        for path in fp.changed_files:
            by_file[path].add(entity_id)

        if not cfg.use_advanced_algorithms:
            continue

        tokens = _material_tokens(fp)

        signature = minhash_signature(
            tokens,
            num_perm=cfg.minhash_num_perm,
            shingle_k=cfg.minhash_shingle_k,
        )
        minhash_signatures[entity_id] = signature
        for band in minhash_lsh_bands(signature, bands=cfg.minhash_bands):
            by_lsh_band[band].add(entity_id)

        simhash_value = simhash64(tokens, bits=cfg.simhash_bits)
        simhash_values[entity_id] = simhash_value
        for chunk in simhash_chunks(simhash_value, bits=cfg.simhash_bits, chunk_bits=cfg.simhash_chunk_bits):
            by_simhash_chunk[chunk].add(entity_id)

        winnow = winnowing_fingerprints(tokens, k=cfg.winnow_kgram, window=cfg.winnow_window)
        winnow_sets[entity_id] = winnow
        for hash_value in winnow:
            by_winnow_hash[hash_value].add(entity_id)

    return CandidateIndex(
        by_module=by_module,
        by_issue=by_issue,
        by_soft_issue=by_soft_issue,
        by_file=by_file,
        by_lsh_band=by_lsh_band,
        by_simhash_chunk=by_simhash_chunk,
        by_winnow_hash=by_winnow_hash,
        minhash_signatures=minhash_signatures,
        simhash_values=simhash_values,
        winnow_sets=winnow_sets,
    )


def retrieve_candidates(
    entity_id: str,
    fp: Fingerprint,
    idx: CandidateIndex,
    cfg: SimilarityConfig,
    *,
    total_entities: int | None = None,
) -> list[str]:
    candidate_votes: dict[str, int] = defaultdict(int)

    def bump(values: set[str]) -> None:
        for candidate in values:
            if candidate != entity_id:
                candidate_votes[candidate] += 1

    for module in fp.module_buckets:
        bump(idx.by_module.get(module, set()))
    for issue in fp.linked_issues:
        bump(idx.by_issue.get(issue, set()))
    for issue in fp.soft_linked_issues:
        bump(idx.by_soft_issue.get(issue, set()))
    for path in fp.changed_files:
        bump(idx.by_file.get(path, set()))

    if cfg.use_advanced_algorithms:
        signature = idx.minhash_signatures.get(entity_id)
        if signature:
            for band in minhash_lsh_bands(signature, bands=cfg.minhash_bands):
                bump(idx.by_lsh_band.get(band, set()))

        simhash_value = idx.simhash_values.get(entity_id)
        if simhash_value is not None:
            for chunk in simhash_chunks(simhash_value, bits=cfg.simhash_bits, chunk_bits=cfg.simhash_chunk_bits):
                bump(idx.by_simhash_chunk.get(chunk, set()))

        for hash_value in idx.winnow_sets.get(entity_id, set()):
            bump(idx.by_winnow_hash.get(hash_value, set()))

    vote_floor = cfg.min_candidate_votes
    if total_entities is not None and total_entities >= cfg.large_run_threshold:
        vote_floor = max(vote_floor, cfg.min_candidate_votes_large)

    ranked = sorted(
        ((candidate, votes) for candidate, votes in candidate_votes.items() if votes >= vote_floor),
        key=lambda item: (-item[1], item[0]),
    )
    return [candidate for candidate, _ in ranked[: cfg.top_k_candidates]]


def _advanced_scores(
    a: Fingerprint,
    b: Fingerprint,
    idx: CandidateIndex,
    cfg: SimilarityConfig,
) -> tuple[float, float, float]:
    if not cfg.use_advanced_algorithms:
        return 0.0, 0.0, 0.0

    sig_a = idx.minhash_signatures.get(a.entity_id)
    sig_b = idx.minhash_signatures.get(b.entity_id)
    minhash = minhash_similarity(sig_a or [], sig_b or [])

    sim_a = idx.simhash_values.get(a.entity_id, 0)
    sim_b = idx.simhash_values.get(b.entity_id, 0)
    simhash = simhash_similarity(sim_a, sim_b, bits=cfg.simhash_bits)

    winnow_a = idx.winnow_sets.get(a.entity_id, set())
    winnow_b = idx.winnow_sets.get(b.entity_id, set())
    winnow = _jaccard(winnow_a, winnow_b)

    return minhash, simhash, winnow


def score_pair(
    a: Fingerprint,
    b: Fingerprint,
    cfg: SimilarityConfig,
    idx: CandidateIndex | None = None,
) -> tuple[float, SimilarityBreakdown]:
    patch_a = set(a.patch_ids)
    patch_b = set(b.patch_ids)
    file_a = set(a.changed_files)
    file_b = set(b.changed_files)
    hunk_a = set(a.hunk_signatures)
    hunk_b = set(b.hunk_signatures)
    hard_issue_a = set(a.linked_issues)
    hard_issue_b = set(b.linked_issues)
    soft_issue_a = set(a.soft_linked_issues)
    soft_issue_b = set(b.soft_linked_issues)

    file_overlap = _jaccard(file_a, file_b)
    hunk_overlap = _jaccard(hunk_a, hunk_b)
    hard_link_overlap = _jaccard(hard_issue_a, hard_issue_b)
    soft_link_overlap = _jaccard(soft_issue_a, soft_issue_b)
    lineage = max(_jaccard(patch_a, patch_b), _overlap_min(patch_a, patch_b))
    structure = 0.7 * hunk_overlap + 0.3 * file_overlap
    text_a = a.text_embedding or a.embedding
    text_b = b.text_embedding or b.embedding
    semantic_text = max(0.0, _cosine(text_a, text_b))
    semantic_diff = max(0.0, _cosine(a.diff_embedding, b.diff_embedding))
    semantic_weight_total = max(1e-9, cfg.semantic_text_share + cfg.semantic_diff_share)
    semantic = (
        cfg.semantic_text_share * semantic_text + cfg.semantic_diff_share * semantic_diff
    ) / semantic_weight_total

    minhash = 0.0
    simhash = 0.0
    winnow = 0.0
    if idx is not None:
        minhash, simhash, winnow = _advanced_scores(a, b, idx, cfg)

    churn_a = max(1, a.additions + a.deletions)
    churn_b = max(1, b.additions + b.deletions)
    size_ratio = max(churn_a, churn_b) / min(churn_a, churn_b)
    size_penalty = min(1.0, max(0.0, math.log10(size_ratio)))

    total = cfg.weight_lineage * lineage + cfg.weight_structure * structure + cfg.weight_semantic * semantic + cfg.weight_minhash * minhash + cfg.weight_simhash * simhash + cfg.weight_winnow * winnow - cfg.size_penalty_weight * size_penalty

    breakdown = SimilarityBreakdown(
        lineage=lineage,
        structure=structure,
        file_overlap=file_overlap,
        hunk_overlap=hunk_overlap,
        hard_link_overlap=hard_link_overlap,
        soft_link_overlap=soft_link_overlap,
        semantic=semantic,
        semantic_text=semantic_text,
        semantic_diff=semantic_diff,
        minhash=minhash,
        simhash=simhash,
        winnow=winnow,
        size_penalty=size_penalty,
        total=total,
    )
    return total, breakdown


def _edge_tier(score: float, breakdown: SimilarityBreakdown, cfg: SimilarityConfig) -> EdgeTier | None:
    if breakdown.hard_link_overlap > 0.0:
        return EdgeTier.STRONG

    has_structure = (breakdown.structure > 0.0) or (breakdown.lineage > 0.0)
    has_lineage_or_hunk = (breakdown.lineage > 0.0) or (breakdown.hunk_overlap > 0.0)
    if breakdown.lineage >= cfg.lineage_strong_overlap or score >= cfg.strong_score:
        return EdgeTier.STRONG
    if has_lineage_or_hunk and (breakdown.minhash >= cfg.strong_minhash_min or breakdown.winnow >= cfg.strong_winnow_min):
        return EdgeTier.STRONG

    # For unstructured entities (e.g., issue templates), require very high semantic + lexical agreement.
    if not has_structure:
        if breakdown.soft_link_overlap > 0.0 and breakdown.semantic >= cfg.weak_semantic_min:
            return EdgeTier.WEAK
        if breakdown.semantic >= cfg.unstructured_semantic_min and breakdown.minhash >= cfg.unstructured_minhash_min and breakdown.winnow >= cfg.unstructured_winnow_min:
            return EdgeTier.WEAK
        return None

    if breakdown.soft_link_overlap > 0.0 and (
        breakdown.structure >= cfg.weak_structure_min or breakdown.semantic >= cfg.weak_semantic_min
    ):
        return EdgeTier.WEAK

    if score >= cfg.min_score and (
        breakdown.structure >= cfg.weak_structure_min or breakdown.semantic >= cfg.weak_semantic_min or breakdown.minhash >= cfg.weak_minhash_min or breakdown.simhash >= cfg.weak_simhash_min or breakdown.winnow >= cfg.weak_winnow_min
    ):
        return EdgeTier.WEAK
    return None


def compute_similarity_edges_with_stats(
    fingerprints: dict[str, Fingerprint],
    cfg: SimilarityConfig,
) -> tuple[list[SimilarityEdge], SimilarityComputationStats]:
    start = time.perf_counter()
    idx = build_candidate_index(fingerprints, cfg)
    pair_seen: set[tuple[str, str]] = set()
    edges: list[SimilarityEdge] = []
    stats = SimilarityComputationStats(entities_total=len(fingerprints))

    total = len(fingerprints)
    for counter, (entity_id, fp) in enumerate(fingerprints.items(), start=1):
        candidates = retrieve_candidates(entity_id, fp, idx, cfg, total_entities=total)
        stats.candidate_links_generated += len(candidates)
        for other_id in candidates:
            a, b = sorted((entity_id, other_id))
            pair = (a, b)
            if pair in pair_seen:
                continue
            pair_seen.add(pair)
            stats.unique_pairs_scored += 1

            score, breakdown = score_pair(fingerprints[a], fingerprints[b], cfg, idx=idx)
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
            stats.edges_emitted += 1

        if counter % 100 == 0 or counter == total:
            logger.debug(
                "Similarity progress: %s/%s entities, edges=%s, elapsed=%.2fs",
                counter,
                total,
                len(edges),
                time.perf_counter() - start,
            )
        if (total >= 1000 and counter % 500 == 0) or counter == total:
            logger.info(
                "Similarity progress: %s/%s entities, edges=%s, elapsed=%.2fs",
                counter,
                total,
                len(edges),
                time.perf_counter() - start,
            )

    return edges, stats


def compute_similarity_edges(fingerprints: dict[str, Fingerprint], cfg: SimilarityConfig) -> list[SimilarityEdge]:
    edges, _ = compute_similarity_edges_with_stats(fingerprints, cfg)
    return edges
