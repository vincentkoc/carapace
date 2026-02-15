"""Main orchestration pipeline for Carapace."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from collections import Counter

from carapace.canonical import rank_canonicals
from carapace.config import CarapaceConfig, load_effective_config
from carapace.embeddings.base import EmbeddingProvider
from carapace.embeddings.local_hash import LocalHashEmbeddingProvider
from carapace.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from carapace.fingerprint import build_fingerprint
from carapace.hooks import HookManager, HookName
from carapace.low_pass import apply_low_pass
from carapace.models import (
    DecisionState,
    EngineReport,
    FilterState,
    LowPassDecision,
    RoutingDecision,
    SourceEntity,
)
from carapace.clustering import build_clusters
from carapace.similarity import compute_similarity_edges
from carapace.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def _provider_from_config(config: CarapaceConfig) -> EmbeddingProvider:
    if config.embedding.provider == "openai-compatible":
        if not config.embedding.endpoint:
            raise ValueError("embedding.endpoint is required for openai-compatible provider")
        return OpenAICompatibleEmbeddingProvider(
            endpoint=config.embedding.endpoint,
            model=config.embedding.model,
            dimensions=config.embedding.dimensions,
            api_key_env=config.embedding.api_key_env,
            timeout_seconds=config.embedding.timeout_seconds,
        )

    return LocalHashEmbeddingProvider(dims=config.embedding.dimensions, model=config.embedding.model)


class CarapaceEngine:
    def __init__(
        self,
        config: CarapaceConfig,
        embedding_provider: EmbeddingProvider | None = None,
        hooks: HookManager | None = None,
        storage: StorageBackend | None = None,
    ) -> None:
        self.config = config
        self.embedding_provider = embedding_provider or _provider_from_config(config)
        self.hooks = hooks or HookManager()
        self.storage = storage or self._storage_from_config(config)
        self.last_fingerprints = {}

    @classmethod
    def from_repo(
        cls,
        repo_path: str | Path,
        org_defaults: dict | None = None,
        system_defaults: dict | None = None,
        runtime_override: dict | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        hooks: HookManager | None = None,
        storage: StorageBackend | None = None,
    ) -> "CarapaceEngine":
        config = load_effective_config(
            repo_path=repo_path,
            org_defaults=org_defaults,
            system_defaults=system_defaults,
            runtime_override=runtime_override,
        )
        return cls(config=config, embedding_provider=embedding_provider, hooks=hooks, storage=storage)

    def scan_entities(self, entities: list[SourceEntity]) -> EngineReport:
        scan_start = time.perf_counter()
        logger.info("Starting scan for %s entities", len(entities))
        context = {"entity_count": len(entities)}
        self.hooks.emit(HookName.BEFORE_LOW_PASS, context, {})

        low_pass: dict[str, LowPassDecision] = {}
        active_entities: list[SourceEntity] = []
        suppressed = 0
        skipped = 0

        for entity in entities:
            decision = apply_low_pass(entity, self.config.low_pass)
            low_pass[entity.id] = decision
            if decision.state == FilterState.PASS:
                active_entities.append(entity)
            elif decision.state == FilterState.SUPPRESS:
                suppressed += 1
            else:
                skipped += 1

        reason_counts: Counter[str] = Counter()
        for decision in low_pass.values():
            reason_counts.update(decision.reason_codes)

        self.hooks.emit(
            HookName.AFTER_LOW_PASS,
            context,
            {
                "active": len(active_entities),
                "suppressed": suppressed,
                "skipped": skipped,
            },
        )
        logger.debug(
            "Low-pass complete: active=%s suppressed=%s skipped=%s",
            len(active_entities),
            suppressed,
            skipped,
        )
        if reason_counts:
            logger.debug("Low-pass reason counts: %s", dict(reason_counts))
        if len(active_entities) == len(entities):
            logger.debug("Low-pass passed all entities. Tune low_pass config if more suppression is expected.")

        fingerprints = {}
        if active_entities:
            fp_start = time.perf_counter()
            self.hooks.emit(HookName.BEFORE_FINGERPRINT, context, {})
            texts = [f"{entity.title}\n{entity.body}" for entity in active_entities]
            vectors = self.embedding_provider.embed_texts(texts)
            for entity, vector in zip(active_entities, vectors):
                fingerprints[entity.id] = build_fingerprint(entity, vector)
            self.hooks.emit(HookName.AFTER_FINGERPRINT, context, {"count": len(fingerprints)})
            logger.debug(
                "Fingerprinting complete: %s fingerprints in %.2fs",
                len(fingerprints),
                time.perf_counter() - fp_start,
            )
        self.last_fingerprints = fingerprints

        sim_start = time.perf_counter()
        self.hooks.emit(HookName.BEFORE_SIMILARITY, context, {})
        edges = compute_similarity_edges(fingerprints, self.config.similarity)
        clusters = build_clusters([entity.id for entity in active_entities], edges)
        self.hooks.emit(HookName.AFTER_SIMILARITY, context, {"edges": len(edges), "clusters": len(clusters)})
        logger.debug(
            "Similarity complete: edges=%s clusters=%s in %.2fs",
            len(edges),
            len(clusters),
            time.perf_counter() - sim_start,
        )

        canonical_start = time.perf_counter()
        self.hooks.emit(HookName.BEFORE_CANONICAL, context, {})
        canonical = rank_canonicals(clusters, fingerprints, edges, low_pass, self.config.canonical)
        self.hooks.emit(HookName.AFTER_CANONICAL, context, {"decisions": len(canonical)})
        logger.debug(
            "Canonical ranking complete: %s cluster decisions in %.2fs",
            len(canonical),
            time.perf_counter() - canonical_start,
        )

        self.hooks.emit(HookName.BEFORE_ACTION, context, {})
        routing = self._build_routing(entities, low_pass, canonical)
        self.hooks.emit(HookName.AFTER_ACTION, context, {"routing": len(routing)})
        logger.debug("Routing decisions generated: %s", len(routing))

        report = EngineReport(
            processed_entities=len(entities),
            active_entities=len(active_entities),
            suppressed_entities=suppressed,
            skipped_entities=skipped,
            clusters=clusters,
            edges=edges,
            canonical_decisions=canonical,
            low_pass=list(low_pass.values()),
            routing=routing,
        )
        if self.storage and self.config.storage.persist_runs:
            self.storage.save_run(
                entities=entities,
                fingerprints=fingerprints,
                report=report,
                embedding_model=self.embedding_provider.model_id(),
            )
            logger.debug("Run persisted via storage backend")
        logger.info(
            "Scan completed: processed=%s clusters=%s total_time=%.2fs",
            len(entities),
            len(clusters),
            time.perf_counter() - scan_start,
        )
        return report

    def _build_routing(
        self,
        entities: list[SourceEntity],
        low_pass: dict[str, LowPassDecision],
        canonical: list,
    ) -> list[RoutingDecision]:
        labels_cfg = self.config.labels
        decision_map: dict[str, tuple[DecisionState, str | None]] = {}

        for cluster_decision in canonical:
            for member in cluster_decision.member_decisions:
                decision_map[member.entity_id] = (member.state, member.duplicate_of)

        routing: list[RoutingDecision] = []
        for entity in entities:
            labels: list[str] = []
            queue = None
            comment = None
            lp = low_pass[entity.id]

            if lp.state in {FilterState.SUPPRESS, FilterState.SKIP}:
                labels.append(labels_cfg.noise_suppressed)
                if self.config.action.queue_on_suppress:
                    labels.append(labels_cfg.quarantine)
                    queue = "quarantine"
            else:
                labels.append(labels_cfg.ready_human)
                state, duplicate_of = decision_map.get(entity.id, (DecisionState.RELATED, None))
                if state == DecisionState.CANONICAL:
                    labels.append(labels_cfg.canonical)
                    if self.config.action.add_comments:
                        comment = "Marked as canonical candidate for this similarity cluster."
                elif state == DecisionState.DUPLICATE:
                    labels.append(labels_cfg.duplicate)
                    if self.config.action.add_comments and duplicate_of:
                        comment = f"Potential duplicate of {duplicate_of}; routed for maintainer confirmation."
                else:
                    labels.append(labels_cfg.related)

            routing.append(RoutingDecision(entity_id=entity.id, labels=sorted(set(labels)), queue_key=queue, comment=comment))

        return routing

    @staticmethod
    def _storage_from_config(config: CarapaceConfig) -> StorageBackend | None:
        if not config.storage.persist_runs:
            return None
        if config.storage.backend == "sqlite":
            from carapace.storage import SQLiteStorage

            return SQLiteStorage(config.storage.sqlite_path)
        return None
