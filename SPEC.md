# Carapace Technical Design Specification (SPEC)

## 1. Purpose
Carapace is a triage orchestration engine for large contribution queues. It detects similarity across pull requests and issues, selects canonical candidates, and routes noisy submissions out of maintainer focus views.

This specification defines a modular, extensible architecture that:
- Works as a live service and offline POC.
- Uses fast local-first embedding and similarity approaches.
- Supports low-pass filtering (labels, stale age, actor behavior, etc.).
- Can swap source systems (GitHub now, Jira or others later) without rewriting core logic.

## 2. Scope

### In Scope
- Similarity detection for PRs and issues.
- Cluster formation and canonical candidate selection.
- Low-pass noise filtering before expensive scoring.
- Pluggable embeddings (local and API-backed).
- Extensible connector and hook system.
- Rule-driven routing actions (label, comment, quarantine queue).
- Offline replay mode and live webhook mode.

### Out of Scope (Initial)
- Full autonomous merge/close decisions by default.
- Full code review replacement (reuse existing review signals).
- UI-heavy platform in v1 (GitHub-native views first).

## 3. System Overview

### 3.1 High-Level Components
1. Ingestion Layer
- Receives events from connectors (GitHub webhooks, batch importers, replay logs).

2. Normalization Layer
- Maps provider payloads into a canonical Carapace domain schema.

3. Low-Pass Filter Layer
- Early noise suppression to reduce compute and maintainer cognitive load.

4. Fingerprint + Embedding Layer
- Builds text, structural, and lineage fingerprints.
- Produces embeddings via pluggable provider abstraction.

5. Similarity + Clustering Layer
- Candidate retrieval + pair scoring + cluster assignment.

6. Canonical Selection Layer
- Scores candidates in each cluster and chooses canonical PR.

7. Action + Policy Layer
- Applies labels/comments/routes through connector sinks.

8. Storage Layer
- Relational store for entities/events/decisions.
- Vector and signature index for retrieval.

9. Hook/SDK Extension Layer
- Lifecycle hooks and plugin APIs for custom logic.

### 3.2 Deployment Modes
- Live mode: webhook-driven, near real-time updates.
- Offline mode: snapshot/event replay; identical core pipeline.

## 4. Architecture Requirements

### Functional Requirements
- FR-1: Ingest PR/issue/update events from GitHub.
- FR-2: Support connector abstraction for alternate systems (Jira, GitLab, Bitbucket).
- FR-3: Apply configurable low-pass rules pre-fingerprint.
- FR-4: Generate hybrid fingerprints (text, structural, lineage, quality signals).
- FR-5: Compute similarity with scalable candidate retrieval.
- FR-6: Cluster related submissions incrementally.
- FR-7: Rank and mark canonical entry per cluster.
- FR-8: Emit routing decisions (labels/comments/queues) with confidence and reason traces.
- FR-9: Expose SDK hooks for custom features and decision overrides.
- FR-10: Operate in offline POC mode with deterministic replay.
- FR-11: Load repository-level configuration from `.carapace.yaml` at repository root.

### Non-Functional Requirements
- NFR-1: Handle 10k+ open entities per repo/org.
- NFR-2: P95 online event processing under 3s excluding external API delays.
- NFR-3: Support horizontal worker scaling.
- NFR-4: Full auditability of decisions and feature contributions.
- NFR-5: Idempotent event processing and exactly-once effective actioning.
- NFR-6: Python services must use Pydantic models for boundary validation and typed contracts.

## 5. Domain Model

### Core Entities
- `SourceEntity`: PR, issue, ticket, task.
- `Fingerprint`: normalized feature bundle for an entity.
- `EmbeddingVector`: dense semantic vector(s) by provider/model.
- `SimilarityEdge`: pairwise score with feature breakdown.
- `Cluster`: group of related entities.
- `CanonicalDecision`: winning entity + ranked alternatives.
- `RoutingDecision`: actions to apply (labels/comments/quarantine).
- `PolicyRule`: low-pass/decision rule with version.

### Key Tables (Relational)
- `entities`
- `entity_events`
- `fingerprints`
- `embeddings`
- `similarity_edges`
- `clusters`
- `canonical_decisions`
- `routing_decisions`
- `policy_rules`
- `connector_state` (cursor/checkpoint state per provider)

## 6. Connector and SDK Design

### 6.1 Connector Interfaces
Define provider-specific connectors behind stable contracts:

```text
SourceConnector
  - subscribe_events()
  - list_open_entities()
  - get_entity(id)
  - get_diff_or_change_set(id)
  - get_reviews_and_checks(id)

SinkConnector
  - apply_labels(id, labels[])
  - post_comment(id, body)
  - set_status(id, state, context)
  - route_to_queue(id, queue_key)
```

### 6.2 Hook Lifecycle
Hooks are invoked with immutable context and mutable decision envelope.

```text
before_normalize
after_normalize
before_low_pass
after_low_pass
before_fingerprint
after_fingerprint
before_similarity
after_similarity
before_canonical
after_canonical
before_action
after_action
on_error
```

### 6.3 SDK Principles
- Provider-neutral schema.
- Versioned hook contracts.
- Deterministic replay support.
- Safe extension boundaries (timeouts + fallback behavior per hook).
- Language-agnostic API transport (HTTP/JSON now, gRPC optional).

## 7. Low-Pass Filter Design

### 7.1 Goals
- Drop or down-prioritize predictable noise early.
- Reduce pairwise comparison cost.
- Preserve recoverability and transparency.

### 7.2 Rule Types
1. Hard skip
- Ignore `stale`, `invalid`, `wontfix`, archived targets, bot-only formatting PRs.

2. Soft suppress
- De-prioritize low-signal entities into quarantine queues.

3. Priority boost
- Elevate labels like `security`, `regression`, `release-blocker`.

### 7.3 Rule Inputs
- Labels allowlist/denylist.
- Age windows (`updated_at`, stale duration).
- Actor type/trust tier (maintainer, external, automation).
- CI status stability.
- File/path filters (vendor/docs-only/config-only).
- Historical behavior (repeat low-signal patterns).

### 7.4 Output
- `filter_state`: `pass`, `suppress`, `skip`.
- `filter_reason_codes`: array of deterministic reason ids.
- `priority_weight`: numeric multiplier for downstream ranking.

### 7.5 Example Policy Snippet
```yaml
low_pass:
  hard_skip_labels: ["invalid", "wontfix", "duplicate", "stale"]
  soft_suppress_labels: ["question", "discussion"]
  suppress_if:
    - condition: "is_docs_only && actor_is_new && ci_state == 'none'"
      reason: "LOW_SIGNAL_DOCS_ONLY"
  boost_if:
    - condition: "has_label('security') || has_label('regression')"
      weight: 1.5
```

### 7.6 Configuration Discovery and Precedence
- Primary repo config path: `.carapace.yaml` at repository root.
- Optional org/global defaults may be provided by service config.
- Precedence order: entity/runtime override > repo `.carapace.yaml` > org defaults > system defaults.
- Missing repo config must not fail processing; system defaults apply.
- Config schema validation must run at load time and produce actionable errors.

## 8. Fingerprinting and Feature Engineering

### 8.1 Text Features
- Title/body normalized tokens.
- Linked issue ids.
- Extracted intent phrases from templates.
- External reviewer summaries (CodeRabbit/Greptile) when present.

### 8.2 Structural Features
- File path set + module buckets.
- Hunk signatures (path + context + normalized token hashes).
- Churn metrics: files changed, additions, deletions.

### 8.3 Lineage Features
- Commit SHAs.
- Patch-id set (when available from cloned refs).
- Base branch/head branch ancestry.

### 8.4 Quality Signals
- CI status history.
- Review approvals/comments.
- External reviewer score/risk/test-gap.

## 9. Embedding Strategy

### 9.1 Requirements
- Local-first for speed/cost/privacy.
- High retrieval quality for intent-level similarity.
- API fallback for quality/capacity bursts.

### 9.2 Provider Abstraction
```text
EmbeddingProvider
  - embed_texts(texts[], mode) -> vectors[]
  - model_id()
  - dimensions()
  - max_batch_size()
  - health_check()
```

### 9.3 Recommended Local Models
Primary (balanced accuracy/speed):
- `BAAI/bge-m3` served via TEI/Infinity.

Fast path:
- `nomic-embed-text-v1.5` or equivalent compact model.

High-accuracy local option:
- `jina-embeddings-v3` class model where hardware allows.

### 9.4 API Option
- OpenAI-compatible embedding endpoint contract:
  - `POST /v1/embeddings`
  - request includes `model`, `input`
  - response normalized into provider abstraction.

### 9.5 Multi-Vector Strategy (Optional)
- Store two vectors per entity:
  - `semantic_text_vector`
  - `review_summary_vector`
- Use weighted late fusion in pair scoring.

### 9.6 Operational Tuning
- Batch size auto-tuning by latency target.
- Quantization support (`int8`/`fp16`) for local serving.
- Model cache warm-up on startup.

## 10. Similarity Engine

### 10.1 Candidate Retrieval (Stage 1)
- Inverted indices on module buckets, linked issue ids, label classes.
- MinHash LSH on diff shingles.
- SimHash on normalized text.

Returns top-K candidates per entity for Stage 2 scoring.

### 10.2 Pair Scoring (Stage 2)
Score uses weighted ensemble:
- Lineage score.
- Structure score.
- Semantic score.
- Shape penalty.

Reference formula:
```text
S = 0.45*Lineage + 0.40*Structure + 0.15*Semantic - 0.10*SizePenalty
```

### 10.3 Edge Gating
- Strong edge if lineage overlap exceeds threshold.
- Medium edge if structure overlap + semantic alignment passes joint threshold.
- No edge otherwise.

### 10.4 Clustering
- Incremental union-find with strong/weak tiers.
- Weak edges require shared strong neighbor to avoid chain bridging.

## 11. Canonical Selection Engine

### 11.1 Scoring Features
- Cluster centrality (mean similarity).
- Coverage of cluster touched surface.
- CI health.
- External reviewer score.
- Approvals/review traction.
- Size penalty.
- Filter priority weight from low-pass layer.

### 11.2 Canonical Formula (Initial)
```text
CanonScore =
  5.0*Coverage +
  4.0*Centrality +
  3.0*CI +
  2.0*ReviewerScore +
  1.5*Approvals +
  1.0*PriorityWeight -
  1.0*SizePenalty
```

### 11.3 Decision States
- `canonical`
- `duplicate_of:<id>`
- `related_non_duplicate`
- `needs_human_tie_break`

## 12. Actioning and Routing

### Label Taxonomy (Baseline)
- `triage/canonical`
- `triage/duplicate`
- `triage/related`
- `triage/quarantine`
- `triage/noise-suppressed`
- `triage/ready-human`

### Action Policies
- Default safe mode: label + comment only.
- Optional strict mode: hide/quarantine routing.
- Auto-close disabled by default; enable only with explicit policy.

## 13. APIs

### Internal Service APIs
- `POST /events/ingest`
- `POST /entities/backfill`
- `GET /config/effective/{repo}`
- `GET /clusters/{id}`
- `GET /entities/{id}/similar`
- `POST /decisions/recompute`
- `GET /health`

### SDK/Hook APIs
- `register_hook(name, callback, timeout_ms)`
- `register_feature_extractor(name, extractor)`
- `register_connector(source|sink, impl)`

## 14. Storage and Indexing
- Relational DB (PostgreSQL preferred) for state and audit.
- Vector index (`pgvector`, Qdrant, or FAISS-backed sidecar).
- Signature stores for MinHash/SimHash keys.
- Blob storage for raw event snapshots in replay mode.

## 15. Reliability, Security, and Compliance
- Idempotency keys per event delivery.
- Replay-safe action deduplication.
- Signed webhook verification.
- Principle-of-least-privilege connector scopes.
- PII-safe logs and retention controls.
- Full decision traceability for maintainer audits.

## 16. Performance Targets
- 10k open entities processed in < 30 min batch refresh.
- Incremental event update P95 < 3s.
- Candidate retrieval P95 < 150ms for top-K query.
- Pair scoring throughput >= 2k pairs/sec per worker (baseline target).

## 17. Observability
- Metrics:
  - ingest lag
  - filter drop/suppress rates
  - candidate retrieval recall proxy
  - cluster churn
  - canonical flip rate
  - action failure rate
- Structured logs with correlation ids.
- Decision explanation payloads persisted.

## 18. Rollout Plan
1. Phase A: Offline POC
- Snapshot ingest, clustering, canonical report output.

2. Phase B: Live Read-Only
- Webhooks, decisions visible, no action writes.

3. Phase C: Live Assistive
- Labels/comments + quarantine routing.

4. Phase D: Optimization
- Learned pair scorer, active-learning thresholds, connector expansion.

## 19. Semantic Commit Strategy
- Use semantic commit prefixes during implementation:
  - `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `chore:`
- Keep commits scoped to single subsystem changes.
- Final cleanup pass:
  - squash noisy interim commits only at release boundary if needed.

## 20. Open Technical Questions
- Which local embedding model meets target latency on available hardware?
- Should canonical formulas remain deterministic or adopt learned ranker in v2?
- Which vector backend is preferred for first production deployment?
- How aggressive should low-pass suppression be by default for new repos?

## 21. Implementation Standards
- Primary implementation language is Python.
- Use Pydantic `BaseModel` for all external payloads, connector DTOs, event envelopes, config schemas, and decision outputs.
- Enable strict validation defaults to catch schema drift early.
- Keep business logic separate from transport models for testability.
- Use typed model fixtures in tests to avoid unvalidated dict-based test inputs.
