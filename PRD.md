# Carapace Product Requirements Document (PRD)

## 1. Product Summary
Carapace is a contribution triage product that helps maintainers handle very high-volume PR and issue queues by:
- Detecting similar/duplicate submissions.
- Selecting canonical candidates inside duplicate clusters.
- Reducing noise with configurable low-pass filtering.
- Preserving workflow in native tools (GitHub first) while enabling future source systems.

## 2. Problem Statement
Large repositories experience PR and issue growth beyond human triage capacity. Maintainers lose time on overlapping submissions, repeated reviews, and low-signal noise. Existing review tools score single PR quality but do not orchestrate repository-level deduplication and canonicalization.

## 3. Goals

### Primary Goals
- G1: Reduce maintainer triage load by clustering similar PRs/issues.
- G2: Recommend a canonical PR per cluster with explainable scoring.
- G3: Lower queue noise through low-pass filters (labels, stale, low-signal patterns).
- G4: Keep architecture extensible across source systems (GitHub now, Jira later).

### Secondary Goals
- S1: Reuse external reviewer signals (CodeRabbit, Greptile) as inputs.
- S2: Support offline POC and live service with same core engine.
- S3: Provide a clean SDK/hook model for custom policy and integrations.

## 4. Non-Goals (Initial Release)
- Fully autonomous merge policy.
- Replacing human code review.
- Building a heavy standalone UI before proving GitHub-native workflows.

## 5. Users and Personas
- Maintainer
  - Needs fast queue reduction and clear canonical recommendations.
- Core reviewer
  - Needs conflict visibility and cluster-level context.
- Repo administrator
  - Needs policy controls, auditability, and safe automation.
- Platform engineer
  - Needs extensible SDK/connectors and reliable operations.

## 6. Jobs To Be Done
- "When new PRs arrive, show me duplicates and the best candidate so I review once."
- "When queue volume spikes, suppress low-signal noise without losing traceability."
- "When using existing AI review tools, aggregate their outputs instead of duplicating effort."
- "When we move beyond GitHub, keep core triage logic unchanged."

## 7. Success Metrics

### Product KPIs
- 30%+ reduction in PRs requiring first-pass human triage within 60 days.
- 70%+ maintainer agreement with canonical recommendation in sampled clusters.
- 50%+ reduction in median time-to-first-triage decision.
- <5% false duplicate label rate in reviewed samples.

### System KPIs
- P95 event-to-decision latency < 3s in live mode.
- Batch processing of 10k open entities < 30 minutes.
- >99% successful action application retries within SLA window.

## 8. User Experience Requirements

### MVP Experience (GitHub-native)
- PR/issue arrives.
- Carapace computes similarity and cluster membership.
- Carapace labels canonical and duplicates.
- Carapace posts explainable summary comment:
  - why grouped
  - why canonical chosen
  - what maintainer can override

### Noise Reduction Experience
- Items matching low-pass noise rules are labeled/suppressed.
- Maintainers operate via filtered saved views/boards.
- Suppressed items remain discoverable and reversible.

### Trust and Control
- Every decision has reason codes.
- Maintainers can override with comments/commands.
- Safe mode defaults to non-destructive actions.

## 9. Functional Requirements

### FR-P1 Similarity and Clustering
- Detect related PRs/issues using text + structural + lineage signals.
- Group results into clusters with stable ids.
- Recompute incrementally on updates.

### FR-P2 Canonical Selection
- Score and rank cluster members.
- Mark one canonical candidate and classify others.
- Expose tie state when confidence is low.

### FR-P3 Low-Pass Filtering
- Configurable hard skip, soft suppress, and priority boost rules.
- First-class rules for labels, stale age, actor type, and docs-only/noisy patterns.

### FR-P4 External Signal Reuse
- Parse and normalize reviewer scores/findings from CodeRabbit/Greptile.
- Use these signals in canonical ranking and tie-breaking.

### FR-P5 Extensibility
- Modular connector SDK and lifecycle hooks.
- Provider-agnostic domain schema for future Jira/GitLab adoption.

### FR-P6 Operating Modes
- Offline replay mode for POC and testing.
- Live webhook mode for production triage.

## 10. Non-Functional Requirements
- Scalability to high-volume repositories and orgs.
- Idempotent processing and deterministic replay.
- Auditable decision records.
- Minimal external dependency footprint for local deployments.
- Python services must use Pydantic models for typed validation at system boundaries.

## 11. Embedding Product Strategy

### Requirements
- Fast, local-first embedding in default path.
- Accuracy strong enough for semantic duplicate retrieval.
- API fallback for teams preferring hosted model operations.

### Product Decision
- Ship with pluggable embedding provider abstraction.
- Default bundles:
  - `fast-local`
  - `balanced-local`
  - `high-accuracy-local` (hardware-dependent)
- Optional OpenAI-compatible embedding endpoint integration.

### Acceptance
- Admin can switch embedding profile by config only.
- Similarity quality remains above minimum precision target in validation set.

## 12. Configuration and Policy Requirements
- Repo-level configuration file at repository root: `.carapace.yaml`.
- `.carapace.yaml` defines:
  - low-pass rules
  - label taxonomy
  - thresholds for clustering/canonical confidence
  - action safety mode
- Organization-level inheritance with repo overrides.
- Precedence order:
  - runtime override
  - repo `.carapace.yaml`
  - org default
  - system default

## 13. Roadmap and Milestones

### Phase 0: Foundation (Week 1-2)
- Domain schema, connector interfaces, offline ingest.
- Basic low-pass filtering and deterministic reports.

### Phase 1: Similarity + Canonical MVP (Week 3-5)
- Candidate retrieval + pair scoring + clustering.
- Canonical ranking and explainability output.

### Phase 2: Live Assistive (Week 6-8)
- GitHub webhook integration.
- Label/comment actioning and filtered queue flows.

### Phase 3: Extensibility + Optimization (Week 9+)
- Hook marketplace patterns.
- Alternate connectors (Jira pilot).
- Learned ranking/scoring options.

## 14. Risks and Mitigations
- Risk: false duplicate grouping harms trust.
  - Mitigation: conservative thresholds, explainability, easy overrides.
- Risk: over-suppression hides important submissions.
  - Mitigation: soft suppress by default, audit queues, periodic review.
- Risk: embedding latency/cost variance.
  - Mitigation: local-first model profiles, caching, API fallback.
- Risk: connector lock-in.
  - Mitigation: strict provider-neutral schema and contract tests.

## 15. Dependencies
- GitHub App permissions for PR/issue/check metadata.
- Optional external review tool output availability.
- Runtime infra for worker + storage + vector/signature index.

## 16. Release Criteria (MVP)
- End-to-end offline run produces clusters + canonical recommendations.
- Live mode labels canonical/duplicates reliably on new PRs.
- Low-pass rules reduce queue size without unacceptable false suppressions.
- Maintainer pilot validates measurable triage-time reduction.

## 17. Semantic Commit Plan
Implementation should follow semantic commits throughout development:
- `feat:` new capabilities
- `perf:` optimization changes
- `fix:` correctness changes
- `docs:` documentation updates
- `test:` validation/coverage updates
- `chore:` infra and maintenance

Cleanup policy:
- Keep semantically meaningful commits during buildout.
- Optional final cleanup before milestone tag (squash only noisy micro-commits).

## 18. Open Product Questions
- Which default low-pass presets should be enabled for first-run safety?
- What maintainer override UX is preferred: comment commands, labels, or both?
- Should canonical recommendation confidence be hidden behind explicit opt-in initially?
- Which connector is highest priority after GitHub: Jira or GitLab?
