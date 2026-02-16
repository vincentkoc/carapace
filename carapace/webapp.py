"""Lightweight graph UI and API for triage visualization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from carapace.config import CarapaceConfig
from carapace.models import EngineReport, SourceEntity
from carapace.storage import SQLiteStorage


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: str
    cluster_type: str
    members: list[str]
    canonical: str | None
    canonical_pr: str | None
    canonical_issue: str | None
    priority: float
    issue_attention: float
    duplicate_pressure: float


def _association_trust(association: str | None) -> float:
    key = (association or "").upper()
    mapping = {
        "OWNER": 1.0,
        "MEMBER": 0.9,
        "COLLABORATOR": 0.8,
        "CONTRIBUTOR": 0.6,
        "FIRST_TIME_CONTRIBUTOR": 0.25,
        "FIRST_TIMER": 0.2,
        "NONE": 0.3,
    }
    return mapping.get(key, 0.4)


def _issue_attention(entity: SourceEntity) -> float:
    comments = int(entity.metadata.get("comment_count", 0) or 0)
    reactions = int(entity.metadata.get("reaction_total", 0) or 0)
    return (comments * 0.6) + (reactions * 0.4)


def _compute_cluster_summaries(
    report: EngineReport,
    entities: dict[str, SourceEntity],
) -> list[ClusterSummary]:
    canonical_by_cluster = {decision.cluster_id: decision for decision in report.canonical_decisions}
    summaries: list[ClusterSummary] = []

    for cluster in report.clusters:
        decision = canonical_by_cluster.get(cluster.id)
        canonical = decision.canonical_entity_id if decision else None
        canonical_pr = decision.canonical_pr_entity_id if decision else None
        canonical_issue = decision.canonical_issue_entity_id if decision else None

        issue_attention = 0.0
        for entity_id in cluster.members:
            entity = entities.get(entity_id)
            if entity and entity.kind.value == "issue":
                issue_attention += _issue_attention(entity)

        duplicate_pressure = max(0, len(cluster.members) - 1)
        cluster_size = len(cluster.members)
        priority = (issue_attention * 0.45) + (cluster_size * 0.35) + (duplicate_pressure * 0.20)

        summaries.append(
            ClusterSummary(
                cluster_id=cluster.id,
                cluster_type=cluster.cluster_type,
                members=cluster.members,
                canonical=canonical,
                canonical_pr=canonical_pr,
                canonical_issue=canonical_issue,
                priority=priority,
                issue_attention=issue_attention,
                duplicate_pressure=float(duplicate_pressure),
            )
        )

    summaries.sort(key=lambda item: (item.priority, len(item.members)), reverse=True)
    return summaries


def _node_size_from_priority(priority: float) -> float:
    return max(12.0, min(80.0, 12.0 + (priority * 1.5)))


def _build_graph_payload(
    *,
    repo: str,
    report: EngineReport,
    entities: dict[str, SourceEntity],
    summaries: dict[str, ClusterSummary],
    author_cache: dict[str, dict[str, float | int | str]],
    cluster_id: str | None,
    min_edge_score: float,
    max_clusters: int,
    include_authors: bool,
) -> dict[str, Any]:
    include_members: set[str] = set()
    include_clusters: set[str] = set()
    cluster_lookup = {cluster.id: cluster for cluster in report.clusters}
    summary_values = sorted(summaries.values(), key=lambda item: (item.priority, len(item.members)), reverse=True)
    default_cluster_ids = [item.cluster_id for item in summary_values if len(item.members) > 1][:max_clusters]

    if cluster_id:
        target = cluster_lookup.get(cluster_id)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Unknown cluster id: {cluster_id}")
        include_clusters.add(target.id)
        include_members.update(target.members)
        include_members.update(target.shadow_members)
    else:
        for cid in default_cluster_ids:
            include_clusters.add(cid)
            cluster = cluster_lookup[cid]
            include_members.update(cluster.members)
            include_members.update(cluster.shadow_members)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for cid in sorted(include_clusters):
        summary = summaries.get(cid)
        if summary is None:
            continue
        nodes.append(
            {
                "data": {
                    "id": cid,
                    "kind": "cluster",
                    "label": cid,
                    "cluster_type": summary.cluster_type,
                    "priority": round(summary.priority, 2),
                    "size": _node_size_from_priority(summary.priority),
                }
            }
        )

    author_nodes: set[str] = set()
    issue_index = {entity.number: entity.id for entity in entities.values() if entity.kind.value == "issue" and entity.number is not None}

    for entity_id in sorted(include_members):
        entity = entities.get(entity_id)
        if entity is None:
            continue

        trust_cache = author_cache.get(entity.author, {})
        merged_pr_count = int(trust_cache.get("merged_pr_count", 0) or 0)
        cached_trust = float(trust_cache.get("trust_score", 0.0) or 0.0)
        trust_score = max(_association_trust(entity.author_association), cached_trust)

        nodes.append(
            {
                "data": {
                    "id": entity.id,
                    "kind": entity.kind.value,
                    "label": entity.title,
                    "author": entity.author,
                    "state": entity.state,
                    "trust_score": round(trust_score, 3),
                    "merged_pr_count": merged_pr_count,
                    "comment_count": int(entity.metadata.get("comment_count", 0) or 0),
                    "reaction_total": int(entity.metadata.get("reaction_total", 0) or 0),
                    "size": 20 if entity.kind.value == "pr" else 18,
                }
            }
        )

        if include_authors:
            author_id = f"author:{entity.author}"
            if author_id not in author_nodes:
                author_nodes.add(author_id)
                nodes.append(
                    {
                        "data": {
                            "id": author_id,
                            "kind": "author",
                            "label": entity.author,
                            "trust_score": round(trust_score, 3),
                            "merged_pr_count": merged_pr_count,
                            "size": 16 + (trust_score * 10.0),
                        }
                    }
                )

            edges.append(
                {
                    "data": {
                        "id": f"authored_by:{entity.id}:{author_id}",
                        "source": entity.id,
                        "target": author_id,
                        "kind": "authored_by",
                        "weight": round(trust_score, 3),
                    }
                }
            )

        linked_issue_ids = set(entity.linked_issues) | set(entity.soft_linked_issues)
        for raw in linked_issue_ids:
            if not raw.isdigit():
                continue
            issue_num = int(raw)
            issue_id = issue_index.get(issue_num)
            if issue_id and issue_id in include_members and issue_id != entity.id:
                edges.append(
                    {
                        "data": {
                            "id": f"links:{entity.id}:{issue_id}:{raw}",
                            "source": entity.id,
                            "target": issue_id,
                            "kind": "references",
                            "weight": 1.0,
                        }
                    }
                )

    cluster_members = {cluster.id: set(cluster.members) for cluster in report.clusters if cluster.id in include_clusters}
    for cid, members in cluster_members.items():
        for entity_id in members:
            if entity_id not in include_members:
                continue
            edges.append(
                {
                    "data": {
                        "id": f"cluster:{cid}:{entity_id}",
                        "source": cid,
                        "target": entity_id,
                        "kind": "in_cluster",
                        "weight": 0.8,
                    }
                }
            )

    # Build lightweight lineage bridges between clusters to avoid isolated islands.
    cluster_authors: dict[str, set[str]] = {}
    cluster_issue_refs: dict[str, set[str]] = {}
    for cid in include_clusters:
        authors: set[str] = set()
        issue_refs: set[str] = set()
        for entity_id in cluster_members.get(cid, set()):
            entity = entities.get(entity_id)
            if entity is None:
                continue
            authors.add(entity.author)
            issue_refs.update(entity.linked_issues)
            issue_refs.update(entity.soft_linked_issues)
            if entity.kind.value == "issue" and entity.number is not None:
                issue_refs.add(str(entity.number))
        cluster_authors[cid] = authors
        cluster_issue_refs[cid] = issue_refs

    bridge_count = 0
    cluster_ids = sorted(include_clusters)
    for idx, left in enumerate(cluster_ids):
        for right in cluster_ids[idx + 1 :]:
            shared_authors = cluster_authors[left] & cluster_authors[right]
            shared_refs = cluster_issue_refs[left] & cluster_issue_refs[right]
            if not shared_authors and not shared_refs:
                continue
            bridge_weight = float(len(shared_authors) * 0.6 + len(shared_refs) * 1.0)
            edges.append(
                {
                    "data": {
                        "id": f"bridge:{left}:{right}",
                        "source": left,
                        "target": right,
                        "kind": "cluster_bridge",
                        "weight": round(bridge_weight, 3),
                    }
                }
            )
            bridge_count += 1
            if bridge_count >= 2000:
                break
        if bridge_count >= 2000:
            break

    for edge in report.edges:
        if edge.score < min_edge_score:
            continue
        if edge.entity_a not in include_members or edge.entity_b not in include_members:
            continue
        edges.append(
            {
                "data": {
                    "id": f"sim:{edge.entity_a}:{edge.entity_b}",
                    "source": edge.entity_a,
                    "target": edge.entity_b,
                    "kind": "similarity",
                    "tier": edge.tier.value,
                    "weight": round(edge.score, 4),
                }
            }
        )

    return {
        "repo": repo,
        "mode": "run",
        "cluster_id": cluster_id,
        "max_clusters": max_clusters,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "elements": {"nodes": nodes, "edges": edges},
    }


def _build_ingest_fallback_graph_payload(
    *,
    repo: str,
    entities: list[SourceEntity],
    author_cache: dict[str, dict[str, float | int | str]],
    max_entities: int,
    include_authors: bool,
) -> dict[str, Any]:
    selected = sorted(entities, key=lambda item: item.updated_at, reverse=True)[:max_entities]
    if not selected:
        return {
            "repo": repo,
            "mode": "ingest_fallback",
            "cluster_id": None,
            "node_count": 0,
            "edge_count": 0,
            "elements": {"nodes": [], "edges": []},
        }

    issue_index = {entity.number: entity.id for entity in selected if entity.kind.value == "issue" and entity.number is not None}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    author_nodes: set[str] = set()

    for entity in selected:
        trust_cache = author_cache.get(entity.author, {})
        merged_pr_count = int(trust_cache.get("merged_pr_count", 0) or 0)
        cached_trust = float(trust_cache.get("trust_score", 0.0) or 0.0)
        trust_score = max(_association_trust(entity.author_association), cached_trust)

        nodes.append(
            {
                "data": {
                    "id": entity.id,
                    "kind": entity.kind.value,
                    "label": entity.title,
                    "author": entity.author,
                    "state": entity.state,
                    "trust_score": round(trust_score, 3),
                    "merged_pr_count": merged_pr_count,
                    "comment_count": int(entity.metadata.get("comment_count", 0) or 0),
                    "reaction_total": int(entity.metadata.get("reaction_total", 0) or 0),
                    "size": 20 if entity.kind.value == "pr" else 18,
                }
            }
        )

        if include_authors:
            author_id = f"author:{entity.author}"
            if author_id not in author_nodes:
                author_nodes.add(author_id)
                nodes.append(
                    {
                        "data": {
                            "id": author_id,
                            "kind": "author",
                            "label": entity.author,
                            "trust_score": round(trust_score, 3),
                            "merged_pr_count": merged_pr_count,
                            "size": 16 + (trust_score * 10.0),
                        }
                    }
                )

            edges.append(
                {
                    "data": {
                        "id": f"authored_by:{entity.id}:{author_id}",
                        "source": entity.id,
                        "target": author_id,
                        "kind": "authored_by",
                        "weight": round(trust_score, 3),
                    }
                }
            )

        linked_issue_ids = set(entity.linked_issues) | set(entity.soft_linked_issues)
        for raw in linked_issue_ids:
            if not raw.isdigit():
                continue
            issue_num = int(raw)
            issue_id = issue_index.get(issue_num)
            if issue_id and issue_id != entity.id:
                edges.append(
                    {
                        "data": {
                            "id": f"links:{entity.id}:{issue_id}:{raw}",
                            "source": entity.id,
                            "target": issue_id,
                            "kind": "references",
                            "weight": 1.0,
                        }
                    }
                )

    return {
        "repo": repo,
        "mode": "ingest_fallback",
        "cluster_id": None,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "elements": {"nodes": nodes, "edges": edges},
    }


def create_app(config: CarapaceConfig) -> FastAPI:
    app = FastAPI(title="Carapace UI")
    storage = SQLiteStorage(config.storage.sqlite_path)
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    def _repo_or_default(repo: str | None) -> str:
        if repo:
            return repo
        repos = storage.list_ingested_repos()
        if not repos:
            raise HTTPException(status_code=404, detail="No ingested repositories found")
        return repos[0]

    def _load_context(repo: str) -> tuple[EngineReport, dict[str, SourceEntity], list[ClusterSummary], dict[str, dict[str, float | int | str]]]:
        report = storage.get_latest_run_report(repo)
        if report is None:
            raise HTTPException(status_code=404, detail=f"No run report found for repo={repo}")

        entity_ids: set[str] = set()
        for cluster in report.clusters:
            entity_ids.update(cluster.members)
            entity_ids.update(cluster.shadow_members)
        entities = storage.load_ingested_entities_by_ids(repo, sorted(entity_ids))
        summaries = _compute_cluster_summaries(report, entities)
        authors = sorted({entity.author for entity in entities.values()})
        author_cache = storage.get_author_metrics_cache(repo, authors)
        return report, entities, summaries, author_cache

    @app.get("/", response_class=HTMLResponse)
    def ui_index(request: Request, repo: str | None = None) -> HTMLResponse:
        selected_repo = _repo_or_default(repo)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "repo": selected_repo,
                "repos": storage.list_ingested_repos(),
            },
        )

    @app.get("/ui/fragments/clusters", response_class=HTMLResponse)
    def ui_cluster_fragment(
        request: Request,
        repo: str,
        kind: str = Query(default="all"),
        min_members: int = Query(default=2, ge=1),
        limit: int = Query(default=150, ge=1, le=1000),
    ) -> HTMLResponse:
        report, entities, summaries, _ = _load_context(repo)
        _ = report

        filtered = [summary for summary in summaries if len(summary.members) >= min_members and (kind == "all" or summary.cluster_type == kind)]
        filtered = filtered[:limit]

        cluster_entity_titles: dict[str, list[str]] = {}
        for summary in filtered:
            titles: list[str] = []
            for entity_id in summary.members:
                entity = entities.get(entity_id)
                if entity:
                    titles.append(f"{entity_id} — {entity.title}")
                else:
                    titles.append(entity_id)
            cluster_entity_titles[summary.cluster_id] = titles

        return templates.TemplateResponse(
            request,
            "clusters_fragment.html",
            {
                "repo": repo,
                "clusters": filtered,
                "cluster_entity_titles": cluster_entity_titles,
            },
        )

    @app.get("/api/repos/{repo:path}/graph", response_class=JSONResponse)
    def api_graph(
        repo: str,
        cluster_id: str | None = None,
        min_edge_score: float = Query(default=0.15, ge=0.0, le=1.0),
        fallback_max_entities: int = Query(default=600, ge=50, le=5000),
        max_clusters: int = Query(default=80, ge=10, le=2000),
        include_authors: bool = Query(default=False),
    ) -> JSONResponse:
        payload: dict[str, Any]
        try:
            report, entities, summaries, author_cache = _load_context(repo)
            summary_map = {summary.cluster_id: summary for summary in summaries}
            payload = _build_graph_payload(
                repo=repo,
                report=report,
                entities=entities,
                summaries=summary_map,
                author_cache=author_cache,
                cluster_id=cluster_id,
                min_edge_score=min_edge_score,
                max_clusters=max_clusters,
                include_authors=include_authors,
            )
            if payload["node_count"] == 0:
                ingest_entities = storage.load_ingested_entities(
                    repo,
                    include_closed=False,
                    include_drafts=False,
                    kind=None,
                )
                authors = sorted({entity.author for entity in ingest_entities})
                author_cache = storage.get_author_metrics_cache(repo, authors)
                payload = _build_ingest_fallback_graph_payload(
                    repo=repo,
                    entities=ingest_entities,
                    author_cache=author_cache,
                    max_entities=fallback_max_entities,
                    include_authors=include_authors,
                )
        except HTTPException:
            ingest_entities = storage.load_ingested_entities(
                repo,
                include_closed=False,
                include_drafts=False,
                kind=None,
            )
            if not ingest_entities:
                raise
            authors = sorted({entity.author for entity in ingest_entities})
            author_cache = storage.get_author_metrics_cache(repo, authors)
            payload = _build_ingest_fallback_graph_payload(
                repo=repo,
                entities=ingest_entities,
                author_cache=author_cache,
                max_entities=fallback_max_entities,
                include_authors=include_authors,
            )
        return JSONResponse(payload)

    @app.get("/api/repos/{repo:path}/clusters", response_class=JSONResponse)
    def api_clusters(
        repo: str,
        kind: str = Query(default="all"),
        min_members: int = Query(default=2, ge=1),
        limit: int = Query(default=200, ge=1, le=2000),
    ) -> JSONResponse:
        _report, _entities, summaries, _author_cache = _load_context(repo)
        rows: list[dict[str, Any]] = []
        for summary in summaries:
            if len(summary.members) < min_members:
                continue
            if kind != "all" and summary.cluster_type != kind:
                continue
            rows.append(
                {
                    "cluster_id": summary.cluster_id,
                    "cluster_type": summary.cluster_type,
                    "priority": round(summary.priority, 3),
                    "member_count": len(summary.members),
                    "canonical": summary.canonical,
                    "canonical_pr": summary.canonical_pr,
                    "canonical_issue": summary.canonical_issue,
                    "issue_attention": round(summary.issue_attention, 3),
                    "duplicate_pressure": round(summary.duplicate_pressure, 3),
                }
            )
            if len(rows) >= limit:
                break
        return JSONResponse({"repo": repo, "clusters": rows})

    @app.get("/api/repos/{repo:path}/authors", response_class=JSONResponse)
    def api_authors(repo: str, limit: int = Query(default=100, ge=1, le=1000)) -> JSONResponse:
        report, entities, _summaries, author_cache = _load_context(repo)
        _ = report
        by_author: dict[str, dict[str, Any]] = defaultdict(lambda: {"open_prs": 0, "open_issues": 0, "association": None, "trust_score": 0.0})
        for entity in entities.values():
            row = by_author[entity.author]
            if entity.kind.value == "pr":
                row["open_prs"] += 1
            elif entity.kind.value == "issue":
                row["open_issues"] += 1
            row["association"] = row["association"] or entity.author_association
            row["trust_score"] = max(float(row["trust_score"]), _association_trust(entity.author_association))

        rows: list[dict[str, Any]] = []
        for author, metrics in by_author.items():
            cache = author_cache.get(author, {})
            rows.append(
                {
                    "author": author,
                    "open_prs": metrics["open_prs"],
                    "open_issues": metrics["open_issues"],
                    "association": metrics["association"],
                    "merged_pr_count": int(cache.get("merged_pr_count", 0) or 0),
                    "trust_score": round(max(float(metrics["trust_score"]), float(cache.get("trust_score", 0.0) or 0.0)), 3),
                    "computed_at": cache.get("computed_at"),
                }
            )

        rows.sort(key=lambda item: (item["trust_score"], item["merged_pr_count"], item["open_prs"]), reverse=True)
        return JSONResponse({"repo": repo, "authors": rows[:limit]})

    return app
