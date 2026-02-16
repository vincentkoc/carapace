"""Route registration and app factory for the Carapace web app."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from carapace.config import CarapaceConfig, load_effective_config
from carapace.embeddings.local_hash import LocalHashEmbeddingProvider
from carapace.models import EngineReport, SourceEntity
from carapace.storage import SQLiteStorage
from carapace.webapp_viewmodels import (
    ClusterSummary,
    _association_trust,
    _build_cluster_detail_payload,
    _build_cluster_map_from_summaries,
    _build_cluster_map_payload,
    _build_embedding_atlas_payload,
    _build_graph_payload,
    _build_ingest_cluster_detail_payload,
    _build_ingest_fallback_graph_payload,
    _build_ingest_linkage_summaries,
    _canonical_title,
    _compute_cluster_summaries,
    _filter_cluster_summaries,
)


def _cluster_maps_from_summaries(summaries: list[ClusterSummary]) -> tuple[dict[str, str], set[str], dict[str, int]]:
    cluster_by_entity: dict[str, str] = {}
    canonical_entities: set[str] = set()
    cluster_sizes: dict[str, int] = {}
    for summary in summaries:
        cluster_sizes[summary.cluster_id] = len(summary.members)
        if summary.canonical:
            canonical_entities.add(summary.canonical)
        for entity_id in summary.members:
            cluster_by_entity[entity_id] = summary.cluster_id
    return cluster_by_entity, canonical_entities, cluster_sizes

def create_app(config: CarapaceConfig) -> FastAPI:
    app = FastAPI(title="Carapace UI")
    storage = SQLiteStorage(config.storage.sqlite_path)
    fallback_embedder = LocalHashEmbeddingProvider(
        dims=max(128, int(config.embedding.dimensions)),
        model=f"atlas-{config.embedding.model}",
    )
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    run_context_cache: dict[str, tuple[str, tuple[EngineReport, dict[str, SourceEntity], list[ClusterSummary], dict[str, dict[str, float | int | str]]]]] = {}
    ingest_context_cache: dict[str, tuple[str, tuple[dict[str, SourceEntity], list[ClusterSummary], dict[str, dict[str, float | int | str]]]]] = {}
    response_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    response_cache_fifo: list[tuple[str, str, str]] = []
    max_response_cache_entries = 64
    report_cache: dict[str, tuple[float, EngineReport]] = {}
    report_cache_ttl_seconds = 5.0

    def _repo_or_default(repo: str | None) -> str:
        if repo:
            return repo
        repos = storage.list_ingested_repos()
        if not repos:
            raise HTTPException(status_code=404, detail="No ingested repositories found")
        return repos[0]

    def _get_cached_payload(repo: str, cache_key: str, signature: str) -> dict[str, Any] | None:
        mem_key = (repo, cache_key, signature)
        cached_mem = response_cache.get(mem_key)
        if cached_mem is not None:
            return cached_mem
        cached_db = storage.load_json_cache(repo, cache_key, source_signature=signature)
        if cached_db is not None:
            response_cache[mem_key] = cached_db
            response_cache_fifo.append(mem_key)
            if len(response_cache_fifo) > max_response_cache_entries:
                evicted = response_cache_fifo.pop(0)
                response_cache.pop(evicted, None)
        return cached_db

    def _store_cached_payload(repo: str, cache_key: str, signature: str, payload: dict[str, Any]) -> None:
        mem_key = (repo, cache_key, signature)
        response_cache[mem_key] = payload
        response_cache_fifo.append(mem_key)
        if len(response_cache_fifo) > max_response_cache_entries:
            evicted = response_cache_fifo.pop(0)
            response_cache.pop(evicted, None)
        storage.upsert_json_cache(repo, cache_key, source_signature=signature, payload=payload)

    def _run_signature(report: EngineReport) -> str:
        return "|".join(
            [
                report.generated_at.isoformat(),
                f"clusters:{len(report.clusters)}",
                f"edges:{len(report.edges)}",
                f"processed:{report.processed_entities}",
            ]
        )

    def _load_report(repo: str) -> EngineReport:
        now = time.monotonic()
        cached = report_cache.get(repo)
        if cached and (now - cached[0]) <= report_cache_ttl_seconds:
            return cached[1]
        report = storage.get_latest_run_report(repo)
        if report is None:
            raise HTTPException(status_code=404, detail=f"No run report found for repo={repo}")
        report_cache[repo] = (now, report)
        return report

    def _load_context(repo: str, report: EngineReport | None = None) -> tuple[EngineReport, dict[str, SourceEntity], list[ClusterSummary], dict[str, dict[str, float | int | str]]]:
        run_report = report or _load_report(repo)
        signature = _run_signature(run_report)
        cached = run_context_cache.get(repo)
        if cached and cached[0] == signature:
            return cached[1]

        entity_ids: set[str] = set()
        for cluster in run_report.clusters:
            entity_ids.update(cluster.members)
            entity_ids.update(cluster.shadow_members)
        entities = storage.load_ingested_entities_by_ids(repo, sorted(entity_ids))
        summaries = _compute_cluster_summaries(run_report, entities)
        authors = sorted({entity.author for entity in entities.values()})
        author_cache = storage.get_author_metrics_cache(repo, authors)
        context = (run_report, entities, summaries, author_cache)
        run_context_cache[repo] = (signature, context)
        return context

    def _ingest_signature(repo: str) -> str:
        summary = storage.ingest_audit_summary(repo)
        return "|".join(
            [
                f"total:{summary['total']}",
                f"prs:{summary['by_kind'].get('pr', 0)}",
                f"issues:{summary['by_kind'].get('issue', 0)}",
                f"drafts:{summary['draft_prs']}",
                f"enriched:{summary['enriched_prs']}",
            ]
        )

    def _load_ingest_context(repo: str) -> tuple[dict[str, SourceEntity], list[ClusterSummary], dict[str, dict[str, float | int | str]]]:
        signature = _ingest_signature(repo)
        cached = ingest_context_cache.get(repo)
        if cached and cached[0] == signature:
            return cached[1]

        ingest_entities = storage.load_ingested_entities(
            repo,
            include_closed=False,
            include_drafts=False,
            kind=None,
        )
        entities = {entity.id: entity for entity in ingest_entities}
        summaries = _build_ingest_linkage_summaries(repo, ingest_entities)
        authors = sorted({entity.author for entity in ingest_entities})
        author_cache = storage.get_author_metrics_cache(repo, authors)
        context = (entities, summaries, author_cache)
        ingest_context_cache[repo] = (signature, context)
        return context

    def _cluster_maps_from_summaries(summaries: list[ClusterSummary]) -> tuple[dict[str, str], set[str], dict[str, int]]:
        cluster_by_entity: dict[str, str] = {}
        canonical_entities: set[str] = set()
        cluster_sizes: dict[str, int] = {}
        for summary in summaries:
            cluster_sizes[summary.cluster_id] = len(summary.members)
            if summary.canonical:
                canonical_entities.add(summary.canonical)
            for entity_id in summary.members:
                cluster_by_entity[entity_id] = summary.cluster_id
        return cluster_by_entity, canonical_entities, cluster_sizes

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
        try:
            _report, entities, summaries, _ = _load_context(repo)
        except HTTPException:
            entities, summaries, _ = _load_ingest_context(repo)

        filtered = _filter_cluster_summaries(
            summaries,
            kind=kind,
            min_members=min_members,
            max_clusters=limit,
        )
        if not filtered:
            ingest_entities, ingest_summaries, _ = _load_ingest_context(repo)
            entities = ingest_entities
            filtered = _filter_cluster_summaries(
                ingest_summaries,
                kind=kind,
                min_members=min_members,
                max_clusters=limit,
            )

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

    @app.get("/api/v1/repos/{repo:path}/graph", response_class=JSONResponse)
    @app.get("/api/repos/{repo:path}/graph", response_class=JSONResponse)
    def api_graph(
        repo: str,
        cluster_id: str | None = None,
        min_edge_score: float = Query(default=0.15, ge=0.0, le=1.0),
        fallback_max_entities: int = Query(default=200, ge=50, le=5000),
        max_clusters: int = Query(default=30, ge=10, le=2000),
        include_authors: bool = Query(default=False),
        max_similarity_edges: int = Query(default=2500, ge=100, le=50000),
    ) -> JSONResponse:
        payload: dict[str, Any]
        try:
            report = _load_report(repo)
            run_sig = _run_signature(report)
            graph_sig = "|".join(
                [
                    run_sig,
                    f"cluster:{cluster_id or 'all'}",
                    f"min_edge:{min_edge_score:.4f}",
                    f"fallback_max:{fallback_max_entities}",
                    f"max_clusters:{max_clusters}",
                    f"authors:{1 if include_authors else 0}",
                    f"max_sim_edges:{max_similarity_edges}",
                ]
            )
            cached = _get_cached_payload(repo, "graph_v2", graph_sig)
            if cached is not None:
                return JSONResponse(cached)

            report, entities, summaries, author_cache = _load_context(repo, report=report)
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
                max_similarity_edges=max_similarity_edges,
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
            _store_cached_payload(repo, "graph_v2", graph_sig, payload)
        except HTTPException:
            ingest_sig = _ingest_signature(repo)
            graph_sig = "|".join(
                [
                    ingest_sig,
                    f"cluster:{cluster_id or 'all'}",
                    f"min_edge:{min_edge_score:.4f}",
                    f"fallback_max:{fallback_max_entities}",
                    f"max_clusters:{max_clusters}",
                    f"authors:{1 if include_authors else 0}",
                    f"max_sim_edges:{max_similarity_edges}",
                ]
            )
            cached = _get_cached_payload(repo, "graph_ingest_v2", graph_sig)
            if cached is not None:
                return JSONResponse(cached)
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
            _store_cached_payload(repo, "graph_ingest_v2", graph_sig, payload)
        return JSONResponse(payload)

    @app.get("/api/v1/repos/{repo:path}/graph/cluster-map", response_class=JSONResponse)
    @app.get("/api/repos/{repo:path}/graph/cluster-map", response_class=JSONResponse)
    def api_cluster_map(
        repo: str,
        kind: str = Query(default="all"),
        min_members: int = Query(default=2, ge=1),
        max_clusters: int = Query(default=60, ge=10, le=2000),
        max_bridges: int = Query(default=400, ge=0, le=5000),
    ) -> JSONResponse:
        try:
            report = _load_report(repo)
            run_sig = _run_signature(report)
            payload_sig = "|".join(
                [
                    run_sig,
                    f"kind:{kind}",
                    f"min_members:{min_members}",
                    f"max_clusters:{max_clusters}",
                    f"max_bridges:{max_bridges}",
                ]
            )
            cached = _get_cached_payload(repo, "cluster_map_v2", payload_sig)
            if cached is not None:
                return JSONResponse(cached)

            report, entities, summaries, _ = _load_context(repo, report=report)
            payload = _build_cluster_map_payload(
                repo=repo,
                report=report,
                entities=entities,
                summaries=summaries,
                kind=kind,
                min_members=min_members,
                max_clusters=max_clusters,
                max_bridges=max_bridges,
            )
            if payload["node_count"] == 0:
                entities, ingest_summaries, _ = _load_ingest_context(repo)
                payload = _build_cluster_map_from_summaries(
                    repo=repo,
                    summaries=ingest_summaries,
                    entities=entities,
                    kind=kind,
                    min_members=min_members,
                    max_clusters=max_clusters,
                    max_bridges=max_bridges,
                    mode="cluster_map_ingest",
                )
            _store_cached_payload(repo, "cluster_map_v2", payload_sig, payload)
        except HTTPException:
            ingest_sig = _ingest_signature(repo)
            payload_sig = "|".join(
                [
                    ingest_sig,
                    f"kind:{kind}",
                    f"min_members:{min_members}",
                    f"max_clusters:{max_clusters}",
                    f"max_bridges:{max_bridges}",
                ]
            )
            cached = _get_cached_payload(repo, "cluster_map_ingest_v2", payload_sig)
            if cached is not None:
                return JSONResponse(cached)
            entities, ingest_summaries, _ = _load_ingest_context(repo)
            payload = _build_cluster_map_from_summaries(
                repo=repo,
                summaries=ingest_summaries,
                entities=entities,
                kind=kind,
                min_members=min_members,
                max_clusters=max_clusters,
                max_bridges=max_bridges,
                mode="cluster_map_ingest",
            )
            _store_cached_payload(repo, "cluster_map_ingest_v2", payload_sig, payload)
        return JSONResponse(payload)

    @app.get("/api/v1/repos/{repo:path}/graph/atlas", response_class=JSONResponse)
    @app.get("/api/repos/{repo:path}/graph/atlas", response_class=JSONResponse)
    def api_embedding_atlas(
        repo: str,
        kind: str = Query(default="all"),
        include_edges: bool = Query(default=True),
        edge_k: int = Query(default=6, ge=0, le=24),
        edge_max_distance: float = Query(default=0.16, ge=0.01, le=2.0),
        min_signal: float = Query(default=0.0, ge=0.0, le=10_000.0),
        max_nodes: int = Query(default=8000, ge=100, le=50_000),
    ) -> JSONResponse:
        selected_kind = None if kind == "all" else kind
        entities = storage.load_ingested_entities(
            repo,
            include_closed=False,
            include_drafts=False,
            kind=selected_kind,
        )
        if not entities:
            return JSONResponse(
                {
                    "repo": repo,
                    "mode": "embedding_atlas",
                    "node_count": 0,
                    "edge_count": 0,
                    "elements": {"nodes": [], "edges": []},
                }
            )

        try:
            _report, _run_entities, summaries, _ = _load_context(repo)
        except HTTPException:
            _ingest_entities, summaries, _ = _load_ingest_context(repo)

        cluster_by_entity, canonical_entities, cluster_sizes = _cluster_maps_from_summaries(summaries)

        entity_ids = [entity.id for entity in entities]
        run_id, vectors = storage.load_latest_run_embeddings(repo, entity_ids=entity_ids)

        missing_entities = [entity for entity in entities if entity.id not in vectors or not vectors.get(entity.id)]
        if missing_entities:
            fallback_inputs = [f"{entity.title}\n{entity.body}" for entity in missing_entities]
            fallback_vectors = fallback_embedder.embed_texts(fallback_inputs)
            for entity, vec in zip(missing_entities, fallback_vectors):
                vectors[entity.id] = vec

        max_updated_at = max(entity.updated_at.isoformat() for entity in entities)
        signature = f"run:{run_id or 0}|count:{len(entities)}|updated:{max_updated_at}|kind:{kind}|edges:{1 if include_edges else 0}|k:{edge_k}|dist:{edge_max_distance:.4f}|min_signal:{min_signal:.4f}|max_nodes:{max_nodes}"
        cache_key = "graph_atlas_v1"
        cached = _get_cached_payload(repo, cache_key, signature)
        if cached is not None:
            return JSONResponse(cached)

        payload = _build_embedding_atlas_payload(
            repo=repo,
            entities=entities,
            vectors=vectors,
            cluster_by_entity=cluster_by_entity,
            canonical_entities=canonical_entities,
            cluster_sizes=cluster_sizes,
            include_edges=include_edges,
            edge_k=edge_k,
            edge_max_distance=edge_max_distance,
            min_signal=min_signal,
            max_nodes=max_nodes,
        )
        _store_cached_payload(repo, cache_key, signature, payload)
        return JSONResponse(payload)

    @app.get("/api/v1/repos/{repo:path}/clusters", response_class=JSONResponse)
    @app.get("/api/repos/{repo:path}/clusters", response_class=JSONResponse)
    def api_clusters(
        repo: str,
        kind: str = Query(default="all"),
        min_members: int = Query(default=2, ge=1),
        limit: int = Query(default=200, ge=1, le=2000),
    ) -> JSONResponse:
        cache_sig: str | None = None
        cache_key: str = "clusters_v2"
        try:
            report = _load_report(repo)
            cache_sig = "|".join(
                [
                    _run_signature(report),
                    f"kind:{kind}",
                    f"min_members:{min_members}",
                    f"limit:{limit}",
                ]
            )
            cached = _get_cached_payload(repo, "clusters_v2", cache_sig)
            if cached is not None:
                return JSONResponse(cached)
        except HTTPException:
            cache_key = "clusters_ingest_v2"
            ingest_sig = _ingest_signature(repo)
            cache_sig = "|".join(
                [
                    ingest_sig,
                    f"kind:{kind}",
                    f"min_members:{min_members}",
                    f"limit:{limit}",
                ]
            )
            cached = _get_cached_payload(repo, "clusters_ingest_v2", cache_sig)
            if cached is not None:
                return JSONResponse(cached)

        try:
            _report, entities, summaries, _author_cache = _load_context(repo)
        except HTTPException:
            entities, summaries, _author_cache = _load_ingest_context(repo)
        rows: list[dict[str, Any]] = []
        selected = _filter_cluster_summaries(
            summaries,
            kind=kind,
            min_members=min_members,
            max_clusters=limit,
        )
        if not selected:
            entities, ingest_summaries, _author_cache = _load_ingest_context(repo)
            selected = _filter_cluster_summaries(
                ingest_summaries,
                kind=kind,
                min_members=min_members,
                max_clusters=limit,
            )
        for summary in selected:
            canonical_title = _canonical_title(summary.canonical, entities)
            rows.append(
                {
                    "cluster_id": summary.cluster_id,
                    "cluster_type": summary.cluster_type,
                    "priority": round(summary.priority, 3),
                    "member_count": len(summary.members),
                    "canonical": summary.canonical,
                    "canonical_title": canonical_title,
                    "canonical_pr": summary.canonical_pr,
                    "canonical_issue": summary.canonical_issue,
                    "issue_attention": round(summary.issue_attention, 3),
                    "duplicate_pressure": round(summary.duplicate_pressure, 3),
                }
            )
        payload = {"repo": repo, "clusters": rows}
        if cache_sig is not None:
            _store_cached_payload(repo, cache_key, cache_sig, payload)
        return JSONResponse(payload)

    @app.get("/api/v1/repos/{repo:path}/clusters/{cluster_id}/detail", response_class=JSONResponse)
    @app.get("/api/repos/{repo:path}/clusters/{cluster_id}/detail", response_class=JSONResponse)
    def api_cluster_detail(
        repo: str,
        cluster_id: str,
        min_edge_score: float = Query(default=0.20, ge=0.0, le=1.0),
        include_authors: bool = Query(default=False),
    ) -> JSONResponse:
        payload: dict[str, Any]
        cache_key = "cluster_detail_v2"
        try:
            report = _load_report(repo)
            detail_sig = "|".join(
                [
                    _run_signature(report),
                    f"cluster:{cluster_id}",
                    f"min_edge:{min_edge_score:.4f}",
                    f"authors:{1 if include_authors else 0}",
                ]
            )
            cached = _get_cached_payload(repo, cache_key, detail_sig)
            if cached is not None:
                return JSONResponse(cached)
        except HTTPException:
            cache_key = "cluster_detail_ingest_v2"
            detail_sig = "|".join(
                [
                    _ingest_signature(repo),
                    f"cluster:{cluster_id}",
                    f"min_edge:{min_edge_score:.4f}",
                    f"authors:{1 if include_authors else 0}",
                ]
            )
            cached = _get_cached_payload(repo, cache_key, detail_sig)
            if cached is not None:
                return JSONResponse(cached)

        try:
            report, entities, summaries, author_cache = _load_context(repo)
            if cluster_id.startswith("ingest-cluster-"):
                ingest_entities, ingest_summaries, ingest_author_cache = _load_ingest_context(repo)
                summary = next((item for item in ingest_summaries if item.cluster_id == cluster_id), None)
                if summary is None:
                    raise HTTPException(status_code=404, detail=f"Unknown cluster id: {cluster_id}")
                payload = _build_ingest_cluster_detail_payload(
                    repo=repo,
                    summary=summary,
                    entities=ingest_entities,
                    author_cache=ingest_author_cache,
                    include_authors=include_authors,
                )
            else:
                payload = _build_cluster_detail_payload(
                    repo=repo,
                    report=report,
                    entities=entities,
                    summaries=summaries,
                    author_cache=author_cache,
                    cluster_id=cluster_id,
                    min_edge_score=min_edge_score,
                    include_authors=include_authors,
                )
        except HTTPException:
            ingest_entities, ingest_summaries, ingest_author_cache = _load_ingest_context(repo)
            summary = next((item for item in ingest_summaries if item.cluster_id == cluster_id), None)
            if summary is None:
                raise
            payload = _build_ingest_cluster_detail_payload(
                repo=repo,
                summary=summary,
                entities=ingest_entities,
                author_cache=ingest_author_cache,
                include_authors=include_authors,
            )
        _store_cached_payload(repo, cache_key, detail_sig, payload)
        return JSONResponse(payload)

    @app.get("/api/v1/repos/{repo:path}/authors", response_class=JSONResponse)
    @app.get("/api/repos/{repo:path}/authors", response_class=JSONResponse)
    def api_authors(repo: str, limit: int = Query(default=100, ge=1, le=1000)) -> JSONResponse:
        try:
            _report, entities, _summaries, author_cache = _load_context(repo)
        except HTTPException:
            entities, _summaries, author_cache = _load_ingest_context(repo)
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


def create_app_from_env() -> FastAPI:
    """Uvicorn factory entrypoint for --reload mode."""
    repo_path = os.environ.get("CARAPACE_REPO_PATH", ".")
    config = load_effective_config(repo_path=repo_path)
    return create_app(config)
