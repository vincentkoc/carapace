"""Shared helpers for CLI command modules."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import yaml

from carapace.actioning import apply_routing_decisions
from carapace.config import CarapaceConfig, load_effective_config
from carapace.models import SourceEntity
from carapace.pipeline import CarapaceEngine
from carapace.repo_validation import validate_repo_path_matches
from carapace.services.command_runtime import CommandRuntime
from carapace.services.interfaces import Storage
from carapace.storage.base import StorageBackend

logger = logging.getLogger(__name__)

ALIAS_TO_CANONICAL = {
    "run": "scan",
    "ingest-github": "ingest",
    "process-stored": "process",
    "enrich-stored": "enrich",
    "scan-github": "triage",
    "db-audit": "audit",
    "serve-ui": "serve",
}


def default_enrich_workers() -> int:
    raw = os.environ.get("CARAPACE_ENRICH_WORKERS")
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    cpu = os.cpu_count() or 8
    return max(4, min(16, cpu))


def normalize_command(name: str) -> str:
    return ALIAS_TO_CANONICAL.get(name, name)


def load_json_entities(path: Path) -> list[SourceEntity]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("Input must be a JSON array of entities")
    return [SourceEntity.model_validate(item) for item in raw]


def load_yaml_dict(path: str | None) -> dict | None:
    if not path:
        return None
    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must decode to a mapping")
    return data


def load_config(args: argparse.Namespace) -> CarapaceConfig:
    return load_effective_config(
        repo_path=args.repo_path,
        org_defaults=load_yaml_dict(args.org_config),
        system_defaults=load_yaml_dict(args.system_config),
        runtime_override=load_yaml_dict(args.runtime_override),
    )


def build_engine(config: CarapaceConfig, storage: StorageBackend | None = None) -> CarapaceEngine:
    return CarapaceEngine(config=config, storage=storage)


def add_common_config_flags(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("--repo-path", default=".", help="Repository root path")
    cmd.add_argument("--org-config", help="Optional org defaults YAML")
    cmd.add_argument("--system-config", help="Optional system defaults YAML")
    cmd.add_argument("--runtime-override", help="Optional runtime override YAML")


def add_repo_validation_flags(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument(
        "--skip-repo-path-check",
        action="store_true",
        help="Skip validation that repo-path origin matches --repo",
    )


def add_report_flags(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument(
        "--report-include-singletons",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include singleton_orphan clusters in triage_report.md detail section",
    )


def add_github_rate_limit_flags(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument(
        "--gh-rate-limit-retries",
        type=int,
        default=2,
        help="Retries per GitHub API call when rate-limited",
    )
    cmd.add_argument(
        "--gh-secondary-backoff-seconds",
        type=float,
        default=5.0,
        help="Base backoff for secondary limits (exponential per retry)",
    )
    cmd.add_argument(
        "--gh-rate-limit-max-sleep-seconds",
        type=float,
        default=90.0,
        help="Maximum automatic sleep before surfacing a rate-limit failure",
    )


def validate_repo_path_if_needed(args: argparse.Namespace) -> None:
    if getattr(args, "skip_repo_path_check", False):
        return
    validate_repo_path_matches(args.repo_path, args.repo)


def kind_filter_from_arg(entity_kind: str) -> str | None:
    if entity_kind == "all":
        return None
    return entity_kind


def load_stored_entities(
    *,
    storage: Storage,
    repo: str,
    config: CarapaceConfig,
    entity_kind: str,
    limit: int,
) -> list[SourceEntity]:
    entities = storage.load_ingested_entities(
        repo=repo,
        include_closed=config.ingest.include_closed,
        include_drafts=config.ingest.include_drafts,
        kind=kind_filter_from_arg(entity_kind),
    )
    if limit > 0:
        entities = entities[:limit]
    return entities


def maybe_apply_routing(args: argparse.Namespace, entities: list[SourceEntity], report, *, runtime: CommandRuntime) -> None:
    if not getattr(args, "apply_routing", False):
        return

    id_to_number: dict[str, int] = {}
    for entity in entities:
        if entity.number is not None:
            id_to_number[entity.id] = entity.number

    sink = runtime.sink_connector_cls(
        repo=args.repo,
        gh_bin=args.gh_bin,
        rate_limit_retries=args.gh_rate_limit_retries,
        secondary_backoff_base_seconds=args.gh_secondary_backoff_seconds,
        rate_limit_max_sleep_seconds=args.gh_rate_limit_max_sleep_seconds,
        entity_number_resolver=lambda entity_id: id_to_number[entity_id],
        dry_run=not args.live_actions,
    )
    apply_routing_decisions(report, sink)
