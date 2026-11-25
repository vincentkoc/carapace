"""Audit command."""

from __future__ import annotations

import argparse
import json

from carapace.commands.common import CommandRuntime, load_config, validate_repo_path_if_needed


def run(args: argparse.Namespace, *, runtime: CommandRuntime) -> int:
    validate_repo_path_if_needed(args)
    config = load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("audit currently requires storage.backend=sqlite")

    storage = runtime.storage_cls(config.storage.sqlite_path)
    state = storage.get_ingest_state(args.repo)
    summary = storage.ingest_audit_summary(args.repo)
    quality = storage.ingest_quality_stats(args.repo)
    pr_quality = storage.ingest_quality_stats(args.repo, kind="pr")
    issue_quality = storage.ingest_quality_stats(args.repo, kind="issue")

    payload = {
        "repo": args.repo,
        "db_path": str(config.storage.sqlite_path),
        "ingest_state": state,
        "summary": summary,
        "quality": {
            "all": quality,
            "pr": pr_quality,
            "issue": issue_quality,
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Repo: {args.repo}")
    print(f"DB: {config.storage.sqlite_path}")
    print("Ingest state: completed={completed} phase={phase} pr_next_page={pr_next_page} issue_next_page={issue_next_page}".format(**state))
    print(
        "Entities: total={total} prs={prs} issues={issues}".format(
            total=summary["total"],
            prs=summary["by_kind"].get("pr", 0),
            issues=summary["by_kind"].get("issue", 0),
        )
    )
    print(
        "PRs: open={open} closed={closed} drafts={drafts} enriched={enriched}".format(
            open=summary["by_kind_state"].get("pr", {}).get("open", 0),
            closed=summary["by_kind_state"].get("pr", {}).get("closed", 0),
            drafts=summary["draft_prs"],
            enriched=summary["enriched_prs"],
        )
    )
    print(
        "Issues: open={open} closed={closed}".format(
            open=summary["by_kind_state"].get("issue", {}).get("open", 0),
            closed=summary["by_kind_state"].get("issue", {}).get("closed", 0),
        )
    )
    print("Quality(all): missing_changed_files={missing_changed_files} missing_diff_hunks={missing_diff_hunks} ci_unknown={ci_unknown} enriched_rows={enriched_rows}".format(**quality))
    print("Quality(pr): missing_changed_files={missing_changed_files} missing_diff_hunks={missing_diff_hunks} ci_unknown={ci_unknown} enriched_rows={enriched_rows}".format(**pr_quality))
    print("Quality(issue): missing_changed_files={missing_changed_files} missing_diff_hunks={missing_diff_hunks} ci_unknown={ci_unknown} enriched_rows={enriched_rows}".format(**issue_quality))
    print(
        "Integrity: kind_id_prefix_mismatch={kind_id_prefix_mismatch} kind_payload_mismatch={kind_payload_mismatch} repo_payload_mismatch={repo_payload_mismatch} entity_number_mismatch={entity_number_mismatch}".format(
            **summary["integrity"]
        )
    )
    levels = summary["enrich_levels"]
    print("Enrich levels (PR): " + ", ".join(f"{level}={count}" for level, count in sorted(levels.items())))
    print(
        "Fingerprint cache rows: total={total} by_model={models}".format(
            total=summary["fingerprint_cache_rows"],
            models=", ".join(f"{model}={count}" for model, count in sorted(summary["fingerprint_cache_by_model"].items())),
        )
    )
    return 0

