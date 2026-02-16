"""Offline scan command."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from carapace.commands.common import CommandRuntime, build_engine, load_config, load_json_entities
from carapace.reporting import write_report_bundle

logger = logging.getLogger(__name__)


def run(args: argparse.Namespace, *, runtime: CommandRuntime) -> int:
    _ = runtime
    entities = load_json_entities(Path(args.input))
    config = load_config(args)
    engine = build_engine(config)
    report = engine.scan_entities(entities)
    write_report_bundle(
        report,
        args.output_dir,
        entities=entities,
        include_singleton_orphans=args.report_include_singletons,
    )
    logger.info("Scan complete: processed=%s clusters=%s", report.processed_entities, len(report.clusters))
    return 0

