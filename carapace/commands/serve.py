"""Serve UI command."""

from __future__ import annotations

import argparse
import logging
import os

from carapace.commands.common import CommandRuntime, load_config, validate_repo_path_if_needed

logger = logging.getLogger(__name__)


def run(args: argparse.Namespace, *, runtime: CommandRuntime) -> int:
    _ = runtime
    validate_repo_path_if_needed(args)
    config = load_config(args)
    if config.storage.backend != "sqlite":
        raise ValueError("serve currently requires storage.backend=sqlite")

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency/runtime
        raise RuntimeError("Missing optional UI dependencies. Install with: pip install 'carapace[ui]'") from exc

    logger.info("Starting UI on http://%s:%s (repo=%s)", args.host, args.port, args.repo or "auto")
    if args.reload:
        os.environ["CARAPACE_REPO_PATH"] = str(args.repo_path)
        uvicorn.run(
            "carapace.webapp:create_app_from_env",
            host=args.host,
            port=args.port,
            reload=True,
            factory=True,
            log_level=args.log_level.lower(),
        )
    else:
        from carapace.webapp import create_app

        app = create_app(config)
        uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level=args.log_level.lower())
    return 0

