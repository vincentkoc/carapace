"""Hook registry and lifecycle orchestration."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from typing import Any


class HookName(str, Enum):
    BEFORE_NORMALIZE = "before_normalize"
    AFTER_NORMALIZE = "after_normalize"
    BEFORE_LOW_PASS = "before_low_pass"
    AFTER_LOW_PASS = "after_low_pass"
    BEFORE_FINGERPRINT = "before_fingerprint"
    AFTER_FINGERPRINT = "after_fingerprint"
    BEFORE_SIMILARITY = "before_similarity"
    AFTER_SIMILARITY = "after_similarity"
    BEFORE_CANONICAL = "before_canonical"
    AFTER_CANONICAL = "after_canonical"
    BEFORE_ACTION = "before_action"
    AFTER_ACTION = "after_action"
    ON_ERROR = "on_error"


HookCallback = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None]


class HookManager:
    """In-process hook manager with deterministic callback ordering."""

    def __init__(self) -> None:
        self._callbacks: dict[HookName, list[HookCallback]] = defaultdict(list)

    def register(self, name: HookName, callback: HookCallback) -> None:
        self._callbacks[name].append(callback)

    def emit(self, name: HookName, context: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
        result = dict(envelope)
        for callback in self._callbacks[name]:
            try:
                patch = callback(context, dict(result))
            except Exception as exc:  # pragma: no cover - error hook path
                self._emit_error(exc, context)
                continue
            if patch:
                result.update(patch)
        return result

    def _emit_error(self, exc: Exception, context: dict[str, Any]) -> None:
        for callback in self._callbacks[HookName.ON_ERROR]:
            callback({"exception": exc, **context}, {})
