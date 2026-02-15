from carapace.config import CarapaceConfig
from carapace.hooks import HookManager, HookName
from carapace.models import EntityKind, SourceEntity
from carapace.pipeline import CarapaceEngine


def test_hook_invocation() -> None:
    calls: list[str] = []
    hooks = HookManager()

    hooks.register(HookName.BEFORE_LOW_PASS, lambda ctx, env: calls.append("before_low_pass") or None)
    hooks.register(HookName.AFTER_CANONICAL, lambda ctx, env: calls.append("after_canonical") or None)

    entity = SourceEntity.model_validate(
        {
            "id": "1",
            "repo": "acme/repo",
            "kind": EntityKind.PR,
            "title": "Test",
            "body": "",
            "author": "dev",
            "changed_files": ["src/a.py"],
        }
    )

    engine = CarapaceEngine(config=CarapaceConfig(), hooks=hooks)
    engine.scan_entities([entity])

    assert "before_low_pass" in calls
    assert "after_canonical" in calls
