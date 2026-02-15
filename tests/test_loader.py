from carapace.config import IngestConfig
from carapace.loader import ingest_github_to_sqlite
from carapace.models import EntityKind, SourceEntity
from carapace.storage import SQLiteStorage


class FakeConnector:
    def __init__(self) -> None:
        self.pull_calls = []
        self.issue_calls = []

    def fetch_pull_page(
        self,
        *,
        page: int,
        per_page: int = 100,
        state: str = "open",
        include_drafts: bool = False,
        enrich_details: bool = False,
        enrich_comments: bool = False,
    ):
        self.pull_calls.append((page, per_page, state, include_drafts, enrich_details, enrich_comments))
        if page == 1:
            return [
                SourceEntity(
                    id="pr:1",
                    repo="acme/repo",
                    kind=EntityKind.PR,
                    number=1,
                    state="open",
                    is_draft=False,
                    title="P1",
                    author="a",
                )
            ]
        if page == 2:
            return [
                SourceEntity(
                    id="pr:2",
                    repo="acme/repo",
                    kind=EntityKind.PR,
                    number=2,
                    state="open",
                    is_draft=False,
                    title="P2",
                    author="a",
                )
            ]
        return []

    def fetch_issue_page(self, *, page: int, per_page: int = 100, state: str = "open"):
        self.issue_calls.append((page, per_page, state))
        return []


def test_loader_is_stateful_and_resumes(tmp_path) -> None:
    storage = SQLiteStorage(tmp_path / "carapace.db")
    connector = FakeConnector()

    cfg = IngestConfig(include_issues=False, page_size=1, resume=True)

    result1 = ingest_github_to_sqlite(
        connector,
        storage,
        repo="acme/repo",
        ingest_cfg=cfg,
        max_prs=1,
        max_issues=0,
    )
    assert result1.prs_ingested == 1

    state = storage.get_ingest_state("acme/repo")
    assert state["pr_next_page"] == 2

    result2 = ingest_github_to_sqlite(
        connector,
        storage,
        repo="acme/repo",
        ingest_cfg=cfg,
        max_prs=2,
        max_issues=0,
    )
    assert result2.prs_ingested == 1

    loaded = storage.load_ingested_entities("acme/repo", include_closed=True, include_drafts=True)
    assert {entity.id for entity in loaded} == {"pr:1", "pr:2"}


def test_loader_restarts_from_page_one_after_completed_cycle(tmp_path) -> None:
    storage = SQLiteStorage(tmp_path / "carapace.db")
    connector = FakeConnector()
    cfg = IngestConfig(include_issues=False, page_size=1, resume=True)

    # Full cycle with no caps should complete and mark done.
    ingest_github_to_sqlite(
        connector,
        storage,
        repo="acme/repo",
        ingest_cfg=cfg,
        max_prs=0,
        max_issues=0,
    )
    state = storage.get_ingest_state("acme/repo")
    assert state["completed"] == 1
    assert state["phase"] == "done"

    prior_calls = len(connector.pull_calls)
    ingest_github_to_sqlite(
        connector,
        storage,
        repo="acme/repo",
        ingest_cfg=cfg,
        max_prs=0,
        max_issues=0,
    )
    new_calls = connector.pull_calls[prior_calls:]
    assert new_calls[0][0] == 1
