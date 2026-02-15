from carapace.fingerprint import build_fingerprint
from carapace.models import DiffHunk, EntityKind, ExternalReviewSignal, SourceEntity


def test_build_fingerprint_extracts_expected_fields() -> None:
    entity = SourceEntity(
        id="pr:12",
        repo="acme/repo",
        kind=EntityKind.PR,
        title="Fix parser tokenization",
        body="Fixes #99 and improves lexer",
        author="alice",
        changed_files=["src/parser/core.py", "tests/test_parser.py"],
        diff_hunks=[
            DiffHunk(
                file_path="src/parser/core.py",
                context="@@ -10,2 +10,3 @@",
                added_lines=["new token"],
                removed_lines=["old token"],
            )
        ],
        linked_issues=["99"],
        approvals=2,
        additions=7,
        deletions=2,
        external_reviews=[ExternalReviewSignal(provider="coderabbit", overall_score=0.8, confidence=0.6)],
    )

    fp = build_fingerprint(entity, text_embedding=[0.1, 0.2, 0.3], diff_embedding=[0.3, 0.2, 0.1])

    assert fp.entity_id == "pr:12"
    assert "src/parser/*" in fp.module_buckets
    assert any(bucket.startswith("tests/") for bucket in fp.module_buckets)
    assert fp.linked_issues == ["99"]
    assert len(fp.hunk_signatures) == 1
    assert fp.reviewer_score == 0.8
    assert fp.embedding == [0.1, 0.2, 0.3]
    assert fp.text_embedding == [0.1, 0.2, 0.3]
    assert fp.diff_embedding == [0.3, 0.2, 0.1]


def test_issue_fingerprint_includes_self_issue_number_in_links() -> None:
    entity = SourceEntity(
        id="issue:42",
        repo="acme/repo",
        kind=EntityKind.ISSUE,
        number=42,
        title="Bug in parser",
        body="Repro steps...",
        author="alice",
    )

    fp = build_fingerprint(entity, text_embedding=[0.1, 0.2, 0.3], diff_embedding=[])
    assert fp.linked_issues == ["42"]
