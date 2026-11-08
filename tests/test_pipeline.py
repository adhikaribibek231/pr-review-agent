from pr_agent.search import Retrieval
from unittest.mock import Mock, patch

from pr_agent.models import Category, ChangedFile, Finding, Severity
from pr_agent.pipeline import (
    aggregate_findings,
    run_review,
    should_post,
)


def make_valid_changed_file() -> ChangedFile:
    return ChangedFile(
        filename="main.py",
        status="modified",
        additions=1,
        deletions=0,
        patch="""@@ -35,0 +36,1 @@
+def example() -> str:
""",
    )


def make_invalid_changed_file() -> ChangedFile:
    return ChangedFile(
        filename="other.py",
        status="modified",
        additions=1,
        deletions=0,
        patch="""@@ -9,0 +10,1 @@
+value = 42
""",
    )


def test_run_review_returns_validated_findings_without_posting() -> None:
    github = Mock()
    pull_request = Mock()

    with (
        patch(
            "pr_agent.pipeline.fetch_pull_request",
            return_value=pull_request,
        ),
        patch(
            "pr_agent.pipeline.fetch_changed_files",
            return_value=[make_valid_changed_file()],
        ),
        patch("pr_agent.pipeline.post_review") as mock_post_review,
    ):
        findings = run_review(
            github=github,
            repository_name="owner/repository",
            pr_number=1,
            post_requested=False,
        )

    assert len(findings) == 1
    assert findings[0].filename == "main.py"
    assert findings[0].line == 36

    mock_post_review.assert_not_called()


def test_run_review_posts_valid_findings_when_requested() -> None:
    github = Mock()
    pull_request = Mock()

    with (
        patch(
            "pr_agent.pipeline.fetch_pull_request",
            return_value=pull_request,
        ),
        patch(
            "pr_agent.pipeline.fetch_changed_files",
            return_value=[make_valid_changed_file()],
        ),
        patch("pr_agent.pipeline.post_review") as mock_post_review,
    ):
        findings = run_review(
            github=github,
            repository_name="owner/repository",
            pr_number=1,
            post_requested=True,
        )

    assert len(findings) == 1
    assert findings[0].filename == "main.py"
    assert findings[0].line == 36

    mock_post_review.assert_called_once_with(
        pull_request=pull_request,
        findings=findings,
    )


def test_run_review_does_not_post_when_no_findings_validate() -> None:
    github = Mock()
    pull_request = Mock()

    with (
        patch(
            "pr_agent.pipeline.fetch_pull_request",
            return_value=pull_request,
        ),
        patch(
            "pr_agent.pipeline.fetch_changed_files",
            return_value=[make_invalid_changed_file()],
        ),
        patch("pr_agent.pipeline.post_review") as mock_post_review,
    ):
        findings = run_review(
            github=github,
            repository_name="owner/repository",
            pr_number=1,
            post_requested=True,
        )

    assert findings == []

    mock_post_review.assert_not_called()


def test_run_review_ignores_noise_files() -> None:
    github = Mock()
    pull_request = Mock()

    noise_file = ChangedFile(
        filename="package-lock.json",
        status="modified",
        additions=1,
        deletions=0,
        patch="""@@ -35,0 +36,1 @@
+fake changed content
""",
    )

    with (
        patch(
            "pr_agent.pipeline.fetch_pull_request",
            return_value=pull_request,
        ),
        patch(
            "pr_agent.pipeline.fetch_changed_files",
            return_value=[noise_file],
        ),
        patch("pr_agent.pipeline.post_review") as mock_post_review,
    ):
        findings = run_review(
            github=github,
            repository_name="owner/repository",
            pr_number=1,
            post_requested=True,
        )

    assert findings == []

    mock_post_review.assert_not_called()


def test_aggregate_findings_sorts_by_filename_then_line_then_severity() -> None:
    findings = [
        Finding(
            filename="b.py",
            line=20,
            severity=Severity.WARNING,
            category=Category.BUG,
            message="Finding B",
        ),
        Finding(
            filename="a.py",
            line=30,
            severity=Severity.CRITICAL,
            category=Category.BUG,
            message="Finding A30",
        ),
        Finding(
            filename="a.py",
            line=10,
            severity=Severity.INFO,
            category=Category.STYLE,
            message="Finding A10",
        ),
        Finding(
            filename="a.py",
            line=30,
            severity=Severity.WARNING,
            category=Category.BUG,
            message="Finding A30 warning",
        ),
    ]

    result = aggregate_findings(findings)

    assert [
        (finding.filename, finding.line, finding.severity)
        for finding in result
    ] == [
        ("a.py", 10, Severity.INFO),
        ("a.py", 30, Severity.CRITICAL),
        ("a.py", 30, Severity.WARNING),
        ("b.py", 20, Severity.WARNING),
    ]


def test_should_post_returns_false_for_empty_findings() -> None:
    assert should_post([]) is False


def test_should_post_returns_true_when_findings_exist() -> None:
    findings = [
        Finding(
            filename="main.py",
            line=36,
            severity=Severity.CRITICAL,
            category=Category.BUG,
            message="Example finding",
        )
    ]

    assert should_post(findings) is True


def test_run_review_hands_hunks_and_retrieved_context_to_the_injected_reviewer() -> None:
    from pr_agent.models import RetrievedContext
    from pr_agent.pipeline import HunkContext

    seen: list[HunkContext] = []
    retrieved = RetrievedContext("lib.py", 1, 3, "def helper(): ...", 0.9)

    def reviewer(items: list[HunkContext]) -> list[Finding]:
        seen.extend(items)
        return [
            Finding(
                filename="main.py",
                line=36,
                severity=Severity.CRITICAL,
                category=Category.BUG,
                message="injected",
            )
        ]

    with (
        patch("pr_agent.pipeline.fetch_pull_request", return_value=Mock()),
        patch("pr_agent.pipeline.fetch_changed_files", return_value=[make_valid_changed_file()]),
        patch("pr_agent.pipeline.retrieve", return_value=Retrieval((retrieved,), ())),
    ):
        findings = run_review(
            github=Mock(),
            repository_name="owner/repository",
            pr_number=1,
            post_requested=False,
            reviewer=reviewer,
            index=Mock(),
        )

    assert [f.message for f in findings] == ["injected"]
    assert len(seen) == 1
    assert seen[0].hunk.filename == "main.py"
    assert seen[0].context == (retrieved,)


def test_run_review_without_an_index_gives_the_reviewer_empty_context() -> None:
    seen = []

    def reviewer(items):  # type: ignore[no-untyped-def]
        seen.extend(items)
        return []

    with (
        patch("pr_agent.pipeline.fetch_pull_request", return_value=Mock()),
        patch("pr_agent.pipeline.fetch_changed_files", return_value=[make_valid_changed_file()]),
    ):
        _ = run_review(Mock(), "owner/repository", 1, False, reviewer=reviewer)

    assert seen and seen[0].context == ()


def test_aggregate_findings_removes_exact_duplicates() -> None:
    one = Finding(filename="a.py", line=1, severity=Severity.WARNING, category=Category.BUG, message="same")
    other = Finding(filename="a.py", line=1, severity=Severity.WARNING, category=Category.BUG, message="different")
    assert aggregate_findings([one, one, other]) == [one, other]


# ---- only source code is reviewed ------------------------------------------------------------------------------------


def _hunks_for(*names: str) -> list[str]:
    from pr_agent.pipeline import build_hunks

    files = [ChangedFile(n, "modified", 1, 0, "@@ -1,0 +1,1 @@\n+x = 1") for n in names]
    return [h.filename for h in build_hunks(files)]


def test_source_files_are_reviewed_in_any_supported_language() -> None:
    names = ["a.py", "web/app.ts", "web/app.TSX", "cmd/main.go", "lib.rs", "run.sh", "q.sql"]
    assert _hunks_for(*names) == names


def test_prose_data_and_binaries_are_not_reviewed() -> None:
    skipped = ["README.md", "notes.txt", "data.json", "config.yaml", "logo.png", "paper.pdf", "poetry.lock",
               "Makefile", "Dockerfile", "outputs/draft.md", ".gitignore", "weird."]
    assert _hunks_for(*skipped) == []


def test_only_the_last_extension_counts() -> None:
    assert _hunks_for("archive.py.md", "template.html.py") == ["template.html.py"]
