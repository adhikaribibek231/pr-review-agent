from pr_agent.diff_parser import DiffHunk
from pr_agent.models import Finding
from pr_agent.validator import validate_findings


def test_valid_added_line_survives() -> None:
    hunk = DiffHunk(
        filename="main.py",
        patch="",
        added_lines=frozenset({36}),
        deleted_lines=frozenset(),
    )

    finding = Finding(
        filename="main.py",
        line=36,
        severity="high",
        category="correctness",
        message="Example issue.",
    )

    result = validate_findings(
        findings=[finding],
        hunks=[hunk],
    )

    assert result == [finding]


def test_nonexistent_line_is_removed() -> None:
    hunk = DiffHunk(
        filename="main.py",
        patch="",
        added_lines=frozenset({36}),
        deleted_lines=frozenset(),
    )

    finding = Finding(
        filename="main.py",
        line=999,
        severity="high",
        category="correctness",
        message="This line does not exist in the diff.",
    )

    result = validate_findings(
        findings=[finding],
        hunks=[hunk],
    )

    assert result == []


def test_correct_line_in_wrong_file_is_removed() -> None:
    hunk = DiffHunk(
        filename="main.py",
        patch="",
        added_lines=frozenset({36}),
        deleted_lines=frozenset(),
    )

    finding = Finding(
        filename="utils.py",
        line=36,
        severity="medium",
        category="correctness",
        message="Correct line number, but wrong file.",
    )

    result = validate_findings(
        findings=[finding],
        hunks=[hunk],
    )

    assert result == []


def test_multiple_valid_findings_survive() -> None:
    hunks = [
        DiffHunk(
            filename="main.py",
            patch="",
            added_lines=frozenset({36, 40}),
            deleted_lines=frozenset(),
        ),
        DiffHunk(
            filename="utils.py",
            patch="",
            added_lines=frozenset({15}),
            deleted_lines=frozenset(),
        ),
    ]

    findings = [
        Finding(
            filename="main.py",
            line=36,
            severity="high",
            category="correctness",
            message="First issue.",
        ),
        Finding(
            filename="main.py",
            line=40,
            severity="medium",
            category="maintainability",
            message="Second issue.",
        ),
        Finding(
            filename="utils.py",
            line=15,
            severity="low",
            category="style",
            message="Third issue.",
        ),
    ]

    result = validate_findings(
        findings=findings,
        hunks=hunks,
    )

    assert result == findings


def test_mixed_valid_and_invalid_findings() -> None:
    hunk = DiffHunk(
        filename="main.py",
        patch="",
        added_lines=frozenset({36}),
        deleted_lines=frozenset(),
    )

    valid_finding = Finding(
        filename="main.py",
        line=36,
        severity="high",
        category="correctness",
        message="Valid issue.",
    )

    invalid_line = Finding(
        filename="main.py",
        line=999,
        severity="high",
        category="correctness",
        message="Invalid line.",
    )

    invalid_file = Finding(
        filename="utils.py",
        line=36,
        severity="medium",
        category="correctness",
        message="Invalid file.",
    )

    result = validate_findings(
        findings=[
            valid_finding,
            invalid_line,
            invalid_file,
        ],
        hunks=[hunk],
    )

    assert result == [valid_finding]


def test_empty_findings_returns_empty_list() -> None:
    hunk = DiffHunk(
        filename="main.py",
        patch="",
        added_lines=frozenset({36}),
        deleted_lines=frozenset(),
    )

    result = validate_findings(
        findings=[],
        hunks=[hunk],
    )

    assert result == []


def test_empty_hunks_rejects_all_findings() -> None:
    finding = Finding(
        filename="main.py",
        line=36,
        severity="high",
        category="correctness",
        message="Example issue.",
    )

    result = validate_findings(
        findings=[finding],
        hunks=[],
    )

    assert result == []


def test_deleted_only_line_is_not_valid_for_right_side_comment() -> None:
    hunk = DiffHunk(
        filename="main.py",
        patch="",
        added_lines=frozenset(),
        deleted_lines=frozenset({36}),
    )

    finding = Finding(
        filename="main.py",
        line=36,
        severity="high",
        category="correctness",
        message="Issue attached only to a deleted line.",
    )

    result = validate_findings(
        findings=[finding],
        hunks=[hunk],
    )

    assert result == []
