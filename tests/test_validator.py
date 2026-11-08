from pr_agent.diff_parser import DiffHunk
from pr_agent.models import Category, Finding, Severity
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
        severity=Severity.CRITICAL,
        category=Category.BUG,
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
        severity=Severity.CRITICAL,
        category=Category.BUG,
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
        severity=Severity.WARNING,
        category=Category.BUG,
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
            severity=Severity.CRITICAL,
            category=Category.BUG,
            message="First issue.",
        ),
        Finding(
            filename="main.py",
            line=40,
            severity=Severity.WARNING,
            category=Category.CONVENTION,
            message="Second issue.",
        ),
        Finding(
            filename="utils.py",
            line=15,
            severity=Severity.INFO,
            category=Category.STYLE,
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
        severity=Severity.CRITICAL,
        category=Category.BUG,
        message="Valid issue.",
    )

    invalid_line = Finding(
        filename="main.py",
        line=999,
        severity=Severity.CRITICAL,
        category=Category.BUG,
        message="Invalid line.",
    )

    invalid_file = Finding(
        filename="utils.py",
        line=36,
        severity=Severity.WARNING,
        category=Category.BUG,
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
        severity=Severity.CRITICAL,
        category=Category.BUG,
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
        severity=Severity.CRITICAL,
        category=Category.BUG,
        message="Issue attached only to a deleted line.",
    )

    result = validate_findings(
        findings=[finding],
        hunks=[hunk],
    )

    assert result == []


# --------------------------------------------------------------------------
# check_findings: the report the retry loop needs
# --------------------------------------------------------------------------


def _report_fixture() -> tuple[list[Finding], list[DiffHunk]]:
    from pr_agent.diff_parser import parse_hunks

    hunks = parse_hunks("a.py", "@@ -1,0 +1,2 @@\n+x = 1\n+y = 2")
    make = lambda filename, line: Finding(  # noqa: E731
        filename=filename, line=line, severity=Severity.WARNING, category=Category.BUG, message="m"
    )
    return [make("a.py", 1), make("a.py", 99), make("other.py", 1), make("a.py", 2)], hunks


def test_check_findings_reports_kept_and_dropped_separately() -> None:
    from pr_agent.validator import check_findings

    findings, hunks = _report_fixture()
    result = check_findings(findings, hunks)

    assert [(f.filename, f.line) for f in result.kept] == [("a.py", 1), ("a.py", 2)]
    assert [(f.filename, f.line) for f in result.dropped] == [("a.py", 99), ("other.py", 1)]
    assert result.has_unsupported


def test_nothing_dropped_means_no_unsupported_findings() -> None:
    from pr_agent.validator import check_findings

    findings, hunks = _report_fixture()
    result = check_findings([findings[0]], hunks)
    assert not result.has_unsupported
    assert result.dropped == ()


def test_kept_plus_dropped_always_accounts_for_every_finding() -> None:
    from pr_agent.validator import check_findings

    findings, hunks = _report_fixture()
    result = check_findings(findings, hunks)
    assert len(result.kept) + len(result.dropped) == len(findings)


def test_validate_findings_is_the_kept_half_of_check_findings() -> None:
    from pr_agent.validator import check_findings

    findings, hunks = _report_fixture()
    assert validate_findings(findings, hunks) == list(check_findings(findings, hunks).kept)


# ---- findings about a pure removal ----------------------------------------------------------------------------------------------------


def _removal_case():
    from pr_agent.diff_parser import parse_hunks
    from pr_agent.validator import check_findings

    hunks = parse_hunks("utils.py", "@@ -1,6 +1,3 @@\n def keep(x):\n     return x\n \n-def legacy(items):\n-    return sum(items)\n-\n def other(x):")
    make = lambda filename, line: Finding(  # noqa: E731
        filename=filename, line=line, severity=Severity.CRITICAL, category=Category.BUG, message="m")
    return hunks, make, check_findings


def test_a_finding_beside_a_pure_removal_is_grounded_on_the_adjacent_context_line() -> None:
    hunks, make, check = _removal_case()
    assert [f.line for f in check([make("utils.py", 3), make("utils.py", 4)], hunks).kept] == [3, 4]


def test_context_lines_far_from_the_removal_are_still_rejected() -> None:
    hunks, make, check = _removal_case()
    assert check([make("utils.py", 1)], hunks).kept == ()


def test_the_removal_exception_does_not_apply_to_other_files() -> None:
    hunks, make, check = _removal_case()
    assert check([make("other.py", 3)], hunks).kept == ()


def test_a_hunk_with_additions_still_rejects_context_lines() -> None:
    from pr_agent.diff_parser import parse_hunks
    from pr_agent.validator import check_findings

    hunks = parse_hunks("a.py", "@@ -1,3 +1,3 @@\n keep\n-old\n+new\n tail")
    context_line = Finding(filename="a.py", line=1, severity=Severity.INFO, category=Category.STYLE, message="m")
    assert check_findings([context_line], hunks).kept == ()
