from pr_agent.models import Category, Finding, Severity
from pr_agent.renderer import render_inline_comment


def test_render_inline_comment_contains_severity() -> None:
    finding = Finding(
        filename="main.py",
        line=36,
        severity=Severity.CRITICAL,
        category=Category.BUG,
        message="The annotation does not match the function return type.",
    )

    result = render_inline_comment(finding)

    assert "CRITICAL" in result


def test_render_inline_comment_contains_category() -> None:
    finding = Finding(
        filename="main.py",
        line=36,
        severity=Severity.CRITICAL,
        category=Category.BUG,
        message="The annotation does not match the function return type.",
    )

    result = render_inline_comment(finding)

    assert "bug" in result


def test_render_inline_comment_contains_message() -> None:
    message = "The annotation does not match the function return type."

    finding = Finding(
        filename="main.py",
        line=36,
        severity=Severity.CRITICAL,
        category=Category.BUG,
        message=message,
    )

    result = render_inline_comment(finding)

    assert message in result


def test_render_inline_comment_returns_string() -> None:
    finding = Finding(
        filename="main.py",
        line=36,
        severity=Severity.WARNING,
        category=Category.BUG,
        message="Potential type mismatch.",
    )

    result = render_inline_comment(finding)

    assert isinstance(result, str)


def test_render_inline_comment_is_not_empty() -> None:
    finding = Finding(
        filename="main.py",
        line=36,
        severity=Severity.INFO,
        category=Category.CONVENTION,
        message="Consider simplifying this expression.",
    )

    result = render_inline_comment(finding)

    assert result.strip()


def test_render_inline_comment_preserves_message_text() -> None:
    message = (
        "This value is annotated as list[str], "
        "but the called function returns Path."
    )

    finding = Finding(
        filename="main.py",
        line=36,
        severity=Severity.CRITICAL,
        category=Category.BUG,
        message=message,
    )

    result = render_inline_comment(finding)

    assert message in result


def test_render_inline_comment_has_balanced_bold_markers() -> None:
    # An unclosed ** makes GitHub bold the rest of the comment. The substring
    # assertions above can't catch that; counting the markers can.
    finding = Finding(
        filename="a.py",
        line=1,
        severity=Severity.WARNING,
        category=Category.SECURITY,
        message="Something is wrong.",
    )
    assert render_inline_comment(finding).count("**") % 2 == 0
