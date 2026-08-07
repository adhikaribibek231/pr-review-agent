from pr_agent.models import Finding
from pr_agent.renderer import render_inline_comment


def test_render_inline_comment_contains_severity() -> None:
    finding = Finding(
        filename="main.py",
        line=36,
        severity="high",
        category="correctness",
        message="The annotation does not match the function return type.",
    )

    result = render_inline_comment(finding)

    assert "high" in result.lower()


def test_render_inline_comment_contains_category() -> None:
    finding = Finding(
        filename="main.py",
        line=36,
        severity="high",
        category="correctness",
        message="The annotation does not match the function return type.",
    )

    result = render_inline_comment(finding)

    assert "correctness" in result


def test_render_inline_comment_contains_message() -> None:
    message = "The annotation does not match the function return type."

    finding = Finding(
        filename="main.py",
        line=36,
        severity="high",
        category="correctness",
        message=message,
    )

    result = render_inline_comment(finding)

    assert message in result


def test_render_inline_comment_returns_string() -> None:
    finding = Finding(
        filename="main.py",
        line=36,
        severity="medium",
        category="type-safety",
        message="Potential type mismatch.",
    )

    result = render_inline_comment(finding)

    assert isinstance(result, str)


def test_render_inline_comment_is_not_empty() -> None:
    finding = Finding(
        filename="main.py",
        line=36,
        severity="low",
        category="maintainability",
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
        severity="high",
        category="type-safety",
        message=message,
    )

    result = render_inline_comment(finding)

    assert message in result
