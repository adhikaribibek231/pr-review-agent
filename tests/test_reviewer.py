from unittest.mock import Mock, patch

import pytest

from pr_agent.diff_parser import parse_hunks
from pr_agent.llm import FakeLLMClient, PromptTooLargeError, findings_reply
from pr_agent.models import Category, ChangedFile, Finding, HunkContext, Severity
from pr_agent.pipeline import run_review
from pr_agent.reviewer import (
    LLMReviewer,
    describe_validation_error,
    extract_json_object,
    parse_review,
)
from pydantic import ValidationError

PATCH = "@@ -1,0 +1,2 @@\n+query = f\"SELECT * FROM t WHERE id={uid}\"\n+run(query)"


def item(filename: str = "app.py") -> HunkContext:
    return HunkContext(parse_hunks(filename, PATCH)[0], ())


def finding(line: int = 1, filename: str = "app.py", message: str = "SQL injection.") -> Finding:
    return Finding(
        filename=filename,
        line=line,
        severity=Severity.CRITICAL,
        category=Category.SECURITY,
        message=message,
    )


# --------------------------------------------------------------------------
# extract_json_object / parse_review
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"findings": []}',
        '  {"findings": []}\n',
        '```json\n{"findings": []}\n```',
        '```\n{"findings": []}\n```',
        'Sure! Here is the review:\n{"findings": []}',
        '{"findings": []}\nHope that helps!',
    ],
)
def test_extracts_the_object_from_common_wrappers(raw: str) -> None:
    assert parse_review(raw).findings == []


def test_extraction_does_not_repair_broken_json() -> None:
    with pytest.raises(ValidationError):
        parse_review('{"findings": [')
    with pytest.raises(ValidationError):
        parse_review("I found no problems.")


def test_extract_json_object_leaves_clean_json_untouched() -> None:
    assert extract_json_object('{"a": 1}') == '{"a": 1}'


def test_error_description_is_one_path_and_problem_per_line() -> None:
    with pytest.raises(ValidationError) as error:
        parse_review('{"findings": [{"filename": "a.py", "line": 0, "severity": "high", '
                     '"category": "bug", "message": "m"}]}')
    text = describe_validation_error(error.value)
    assert "- findings.0.line:" in text
    assert "- findings.0.severity:" in text
    assert "pydantic.dev" not in text  # no doc URLs in what the model reads


def test_whole_reply_errors_have_a_readable_path() -> None:
    with pytest.raises(ValidationError) as error:
        parse_review("not json")
    assert "(whole reply)" in describe_validation_error(error.value)


# --------------------------------------------------------------------------
# LLMReviewer
# --------------------------------------------------------------------------


def test_returns_validated_findings_from_a_good_reply() -> None:
    client = FakeLLMClient([findings_reply([finding(1), finding(2, message="Second.")])])
    result = LLMReviewer(client)([item()])
    assert [(f.line, f.severity) for f in result] == [(1, Severity.CRITICAL), (2, Severity.CRITICAL)]
    assert len(client.calls) == 1


def test_an_empty_reply_is_a_clean_review_with_no_retry() -> None:
    client = FakeLLMClient()
    assert LLMReviewer(client)([item()]) == []
    assert len(client.calls) == 1


def test_sends_the_numbered_hunk_to_the_client() -> None:
    client = FakeLLMClient()
    _ = LLMReviewer(client)([item()])
    assert "     1 +query" in client.calls[0].user
    assert "findings" in client.calls[0].system


def test_invalid_reply_is_retried_once_with_the_error_attached() -> None:
    bad = '{"findings": [{"filename": "app.py", "line": 1, "severity": "high", "category": "bug", "message": "m"}]}'
    client = FakeLLMClient([bad, findings_reply([finding(1)])])
    result = LLMReviewer(client)([item()])

    assert [f.line for f in result] == [1]
    assert len(client.calls) == 2
    retry = client.calls[1]
    assert retry.system == client.calls[0].system  # same instructions
    assert "findings.0.severity" in retry.user  # what was wrong
    assert '"severity": "high"' in retry.user  # what the model said
    assert retry.user.startswith(client.calls[0].user)  # original task still present


def test_gives_up_after_the_attempt_cap_and_reports_nothing() -> None:
    client = FakeLLMClient(["garbage", "more garbage", "never reached"])
    assert LLMReviewer(client)([item()]) == []
    assert len(client.calls) == 2  # the governor: one retry, never an endless loop


def test_max_attempts_one_disables_the_retry() -> None:
    client = FakeLLMClient(["garbage", findings_reply([finding()])])
    assert LLMReviewer(client, max_attempts=1)([item()]) == []
    assert len(client.calls) == 1


def test_max_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError):
        LLMReviewer(FakeLLMClient(), max_attempts=0)


def test_hunks_are_reviewed_independently() -> None:
    client = FakeLLMClient(["garbage", "garbage", findings_reply([finding(1, "b.py")])])
    result = LLMReviewer(client)([item("a.py"), item("b.py")])
    assert [f.filename for f in result] == ["b.py"]  # a.py failed, b.py unaffected
    assert len(client.calls) == 3
    assert 'file="b.py"' in client.calls[2].user


def test_one_call_per_hunk_in_order() -> None:
    client = FakeLLMClient()
    _ = LLMReviewer(client)([item("a.py"), item("b.py"), item("c.py")])
    assert [c.user.split('"')[1] for c in client.calls] == ["a.py", "b.py", "c.py"]


def test_transport_errors_propagate_instead_of_looking_like_a_clean_review() -> None:
    client = Mock()
    client.complete_json.side_effect = TimeoutError("model timed out")
    with pytest.raises(TimeoutError):
        LLMReviewer(client)([item()])


def test_stricter_uses_the_tightened_system_prompt() -> None:
    client = FakeLLMClient()
    reviewer = LLMReviewer(client)
    _ = reviewer.review([item()])
    _ = reviewer.review([item()], stricter=True)
    assert "cite ONLY" not in client.calls[0].system
    assert "cite ONLY" in client.calls[1].system


def test_no_items_means_no_calls() -> None:
    client = FakeLLMClient()
    assert LLMReviewer(client)([]) == []
    assert client.calls == []


# --------------------------------------------------------------------------
# plugged into the real pipeline
# --------------------------------------------------------------------------


def test_llm_reviewer_drops_into_run_review_and_grounding_still_applies() -> None:
    """End to end: the fake model cites one real added line and one that isn't in
    the diff. The pipeline's validator must drop the second."""
    changed = ChangedFile("app.py", "modified", 2, 0, PATCH)
    client = FakeLLMClient([findings_reply([finding(1), finding(999, message="Hallucinated line.")])])

    with (
        patch("pr_agent.pipeline.fetch_pull_request", return_value=Mock()),
        patch("pr_agent.pipeline.fetch_changed_files", return_value=[changed]),
    ):
        result = run_review(Mock(), "o/r", 1, False, reviewer=LLMReviewer(client))

    assert [f.line for f in result] == [1]


# ---- a prompt too large for the provider ------------------------------------------------------------------


class TooBigUnless:
    """Rejects any prompt longer than `limit` characters, like a provider with a request cap."""

    def __init__(self, limit: int) -> None:
        self.limit, self.sizes = limit, []

    def complete_json(self, *, system: str, user: str) -> str:
        self.sizes.append(len(user))
        if len(user) > self.limit:
            raise PromptTooLargeError("too big")
        return findings_reply([finding(1)])


def big_item() -> HunkContext:
    from pr_agent.models import RetrievedContext

    chunks = tuple(RetrievedContext(f"lib{i}.py", 1, 5, "x = 1\n" * 300, 0.5, "reason") for i in range(6))
    return HunkContext(parse_hunks("app.py", PATCH)[0], chunks)


def test_a_too_large_prompt_is_retried_with_less_context_and_still_reviewed() -> None:
    client = TooBigUnless(limit=2500)
    result = LLMReviewer(client)([big_item()])
    assert [f.line for f in result] == [1]  # reviewed, with less context
    assert client.sizes == sorted(client.sizes, reverse=True)  # each attempt was smaller
    assert len(client.sizes) > 1


def test_when_even_no_context_fits_the_hunk_is_skipped_not_crashed() -> None:
    client = TooBigUnless(limit=10)
    assert LLMReviewer(client)([big_item()]) == []
    assert len(client.sizes) <= 8  # it gave up, rather than looping


def test_a_prompt_that_fits_is_sent_once_with_all_its_context() -> None:
    client = TooBigUnless(limit=1_000_000)
    _ = LLMReviewer(client)([big_item()])
    assert len(client.sizes) == 1
