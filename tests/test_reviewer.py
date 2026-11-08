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


# ---- one call per file -------------------------------------------------------------------------------------------------


def hunk_item(filename: str, patch: str) -> HunkContext:
    return HunkContext(parse_hunks(filename, patch)[0], ())


def added(n: int, start: int = 1, tag: str = "x") -> str:
    return f"@@ -{start - 1},0 +{start},{n} @@\n" + "\n".join(f"+{tag}{i} = {i}" for i in range(n))


def test_all_hunks_of_one_file_are_reviewed_in_a_single_call() -> None:
    from pr_agent.reviewer import group_into_batches

    items = [hunk_item("a.py", added(2, 1, "p")), hunk_item("a.py", added(2, 40, "q"))]
    assert [len(b) for b in group_into_batches(items)] == [2]
    client = FakeLLMClient()
    _ = LLMReviewer(client)(items)
    assert len(client.calls) == 1
    user = client.calls[0].user
    assert user.count("<diff file=\"a.py\">") == 2 and "p0 = 0" in user and "q0 = 0" in user


def test_a_single_prompt_carries_one_copy_of_the_instructions_for_the_whole_batch() -> None:
    client = FakeLLMClient()
    _ = LLMReviewer(client)([hunk_item("a.py", added(1, 1)), hunk_item("a.py", added(1, 30))])
    assert client.calls[0].system.count("JSON Schema") == 1


def test_different_files_get_separate_calls_in_order_of_first_appearance() -> None:
    from pr_agent.reviewer import group_into_batches

    items = [hunk_item("a.py", added(1)), hunk_item("b.py", added(1)), hunk_item("a.py", added(1, 30))]
    batches = group_into_batches(items)
    assert [[i.hunk.filename for i in b] for b in batches] == [["a.py", "a.py"], ["b.py"]]  # non-adjacent a.py hunks grouped


def test_a_file_with_large_diffs_is_split_into_several_calls() -> None:
    from pr_agent.reviewer import MAX_BATCH_DIFF_CHARS, group_into_batches

    big = added(MAX_BATCH_DIFF_CHARS // 20, 1)  # each patch is about a third of the limit... times a few
    items = [hunk_item("a.py", big.replace("+x", f"+h{k}_")) for k in range(6)]
    batches = group_into_batches(items)
    assert len(batches) > 1
    for batch in batches[:-1]:
        assert sum(len(i.hunk.patch) for i in batch) <= MAX_BATCH_DIFF_CHARS or len(batch) == 1


def test_the_number_of_hunks_per_call_is_capped() -> None:
    from pr_agent.reviewer import MAX_BATCH_HUNKS, group_into_batches

    items = [hunk_item("a.py", added(1, 10 * k + 1, f"v{k}_")) for k in range(MAX_BATCH_HUNKS + 3)]
    assert [len(b) for b in group_into_batches(items)] == [MAX_BATCH_HUNKS, 3]


def test_findings_from_a_batch_can_cite_any_hunk_in_it() -> None:
    items = [hunk_item("a.py", added(2, 1)), hunk_item("a.py", added(2, 40))]
    client = FakeLLMClient([findings_reply([finding(1, "a.py"), finding(41, "a.py")])])
    assert [f.line for f in LLMReviewer(client)(items)] == [1, 41]


def test_an_empty_list_of_items_makes_no_calls() -> None:
    from pr_agent.reviewer import group_into_batches

    assert group_into_batches([]) == []
