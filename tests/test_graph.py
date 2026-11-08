import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from pr_agent.checkpointing import memory_checkpointer, sqlite_checkpointer
from pr_agent.graph import (
    AGGREGATE, FETCH_PR, PLAN_HUNKS, POST_REVIEW, RETRIEVE, REVIEW, TRIAGE, VALIDATE,
    build_review_graph, llm_review_step, merge_findings, plain_review_step,
    review_pull_request, thread_id_for,
)
from pr_agent.github_client import PublishResult
from pr_agent.llm import FakeLLMClient, findings_reply
from pr_agent.models import Category, ChangedFile, Finding, RetrievedContext, Severity
from pr_agent.reviewer import LLMReviewer
from pr_agent.search import Retrieval

CODE = "@@ -1,0 +1,3 @@\n+query = f\"SELECT {uid}\"\n+run(query)\n+done = True"  # added lines 1-3
COMMENT_ONLY = "@@ -1,0 +1,1 @@\n+# just a note"
FULL_PATH = [FETCH_PR, PLAN_HUNKS, TRIAGE, RETRIEVE, REVIEW, VALIDATE, AGGREGATE, POST_REVIEW]


def finding(line: int, filename: str = "app.py", message: str = "m") -> Finding:
    return Finding(filename=filename, line=line, severity=Severity.WARNING,
                   category=Category.BUG, message=message)


def changed(filename: str = "app.py", patch: str = CODE) -> ChangedFile:
    return ChangedFile(filename, "modified", 3, 0, patch)


@pytest.fixture
def env() -> Any:
    """GitHub is mocked at the points the graph calls it; the files are set per test."""
    with (
        patch("pr_agent.graph.fetch_pull_request") as fetch_pr,
        patch("pr_agent.graph.fetch_changed_files") as fetch_files,
        patch("pr_agent.graph.post_review_to_github") as post,
    ):
        post.return_value = PublishResult("inline")
        fetch_files.return_value = [changed()]
        yield SimpleNamespace(fetch_pr=fetch_pr, fetch_files=fetch_files, post=post)


def graph_with(client: FakeLLMClient, **options: Any) -> Any:
    return build_review_graph(github=Mock(), review_step=llm_review_step(LLMReviewer(client)), **options)


def run(graph: Any, thread: str = "t") -> tuple[list[str], dict[str, Any]]:
    """Run once; return (nodes in the order they executed, final state)."""
    config = {"configurable": {"thread_id": thread}}
    visited: list[str] = []
    final: dict[str, Any] = {}
    for mode, chunk in graph.stream({"repo": "o/r", "pr_number": 1}, config, stream_mode=["updates", "values"]):
        if mode == "updates":
            visited.extend(chunk.keys())
        else:
            final = chunk
    return visited, final


# --------------------------------------------------------------------------
# the path through the graph
# --------------------------------------------------------------------------


def test_happy_path_visits_every_node_in_order(env: Any) -> None:
    client = FakeLLMClient([findings_reply([finding(2)])])
    visited, final = run(graph_with(client))

    assert visited == FULL_PATH
    assert [f.line for f in final["findings"]] == [2]
    assert len(client.calls) == 1
    assert final["posted"] is False


def test_trivial_only_pr_skips_retrieval_and_review_entirely(env: Any) -> None:
    env.fetch_files.return_value = [changed(patch=COMMENT_ONLY)]
    client = FakeLLMClient()
    visited, final = run(graph_with(client))

    assert visited == [FETCH_PR, PLAN_HUNKS, TRIAGE, AGGREGATE, POST_REVIEW]
    assert client.calls == []  # the whole point: no model call was spent
    assert final["findings"] == []
    assert final["skipped_trivial"] == 1


def test_a_pr_with_no_reviewable_files_also_skips(env: Any) -> None:
    env.fetch_files.return_value = [changed("package-lock.json")]  # noise file
    visited, _ = run(graph_with(FakeLLMClient()))
    assert RETRIEVE not in visited and REVIEW not in visited


def test_triage_keeps_real_hunks_when_mixed_with_trivial_ones(env: Any) -> None:
    env.fetch_files.return_value = [changed("a.py", COMMENT_ONLY), changed("b.py")]
    client = FakeLLMClient()
    _, final = run(graph_with(client))
    assert [h.filename for h in final["reviewable"]] == ["b.py"]
    assert final["skipped_trivial"] == 1
    assert len(client.calls) == 1


# --------------------------------------------------------------------------
# decision 2: validate -> retry
# --------------------------------------------------------------------------


def test_ungrounded_finding_triggers_exactly_one_stricter_retry(env: Any) -> None:
    client = FakeLLMClient([
        findings_reply([finding(1), finding(99, message="hallucinated")]),  # 99 is not in the diff
        findings_reply([finding(1), finding(2)]),
    ])
    visited, final = run(graph_with(client))

    assert visited == [FETCH_PR, PLAN_HUNKS, TRIAGE, RETRIEVE, REVIEW, VALIDATE, REVIEW, VALIDATE, AGGREGATE, POST_REVIEW]
    assert len(client.calls) == 2
    assert "cite ONLY" not in client.calls[0].system
    assert "cite ONLY" in client.calls[1].system  # the retry used the stricter prompt
    # pass-1's grounded finding survives the retry, and re-finding it doesn't duplicate it
    assert [f.line for f in final["findings"]] == [1, 2]
    assert final["attempts"] == 2


def test_the_retry_governor_stops_a_model_that_always_hallucinates(env: Any) -> None:
    client = FakeLLMClient(default=findings_reply([finding(99)]))
    visited, final = run(graph_with(client))

    assert visited.count(REVIEW) == 2  # one pass + one retry, then it moves on
    assert len(client.calls) == 2
    assert final["findings"] == []


@pytest.mark.parametrize(("max_retries", "expected_passes"), [(0, 1), (1, 2), (3, 4)])
def test_max_retries_bounds_the_number_of_review_passes(env: Any, max_retries: int, expected_passes: int) -> None:
    client = FakeLLMClient(default=findings_reply([finding(99)]))
    visited, _ = run(graph_with(client, max_retries=max_retries))
    assert visited.count(REVIEW) == expected_passes


def test_no_retry_when_nothing_was_dropped(env: Any) -> None:
    client = FakeLLMClient([findings_reply([finding(1)])])
    visited, _ = run(graph_with(client))
    assert visited.count(REVIEW) == 1


def test_retry_only_re_reviews_files_where_findings_were_dropped(env: Any) -> None:
    env.fetch_files.return_value = [changed("a.py"), changed("b.py")]
    client = FakeLLMClient([
        findings_reply([finding(1, "a.py")]),       # a.py: fine
        findings_reply([finding(99, "b.py")]),      # b.py: hallucinated line -> dropped
        findings_reply([finding(2, "b.py")]),       # retry, b.py only
    ])
    _, final = run(graph_with(client))

    assert len(client.calls) == 3
    assert 'file="b.py"' in client.calls[2].user and 'file="a.py"' not in client.calls[2].user
    assert sorted((f.filename, f.line) for f in final["findings"]) == [("a.py", 1), ("b.py", 2)]


# --------------------------------------------------------------------------
# aggregate and post
# --------------------------------------------------------------------------


def test_final_findings_are_sorted_and_deduplicated(env: Any) -> None:
    client = FakeLLMClient([findings_reply([finding(3), finding(1), finding(3), finding(2)])])
    _, final = run(graph_with(client))
    assert [f.line for f in final["findings"]] == [1, 2, 3]


def test_posts_once_when_requested_and_there_are_findings(env: Any) -> None:
    client = FakeLLMClient([findings_reply([finding(1)])])
    _, final = run(graph_with(client, post_requested=True))
    env.post.assert_called_once()
    assert [f.line for f in env.post.call_args.args[1]] == [1]
    assert final["posted"] is True


def test_does_not_post_when_not_requested(env: Any) -> None:
    _, final = run(graph_with(FakeLLMClient([findings_reply([finding(1)])]), post_requested=False))
    env.post.assert_not_called()
    assert final["posted"] is False


def test_does_not_post_an_empty_review(env: Any) -> None:
    _, final = run(graph_with(FakeLLMClient(), post_requested=True))
    env.post.assert_not_called()
    assert final["posted"] is False


# --------------------------------------------------------------------------
# retrieval reaches the reviewer
# --------------------------------------------------------------------------


def test_retrieved_context_is_handed_to_the_review_step(env: Any) -> None:
    seen: list[Any] = []
    chunk = RetrievedContext("lib.py", 1, 3, "def helper(): ...", 0.9)

    def step(items: Any, stricter: bool) -> list[Finding]:
        seen.extend(items)
        return []

    with patch("pr_agent.pipeline.retrieve", return_value=Retrieval((chunk,), ())):
        graph = build_review_graph(github=Mock(), review_step=step, index=Mock())
        run(graph)

    assert len(seen) == 1 and seen[0].context == (chunk,)


def test_plain_reviewer_callables_can_be_used_too(env: Any) -> None:
    graph = build_review_graph(github=Mock(), review_step=plain_review_step(lambda items: [finding(1)]))
    _, final = run(graph)
    assert [f.line for f in final["findings"]] == [1]


# --------------------------------------------------------------------------
# the reducer
# --------------------------------------------------------------------------


def test_merge_findings_appends_new_and_skips_known() -> None:
    a, b, c = finding(1), finding(2), finding(3)
    assert merge_findings([a, b], [b, c]) == [a, b, c]


def test_merge_findings_collapses_duplicates_within_a_single_update() -> None:
    a = finding(1)
    assert merge_findings([], [a, finding(2), a]) == [a, finding(2)]


def test_merge_findings_is_idempotent_and_does_not_mutate() -> None:
    existing = [finding(1)]
    once = merge_findings(existing, [finding(2)])
    assert merge_findings(once, [finding(2)]) == once
    assert existing == [finding(1)]


# --------------------------------------------------------------------------
# checkpointing, resume, and idempotency
# --------------------------------------------------------------------------


def test_thread_id_changes_with_the_commit() -> None:
    assert thread_id_for("o/r", 7, "abc") != thread_id_for("o/r", 7, "def")
    assert thread_id_for("o/r", 7, "abc") == thread_id_for("o/r", 7, "abc")


def test_checkpointer_stores_the_state_after_a_run(env: Any) -> None:
    graph = graph_with(FakeLLMClient([findings_reply([finding(1)])]), checkpointer=memory_checkpointer())
    run(graph, thread="t1")
    snapshot = graph.get_state({"configurable": {"thread_id": "t1"}})
    assert [f.line for f in snapshot.values["findings"]] == [1]
    assert snapshot.next == ()
    assert graph.get_state({"configurable": {"thread_id": "other"}}).values == {}


def test_every_state_type_is_registered_for_checkpoint_deserialization(
    env: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """If a new type enters ReviewState and isn't in STATE_TYPES, LangGraph warns
    (and will eventually refuse to load the checkpoint). Exercise every field."""
    client = FakeLLMClient([findings_reply([finding(1), finding(99)]), findings_reply([finding(2)])])
    graph = graph_with(client, checkpointer=memory_checkpointer())
    with caplog.at_level(logging.WARNING):
        run(graph)
        _ = graph.get_state({"configurable": {"thread_id": "t"}})
    assert not [r for r in caplog.records if "unregistered type" in r.getMessage()]


def test_a_crash_mid_run_resumes_without_redoing_finished_nodes(env: Any) -> None:
    attempts = {"n": 0}

    def flaky(items: Any, stricter: bool) -> list[Finding]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("model timed out")
        return [finding(1)]

    graph = build_review_graph(github=Mock(), review_step=flaky, checkpointer=memory_checkpointer())
    with pytest.raises(TimeoutError):
        review_pull_request(graph, "o/r", 1, "t")
    assert env.fetch_files.call_count == 1

    result = review_pull_request(graph, "o/r", 1, "t")  # same thread: resumes at `review`

    assert [f.line for f in result["findings"]] == [1]
    assert env.fetch_files.call_count == 1  # fetch_pr was NOT repeated
    assert attempts["n"] == 2


def test_rerunning_a_finished_thread_returns_the_cached_result(env: Any) -> None:
    client = FakeLLMClient([findings_reply([finding(1)])])
    graph = graph_with(client, post_requested=True, checkpointer=memory_checkpointer())

    first = review_pull_request(graph, "o/r", 1, "same-commit")
    second = review_pull_request(graph, "o/r", 1, "same-commit")

    assert second["findings"] == first["findings"]
    assert len(client.calls) == 1  # no second model call
    env.post.assert_called_once()  # and no duplicate review posted
    assert env.fetch_files.call_count == 1


def test_a_new_commit_gets_a_fresh_thread_and_a_fresh_review(env: Any) -> None:
    client = FakeLLMClient([findings_reply([finding(1)]), findings_reply([finding(2)])])
    graph = graph_with(client, checkpointer=memory_checkpointer())

    old = review_pull_request(graph, "o/r", 1, thread_id_for("o/r", 1, "sha1"))
    new = review_pull_request(graph, "o/r", 1, thread_id_for("o/r", 1, "sha2"))

    assert [f.line for f in old["findings"]] == [1]
    assert [f.line for f in new["findings"]] == [2]  # nothing leaked from the old commit
    assert len(client.calls) == 2


def test_review_pull_request_works_without_a_checkpointer(env: Any) -> None:
    graph = graph_with(FakeLLMClient([findings_reply([finding(1)])]))
    result = review_pull_request(graph, "o/r", 1, "t")
    assert [f.line for f in result["findings"]] == [1]


def test_state_survives_a_process_restart_with_sqlite(env: Any, tmp_path: Any) -> None:
    """Crash, then 'restart': a brand-new graph and connection resume from the file."""
    db = tmp_path / "checkpoints.sqlite"
    calls = {"n": 0}

    def flaky(items: Any, stricter: bool) -> list[Finding]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("boom")
        return [finding(2)]

    with sqlite_checkpointer(db) as saver:
        graph = build_review_graph(github=Mock(), review_step=flaky, checkpointer=saver)
        with pytest.raises(TimeoutError):
            review_pull_request(graph, "o/r", 1, "t")

    with sqlite_checkpointer(db) as saver:  # new connection, new graph object
        graph = build_review_graph(github=Mock(), review_step=flaky, checkpointer=saver)
        result = review_pull_request(graph, "o/r", 1, "t")

    assert [f.line for f in result["findings"]] == [2]
    assert env.fetch_files.call_count == 1  # the fetch happened before the crash, never again


def test_the_post_mode_is_recorded_in_the_state(env: Any) -> None:
    env.post.return_value = PublishResult("summary-only")
    _, final = run(graph_with(FakeLLMClient([findings_reply([finding(1)])]), post_requested=True))
    assert final["posted"] is True and final["post_mode"] == "summary-only"


def test_a_duplicate_review_is_reported_as_not_posted(env: Any) -> None:
    env.post.return_value = PublishResult("skipped-duplicate")
    _, final = run(graph_with(FakeLLMClient([findings_reply([finding(1)])]), post_requested=True))
    assert final["posted"] is False and final["post_mode"] == "skipped-duplicate"


def test_changes_supplied_by_the_caller_skip_fetching(env: Any) -> None:
    """Reviewing a past commit has no pull request: the files come in through the input."""
    graph = graph_with(FakeLLMClient([findings_reply([finding(1)])]))
    result = review_pull_request(graph, "o/r", 0, "commit-abc", supplied={"changed_files": [changed()]})
    assert [f.line for f in result["findings"]] == [1]
    env.fetch_pr.assert_not_called()
    env.fetch_files.assert_not_called()
