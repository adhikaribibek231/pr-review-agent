"""The review agent as a LangGraph state machine.

    START -> fetch_pr -> plan_hunks -> triage --(nothing worth reviewing)--------+
                                          |                                      |
                                          v                                      |
                                       retrieve -> review -> validate --(done)--> aggregate -> post_review -> END
                                                      ^          |
                                                      +-(retry)--+   (at most `max_retries` times)

Every node is a thin wrapper over a function that already existed and was already
tested (build_hunks, triage_hunks, gather_context, check_findings, ...). The graph
adds no review logic; it adds *state, routing and checkpointing*.

Rules every node follows (they are what make resuming safe):
  * It receives the whole state and returns ONLY the keys it changes.
  * State holds plain data. Live objects (the GitHub client, the PullRequest,
    the embedding model) stay out of it: after a crash a new process has none of
    them, so nodes that need them get them from the closure or re-fetch.
"""

from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from github import Github
from langgraph.graph import END, START, StateGraph

from pr_agent.diff_parser import DiffHunk
from pr_agent.github_client import fetch_changed_files, fetch_pull_request
from pr_agent.github_client import post_review as post_review_to_github
from pr_agent.models import ChangedFile, Finding, HunkContext, Reviewer
from pr_agent.pipeline import aggregate_findings, build_hunks, gather_context, should_post
from pr_agent.reviewer import LLMReviewer
from pr_agent.search import CodeIndex
from pr_agent.triage import triage_hunks
from pr_agent.validator import check_findings

DEFAULT_MAX_RETRIES = 1

# Node names, as constants so the wiring and the tests can't drift apart by typo.
FETCH_PR, PLAN_HUNKS, TRIAGE, RETRIEVE = "fetch_pr", "plan_hunks", "triage", "retrieve"
REVIEW, VALIDATE, AGGREGATE, POST_REVIEW = "review", "validate", "aggregate", "post_review"

# What a review pass looks like to the graph: (hunks with context, stricter?) -> findings.
ReviewStep = Callable[[list[HunkContext], bool], list[Finding]]


def llm_review_step(reviewer: LLMReviewer) -> ReviewStep:
    """Adapt an LLMReviewer so the retry edge can ask for the stricter prompt."""
    return lambda items, stricter: reviewer.review(items, stricter=stricter)


def plain_review_step(reviewer: Reviewer) -> ReviewStep:
    """Adapt any plain `Reviewer` callable; it has no stricter mode, so that is ignored."""
    return lambda items, _stricter: reviewer(items)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


def merge_findings(existing: list[Finding], new: list[Finding]) -> list[Finding]:
    """Reducer for `validated`: append new findings, skipping ones already present.

    A reducer answers "old value + a node's update = ?". The default is
    "= update" (overwrite). `operator.add` would be "= old + update", which
    duplicates every finding the retry pass re-discovers. This one is idempotent:
    merging the same finding twice is a no-op (Finding is frozen, so hashable and
    comparable by value).
    """
    merged = list(existing)
    for finding in new:
        if finding not in merged:  # checks against `merged`, so duplicates *within* `new` collapse too
            merged.append(finding)
    return merged


class ReviewState(TypedDict, total=False):
    """Everything the agent knows. `total=False`: keys appear as nodes fill them in."""

    # input
    repo: str
    pr_number: int
    # fetch_pr / plan_hunks / triage
    changed_files: list[ChangedFile]
    hunks: list[DiffHunk]  # every reviewable-file hunk (noise files already dropped)
    reviewable: list[DiffHunk]  # the hunks that survived triage
    skipped_trivial: int
    # retrieve
    items: list[HunkContext]  # each reviewable hunk with its retrieved context
    # review / validate loop
    attempts: int  # review passes run so far; the retry governor reads this
    candidates: list[Finding]  # the latest review pass's raw output (overwritten each pass)
    dropped: list[Finding]  # the latest validation's rejects (overwritten each pass)
    validated: Annotated[list[Finding], merge_findings]  # grounded findings, accumulated
    # aggregate / post_review
    findings: list[Finding]  # final: deduplicated and sorted. A separate field because a
    # reducer-managed field is append-only and could not be re-ordered.
    posted: bool
    post_mode: str  # inline | summary-only | skipped-duplicate | nothing-to-post


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------


def build_review_graph(
    *,
    github: Github,
    review_step: ReviewStep,
    index: CodeIndex | None = None,
    post_requested: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    checkpointer: Any = None,
) -> Any:
    """Wire the nodes and edges and compile. Dependencies come in as arguments.

    Nothing here is global, so a test can build a graph around a fake model and a
    mocked GitHub, and production builds the same graph around real ones.
    """

    # ---- nodes -----------------------------------------------------------

    def fetch_pr(state: ReviewState) -> dict[str, Any]:
        if "changed_files" in state:
            return {}  # supplied by the caller, e.g. a review of a past commit: nothing to fetch
        pull_request = fetch_pull_request(github, state["repo"], state["pr_number"])
        return {"changed_files": fetch_changed_files(pull_request)}

    def plan_hunks(state: ReviewState) -> dict[str, Any]:
        return {"hunks": build_hunks(state["changed_files"])}

    def triage(state: ReviewState) -> dict[str, Any]:
        result = triage_hunks(state["hunks"])
        return {"reviewable": list(result.reviewable), "skipped_trivial": len(result.skipped)}

    def retrieve(state: ReviewState) -> dict[str, Any]:
        return {"items": gather_context(state["reviewable"], index, all_hunks=state["hunks"])}

    def review(state: ReviewState) -> dict[str, Any]:
        attempts = state.get("attempts", 0)
        items = state["items"]
        if attempts > 0:
            # A retry: only re-review files where validation rejected something.
            # Re-reviewing everything would double the token cost for no reason.
            bad_files = {finding.filename for finding in state.get("dropped", [])}
            items = [item for item in items if item.hunk.filename in bad_files]
        return {
            "candidates": review_step(items, attempts > 0),
            "attempts": attempts + 1,  # counted HERE, so the router sees the pass that just ran
        }

    def validate(state: ReviewState) -> dict[str, Any]:
        result = check_findings(state["candidates"], state["hunks"])
        return {"validated": list(result.kept), "dropped": list(result.dropped)}

    def aggregate(state: ReviewState) -> dict[str, Any]:
        return {"findings": aggregate_findings(state.get("validated", []))}

    def post_review(state: ReviewState) -> dict[str, Any]:
        findings = state.get("findings", [])
        if not (post_requested and should_post(findings)):
            return {"posted": False}
        # Re-fetched rather than kept in state: a PullRequest object can't be
        # checkpointed, and a resumed process wouldn't have it anyway.
        pull_request = fetch_pull_request(github, state["repo"], state["pr_number"])
        result = post_review_to_github(pull_request, findings)
        return {"posted": result.posted, "post_mode": result.mode}

    # ---- the two decisions -----------------------------------------------

    def route_after_triage(state: ReviewState) -> str:
        return "review_path" if state["reviewable"] else "skip"

    def route_after_validate(state: ReviewState) -> str:
        # `attempts` already includes the pass that just ran. With max_retries=1:
        # after pass 1, 1 <= 1 -> retry; after pass 2, 2 <= 1 is False -> move on.
        # Without this governor a model that always hallucinates would loop forever.
        if state["dropped"] and state["attempts"] <= max_retries:
            return "retry"
        return "done"

    # ---- wiring ----------------------------------------------------------

    graph = StateGraph(ReviewState)
    for name, node in [
        (FETCH_PR, fetch_pr), (PLAN_HUNKS, plan_hunks), (TRIAGE, triage), (RETRIEVE, retrieve),
        (REVIEW, review), (VALIDATE, validate), (AGGREGATE, aggregate), (POST_REVIEW, post_review),
    ]:
        graph.add_node(name, node)

    graph.add_edge(START, FETCH_PR)
    graph.add_edge(FETCH_PR, PLAN_HUNKS)
    graph.add_edge(PLAN_HUNKS, TRIAGE)
    graph.add_conditional_edges(TRIAGE, route_after_triage, {"review_path": RETRIEVE, "skip": AGGREGATE})
    graph.add_edge(RETRIEVE, REVIEW)
    graph.add_edge(REVIEW, VALIDATE)
    graph.add_conditional_edges(VALIDATE, route_after_validate, {"retry": REVIEW, "done": AGGREGATE})
    graph.add_edge(AGGREGATE, POST_REVIEW)
    graph.add_edge(POST_REVIEW, END)
    return graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------


def thread_id_for(repo: str, pr_number: int, head_sha: str) -> str:
    """Checkpoint key: one thread per PR *at one commit*.

    Keying on the PR number alone is a trap. A thread remembers state, so a run
    after a new push would start from the previous push's findings. Including the
    head SHA gives each commit a fresh thread, while re-running the same commit
    (a retry, a crashed run) maps to the same thread.
    """
    return f"{repo}#{pr_number}@{head_sha}"


def review_pull_request(graph: Any, repo: str, pr_number: int, thread_id: str,
                        supplied: dict[str, Any] | None = None) -> ReviewState:
    """Run the graph for one PR, honouring what the checkpointer already knows.

    The thread's status decides what happens:

      finished   -> return the saved result untouched. No re-review, no second
                    post: this is what makes re-running the same commit safe.
      interrupted-> resume from the node that failed; finished nodes don't re-run.
      new        -> run from the start.

    Calling `graph.invoke(...)` directly on a *finished* thread is a trap: it
    re-runs every node on top of the stale state (e.g. `attempts` left over from
    the previous run), so the retry logic misfires.
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial = {"repo": repo, "pr_number": pr_number, **(supplied or {})}
    if graph.checkpointer is None:
        return graph.invoke(initial, config)

    snapshot = graph.get_state(config)
    if snapshot.values and not snapshot.next:
        return snapshot.values  # finished: cached
    if snapshot.next:
        return graph.invoke(None, config)  # interrupted: resume
    return graph.invoke(initial, config)
