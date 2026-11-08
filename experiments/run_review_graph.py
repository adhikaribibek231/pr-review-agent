"""The LangGraph agent end to end, offline.

    python experiments/run_review_graph.py

Real:  retrieval over this repo, prompts, validation, routing, SQLite checkpoints.
Fake:  GitHub (patched) and the model (scripted: its first reply hallucinates a line).
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from retrieval_eval import PROJECT_ROOT, hunk_for  # also puts src/ on sys.path

from pr_agent.checkpointing import sqlite_checkpointer
from pr_agent.embedding import SentenceTransformerEmbedder
from pr_agent.graph import build_review_graph, llm_review_step, review_pull_request, thread_id_for
from pr_agent.llm import FakeLLMClient, findings_reply
from pr_agent.models import Category, ChangedFile, Finding, Severity
from pr_agent.reviewer import LLMReviewer
from pr_agent.search import build_index


def banner(title: str) -> None:
    print(f"\n{'=' * 6} {title} {'=' * 6}")


def main() -> None:
    hunk = hunk_for(PROJECT_ROOT, ("src/pr_agent/github_client.py", "post_review"))
    real_line = min(hunk.added_lines) + 2
    pr_files = [
        ChangedFile(hunk.filename, "modified", len(hunk.added_lines), 0, hunk.patch),
        ChangedFile("notes.py", "modified", 1, 0, "@@ -1,0 +1,1 @@\n+# reworded a comment"),  # trivial
        ChangedFile("poetry.lock", "modified", 9, 9, "@@ -1,1 +1,1 @@\n-a\n+b"),  # noise
    ]

    def finding(line: int, message: str) -> Finding:
        return Finding(filename=hunk.filename, line=line, severity=Severity.WARNING,
                       category=Category.BUG, message=message)

    client_pass1 = [finding(real_line, "Real finding."), finding(9999, "Hallucinated line.")]
    client = FakeLLMClient([
        findings_reply(client_pass1),
        findings_reply([finding(real_line, "Real finding."), finding(real_line + 1, "Found on retry.")]),
    ])

    index = build_index(PROJECT_ROOT, SentenceTransformerEmbedder(), extra_excluded_dirs={"experiments"})
    thread = thread_id_for("owner/repo", 7, "abc123")

    with (
        patch("pr_agent.graph.fetch_pull_request", return_value=Mock()),
        patch("pr_agent.graph.fetch_changed_files", return_value=pr_files) as fetch_files,
        patch("pr_agent.graph.post_review_to_github") as post,
        tempfile.TemporaryDirectory() as tmp,
    ):
        db = Path(tmp) / "checkpoints.sqlite"

        banner("run 1: watch the nodes execute")
        with sqlite_checkpointer(db) as saver:
            graph = build_review_graph(
                github=Mock(), index=index, post_requested=True, checkpointer=saver,
                review_step=llm_review_step(LLMReviewer(client)),
            )
            config: Any = {"configurable": {"thread_id": thread}}
            for update in graph.stream({"repo": "owner/repo", "pr_number": 7}, config, stream_mode="updates"):
                for node, delta in update.items():
                    keys = ", ".join(delta) if delta else "-"
                    print(f"  {node:<12} wrote: {keys}")
            final = graph.get_state(config).values
        print(f"\n  hunks={len(final['hunks'])} (noise file dropped)  reviewable={len(final['reviewable'])}  "
              f"skipped_trivial={final['skipped_trivial']}")
        print(f"  review passes={final['attempts']}  model calls={len(client.calls)}  "
              f"(retry prompt was stricter: {'cite ONLY' in client.calls[1].system})")
        # `dropped` is overwritten on every validation, so it describes the LAST pass only.
        print(f"  pass 1 proposed lines {[f.line for f in client_pass1]}; the validator dropped 9999, "
              f"the retry fixed it (last validation dropped: {[f.line for f in final['dropped']] or 'nothing'})")
        print(f"  final findings: {[(f.line, f.message) for f in final['findings']]}")
        print(f"  posted to GitHub: {final['posted']} ({post.call_count} call)")

        banner("run 2: same commit again -> cached, nothing re-done")
        with sqlite_checkpointer(db) as saver:   # a fresh connection, as after a restart
            graph = build_review_graph(github=Mock(), index=index, post_requested=True, checkpointer=saver,
                                       review_step=llm_review_step(LLMReviewer(client)))
            again = review_pull_request(graph, "owner/repo", 7, thread)
        print(f"  model calls still {len(client.calls)}, GitHub posts still {post.call_count}, "
              f"fetches still {fetch_files.call_count}; same findings: {again['findings'] == final['findings']}")

        banner("run 3: crash in `review`, 'restart', resume")
        crash_thread = thread_id_for("owner/repo", 8, "def456")
        boom = {"armed": True}

        def flaky(items: Any, stricter: bool) -> list[Finding]:
            if boom["armed"]:
                boom["armed"] = False
                raise TimeoutError("model timed out")
            return [finding(real_line, "Survived the crash.")]

        fetches_before = fetch_files.call_count
        with sqlite_checkpointer(db) as saver:
            graph = build_review_graph(github=Mock(), index=index, review_step=flaky, checkpointer=saver)
            try:
                review_pull_request(graph, "owner/repo", 8, crash_thread)
            except TimeoutError as error:
                snap = graph.get_state({"configurable": {"thread_id": crash_thread}})
                print(f"  crashed: {error!r}; saved state says next node = {snap.next}")
        with sqlite_checkpointer(db) as saver:   # new process, new graph, same file
            graph = build_review_graph(github=Mock(), index=index, review_step=flaky, checkpointer=saver)
            resumed = review_pull_request(graph, "owner/repo", 8, crash_thread)
        print(f"  resumed: {[(f.line, f.message) for f in resumed['findings']]}")
        print(f"  GitHub fetches across crash+resume: {fetch_files.call_count - fetches_before} "
              f"(fetch_pr/plan/triage/retrieve were not repeated)")


if __name__ == "__main__":
    main()
