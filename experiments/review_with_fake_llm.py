"""The whole review architecture, end to end, with a scripted fake model.

    python experiments/review_with_fake_llm.py

Real:  chunking, embeddings, retrieval, prompt building, parsing, retry, validation.
Fake:  only the model's replies. Swapping in Groq later replaces FakeLLMClient
       with one class; every other line below stays the same.
"""

import logging

from retrieval_eval import PROJECT_ROOT, hunk_for  # also puts src/ on sys.path

from pr_agent.embedding import SentenceTransformerEmbedder
from pr_agent.llm import FakeLLMClient, findings_reply
from pr_agent.models import Category, Finding, Severity
from pr_agent.pipeline import aggregate_findings, gather_context
from pr_agent.reviewer import LLMReviewer
from pr_agent.search import build_index
from pr_agent.validator import validate_findings

logging.basicConfig(level=logging.WARNING, format="  [log] %(message)s")


def banner(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def main() -> None:
    # 1. A "changed hunk": pretend post_review was just added to github_client.py.
    hunk = hunk_for(PROJECT_ROOT, ("src/pr_agent/github_client.py", "post_review"))
    real_line = min(hunk.added_lines) + 2  # a line the diff really added

    # 2. Retrieval: index the repo, fetch context for the hunk.
    index = build_index(PROJECT_ROOT, SentenceTransformerEmbedder(), extra_excluded_dirs={"experiments"})
    items = gather_context([hunk], index)
    banner("retrieved context")
    for chunk in items[0].context:
        print(f"  {chunk.score:.3f}  {chunk.filename}:{chunk.start_line}-{chunk.end_line}")

    # 3. The fake model. Reply 1 is invalid (severity "high" is not in the schema).
    #    Reply 2 is valid but cites one real added line and one that is not in the diff.
    def finding(line: int, message: str) -> Finding:
        return Finding(filename=hunk.filename, line=line, severity=Severity.WARNING,
                       category=Category.BUG, message=message)

    client = FakeLLMClient([
        '{"findings": [{"filename": "x.py", "line": 1, "severity": "high", "category": "bug", "message": "m"}]}',
        findings_reply([finding(real_line, "Real finding on an added line."),
                        finding(9999, "Hallucinated: this line is not in the diff.")]),
    ])
    reviewer = LLMReviewer(client)

    # 4. Review -> validate -> aggregate: the same stages run_review performs.
    candidates = reviewer(items)
    banner(f"model calls made: {len(client.calls)}  (1 invalid reply, 1 retry)")
    banner("what the retry request added to the prompt")
    print(client.calls[1].user[len(client.calls[0].user):])
    validated = validate_findings(candidates, [hunk])
    final = aggregate_findings(validated)
    banner("grounding")
    print(f"  model proposed {len(candidates)}, validator kept {len(final)}")
    for f in final:
        print(f"  KEPT    {f.filename}:{f.line} [{f.severity.value}/{f.category.value}] {f.message}")
    for f in candidates:
        if f not in final:
            print(f"  DROPPED {f.filename}:{f.line} {f.message}")

    banner("first user prompt (what the model was shown)")
    print(client.calls[0].user[:2600])
    banner("system prompt, first 14 lines")
    print("\n".join(client.calls[0].system.splitlines()[:14]))


if __name__ == "__main__":
    main()
