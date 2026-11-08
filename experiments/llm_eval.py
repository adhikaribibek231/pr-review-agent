"""Measure a provider/model on the planted-bug cases, within a hard free-tier budget.

    python experiments/llm_eval.py --dry-run                         # forecast only, no API calls
    python experiments/llm_eval.py --provider groq --model openai/gpt-oss-120b --effort medium

Real: chunking, retrieval, impact analysis, prompts, parsing, grounding, pacing, and the live
model. Only synthetic cases are sent. `--max-tokens` and `--max-calls` are hard caps: the
client refuses to make a call that would pass them.

A finding is a hit if it lands on the bug line (+/- 1). It is a false positive if it is
grounded (on an added line) but elsewhere, or if it appears on a clean case. Findings on
lines that are not in the diff are counted separately as ungrounded: the validator drops them.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planted_bugs import CASES, Case, hunks_of, write_repo  # noqa: E402

from pr_agent.chat_client import BudgetExceededError, estimate_tokens, make_client  # noqa: E402
from pr_agent.embedding import SentenceTransformerEmbedder  # noqa: E402
from pr_agent.env import load_env  # noqa: E402
from pr_agent.models import Finding  # noqa: E402
from pr_agent.pipeline import gather_context  # noqa: E402
from pr_agent.prompts import build_review_prompt  # noqa: E402
from pr_agent.providers import PROVIDERS  # noqa: E402
from pr_agent.reviewer import LLMReviewer  # noqa: E402
from pr_agent.search import build_index  # noqa: E402
from pr_agent.triage import triage_hunks  # noqa: E402
from pr_agent.validator import check_findings  # noqa: E402


def prepare(case: Case, embedder: SentenceTransformerEmbedder):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_repo(root, case.after)
        hunks = hunks_of(case)
        index = build_index(root, embedder)
        reviewable = list(triage_hunks(hunks).reviewable)
        return hunks, gather_context(reviewable, index, all_hunks=hunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--provider", default="groq")
    _ = parser.add_argument("--model", default=None)
    _ = parser.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    _ = parser.add_argument("--reply-tokens", type=int, default=2000, help="max_tokens reserved per reply")
    _ = parser.add_argument("--context-chars", type=int, default=6000)
    _ = parser.add_argument("--max-tokens", type=int, default=45_000, help="hard cap on tokens actually used")
    _ = parser.add_argument("--max-calls", type=int, default=24)
    _ = parser.add_argument("--cases", default="", help="comma-separated case names; default all")
    _ = parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env()
    chosen = [c for c in CASES if not args.cases or c.name in args.cases.split(",")]
    embedder = SentenceTransformerEmbedder()
    prepared = {c.name: prepare(c, embedder) for c in chosen}

    if args.dry_run:
        total = 0
        print(f"{'case':<36}{'hunks':>6}{'~prompt tokens':>16}")
        for case in chosen:
            _, items = prepared[case.name]
            tokens = sum(estimate_tokens(p.system) + estimate_tokens(p.user)
                         for p in (build_review_prompt(i, max_context_chars=args.context_chars) for i in items))
            total += tokens
            print(f"{case.name:<36}{len(items):>6}{tokens:>16}")
        calls = sum(len(prepared[c.name][1]) for c in chosen)
        print(f"\ntotal: {calls} calls, ~{total} prompt tokens (conservative estimate), "
              f"plus up to {args.reply_tokens} reply tokens each = ~{total + calls * 1000} realistic.")
        print("Groq free tier: 8000 tokens/minute, so paced runs take at least "
              f"~{(total + calls * args.reply_tokens) // 7200 + 1} minute(s).")
        return

    spec = PROVIDERS[args.provider]
    env = dict(os.environ)
    env["LLM_PROVIDER"] = args.provider
    if args.model:
        env[spec.model_env] = args.model
    client = make_client(env, reasoning_effort=args.effort, max_tokens=args.reply_tokens,
                         max_calls=args.max_calls, max_total_tokens=args.max_tokens)
    reviewer = LLMReviewer(client, max_context_chars=args.context_chars)  # type: ignore[arg-type]

    tp = fn = fp = ungrounded = 0
    clean_fp = clean_cases = 0
    started = time.perf_counter()
    print(f"provider={args.provider} model={getattr(client, 'model', '?')} effort={args.effort} "
          f"reply_tokens={args.reply_tokens} context_chars={args.context_chars}\n")
    try:
        for case in chosen:
            hunks, items = prepared[case.name]
            before_calls = client.usage.calls  # type: ignore[attr-defined]
            findings: list[Finding] = reviewer(items)
            result = check_findings(findings, hunks)
            bug_line = case.bug_line()
            if case.clean:
                clean_cases += 1
                clean_fp += len(result.kept) > 0
                fp += len(result.kept)
                verdict = "OK (silent)" if not result.kept else f"FALSE POSITIVE x{len(result.kept)}"
            else:
                hit = any(f.filename == case.bug_file and abs(f.line - (bug_line or 0)) <= 1 for f in result.kept)
                tp += hit
                fn += not hit
                extra = [f for f in result.kept if not (f.filename == case.bug_file and abs(f.line - (bug_line or 0)) <= 1)]
                fp += len(extra)
                verdict = ("HIT" if hit else "MISSED") + (f" +{len(extra)} extra" if extra else "")
            ungrounded += len(result.dropped)
            retries = client.usage.calls - before_calls - len(items)  # type: ignore[attr-defined]
            print(f"{case.name:<36}{verdict:<24}calls={client.usage.calls - before_calls}"  # type: ignore[attr-defined]
                  f"{' (retry)' if retries > 0 else ''}  ungrounded={len(result.dropped)}")
            for f in result.kept:
                print(f"      {f.filename}:{f.line} [{f.severity.value}/{f.category.value}] {f.message[:150]}")
            for f in result.dropped:
                print(f"      DROPPED {f.filename}:{f.line} {f.message[:100]}")
    except BudgetExceededError as error:
        print(f"\nSTOPPED by budget cap: {error}")

    bugs = tp + fn
    usage = client.usage  # type: ignore[attr-defined]
    print(f"\nbugs found {tp}/{bugs}   false positives {fp} (clean cases with findings: {clean_fp}/{clean_cases})   "
          f"ungrounded {ungrounded}")
    print(f"calls={usage.calls}  tokens={usage.total_tokens} (prompt {usage.prompt_tokens} + reply {usage.completion_tokens})  "
          f"time={time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
