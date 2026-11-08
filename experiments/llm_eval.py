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
from pr_agent.verify import Verifier, dependents_by_file  # noqa: E402


def prepare(case: Case, embedder: SentenceTransformerEmbedder):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_repo(root, case.after)
        hunks = hunks_of(case)
        index = build_index(root, embedder)
        reviewable = list(triage_hunks(hunks).reviewable)
        return hunks, gather_context(reviewable, index, all_hunks=hunks), index.symbols


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
    _ = parser.add_argument("--verify", action="store_true",
                            help="also run the verification pass; reports results before AND after it")
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
            _, items, _ = prepared[case.name]
            tokens = sum(estimate_tokens(p.system) + estimate_tokens(p.user)
                         for p in (build_review_prompt(i, max_context_chars=args.context_chars) for i in items))
            total += tokens
            print(f"{case.name:<36}{len(items):>6}{tokens:>16}")
        calls = sum(len({i.hunk.filename for i in prepared[c.name][1]}) for c in chosen)
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
    caps = {"max_calls": args.max_calls, "max_total_tokens": args.max_tokens}
    client = make_client(env, reasoning_effort=args.effort, max_tokens=args.reply_tokens, **caps)
    verify_client = make_client(env, reasoning_effort="low", max_tokens=800, **caps) if args.verify else None
    reviewer = LLMReviewer(client, max_context_chars=args.context_chars)  # type: ignore[arg-type]

    def score(case: Case, findings: list[Finding]) -> tuple[bool | None, int]:
        """(hit?, extra findings). hit is None for a clean case."""
        line = case.bug_line()
        if case.clean:
            return None, len(findings)
        near = lambda f: f.filename == case.bug_file and abs(f.line - (line or 0)) <= 1  # noqa: E731
        return any(near(f) for f in findings), sum(1 for f in findings if not near(f))

    totals = {"pre": [0, 0, 0], "post": [0, 0, 0]}  # hits, extras (false positives), misses
    clean_flagged = {"pre": 0, "post": 0}
    clean_cases = ungrounded = 0
    started = time.perf_counter()
    print(f"provider={args.provider} model={getattr(client, 'model', '?')} effort={args.effort} "
          f"verify={args.verify} reply_tokens={args.reply_tokens} context_chars={args.context_chars}\n")
    try:
        for case in chosen:
            hunks, items, symbols = prepared[case.name]
            before = client.usage.calls  # type: ignore[attr-defined]
            result = check_findings(reviewer(items), hunks)
            pre = list(result.kept)
            post, rejected = pre, ()
            if verify_client is not None and pre:
                outcome = Verifier(verify_client, symbols).verify(pre, dependents_by_file(items))  # type: ignore[arg-type]
                post, rejected = list(outcome.kept), outcome.rejected
            ungrounded += len(result.dropped)
            clean_cases += case.clean
            cells = []
            for stage, found in (("pre", pre), ("post", post)):
                hit, extra = score(case, found)
                totals[stage][1] += extra
                if case.clean:
                    clean_flagged[stage] += extra > 0
                else:
                    totals[stage][0] += bool(hit)
                    totals[stage][2] += not hit
                cells.append("OK" if (hit is None and not extra) else "FP x%d" % extra if hit is None
                             else ("HIT" if hit else "MISS") + (f"+{extra}fp" if extra else ""))
            shown = cells[0] if not args.verify else f"{cells[0]} -> {cells[1]}"
            print(f"{case.name:<38}{shown:<24}calls={client.usage.calls - before}  ungrounded={len(result.dropped)}")  # type: ignore[attr-defined]
            for f in pre:
                mark = "REJECTED" if any(f == r[0] for r in rejected) else "kept    "
                print(f"      {mark if args.verify else '        '} {f.filename}:{f.line} [{f.severity.value}] {f.message[:140]}")
            for f, why in rejected:
                print(f"           verifier: {why[:150]}")
    except BudgetExceededError as error:
        print(f"\nSTOPPED by budget cap: {error}")
    except Exception as error:  # noqa: BLE001 - e.g. the provider's daily quota: keep the partial result
        print(f"\nSTOPPED by a provider error: {type(error).__name__}: {str(error)[:200]}")

    stages = ["pre", "post"] if args.verify else ["pre"]
    for stage in stages:
        hits, fps, misses = totals[stage]
        label = "after verification" if stage == "post" else ("before verification" if args.verify else "result")
        print(f"\n{label}: bugs found {hits}/{hits + misses}   false positives {fps} "
              f"(clean cases flagged: {clean_flagged[stage]}/{clean_cases})")
    usage = client.usage  # type: ignore[attr-defined]
    extra_usage = verify_client.usage if verify_client is not None else None  # type: ignore[attr-defined]
    total_tokens = usage.total_tokens + (extra_usage.total_tokens if extra_usage else 0)
    total_calls = usage.calls + (extra_usage.calls if extra_usage else 0)
    print(f"ungrounded (dropped by the validator) {ungrounded}   calls={total_calls}  tokens={total_tokens}  "
          f"time={time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
