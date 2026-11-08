"""Command line interface.

    python -m pr_agent review --repo owner/name --pr 7            # dry run: prints, posts nothing
    python -m pr_agent review --repo owner/name --pr 7 --post     # also posts the review
    python -m pr_agent review --repo owner/name --commit abc123   # review a past commit, read-only

Exit codes: 0 done (and nothing at or above --fail-on), 1 findings at or above --fail-on,
2 the run could not complete (bad arguments, configuration, GitHub, or a budget cap).
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from github import Github, GithubException

from pr_agent.chat_client import BudgetExceededError, make_client
from pr_agent.checkpointing import sqlite_checkpointer
from pr_agent.embedding import Embedder, SentenceTransformerEmbedder
from pr_agent.env import load_env
from pr_agent.github_client import fetch_changed_files, fetch_commit_files, fetch_pull_request, get_github_client
from pr_agent.graph import build_review_graph, llm_review_step, review_pull_request, thread_id_for
from pr_agent.llm import LLMClient, Usage
from pr_agent.models import ChangedFile, Finding, Severity
from pr_agent.providers import PROVIDERS, LLMConfigError, describe, select_providers
from pr_agent.retriever import discover_python_files
from pr_agent.reviewer import LLMReviewer
from pr_agent.search import CodeIndex, build_index
from pr_agent.snapshot import fetch_snapshot
from pr_agent.verify import Verifier

logger = logging.getLogger("pr_agent")

EXIT_OK, EXIT_FINDINGS, EXIT_ERROR = 0, 1, 2


@dataclass
class Dependencies:
    """The outside world, as replaceable parts, so every path can be tested offline."""

    github_factory: Callable[[str | None], Github] = get_github_client
    embedder_factory: Callable[[], Embedder] = SentenceTransformerEmbedder
    client_factory: Callable[..., LLMClient] = make_client
    snapshot: Callable[[Any, str, Path], Path] = fetch_snapshot
    tempdir: Callable[[], AbstractContextManager[str]] = field(default=tempfile.TemporaryDirectory)


class CliError(Exception):
    """A problem to report to the user in one line, without a traceback."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pr-agent", description="Review a GitHub pull request with an LLM.")
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="review a pull request or a past commit")
    review.add_argument("--repo", required=True, metavar="OWNER/NAME")
    target = review.add_mutually_exclusive_group(required=True)
    target.add_argument("--pr", type=int, metavar="N", help="pull request number")
    target.add_argument("--commit", metavar="SHA", help="review one past commit (read-only; cannot --post)")
    review.add_argument("--post", action="store_true", help="post the review to the PR (default: dry run)")
    review.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    review.add_argument("--fail-on", choices=["critical", "warning", "info", "none"], default="none",
                        help="exit 1 if any finding is at least this severe (for CI)")
    review.add_argument("--root", type=Path, metavar="DIR",
                        help="index this local checkout instead of downloading the head commit")
    review.add_argument("--no-retrieval", action="store_true",
                        help="skip indexing: faster and cheaper, but the reviewer sees only the diff")
    review.add_argument("--no-verify", action="store_true",
                        help="skip the verification pass that re-checks each finding against the code it depends on")
    review.add_argument("--provider", help="use only this provider (overrides LLM_PROVIDER)")
    review.add_argument("--model", help="model id for --provider (overrides its *_MODEL variable)")
    review.add_argument("--effort", default="medium", choices=["low", "medium", "high"],
                        help="reasoning effort for providers that support it (default: medium)")
    review.add_argument("--reply-tokens", type=int, default=2000, help="max tokens reserved per model reply")
    review.add_argument("--context-chars", type=int, default=6000, help="retrieved-context budget per hunk")
    review.add_argument("--checkpoint", type=Path, metavar="FILE",
                        help="SQLite file: resume after a crash, and re-running a commit is a no-op")
    review.add_argument("-v", "--verbose", action="store_true")
    return parser


# --------------------------------------------------------------------------
# loading what to review
# --------------------------------------------------------------------------


@dataclass
class Source:
    github: Github
    head_sha: str
    changed_files: list[ChangedFile]
    label: str
    is_pull_request: bool


def _open_source(deps: Dependencies, args: argparse.Namespace, token: str | None) -> Source:
    """Fetch the change. A rejected token falls back to anonymous reading when not posting."""
    github = deps.github_factory(token)
    try:
        return _load(github, args)
    except GithubException as error:
        if error.status == 401:
            if args.post or not token:
                raise CliError("GitHub rejected GITHUB_TOKEN (401 Bad credentials). It is probably expired or "
                               "mistyped: create a new token and put it in .env.") from error
            logger.warning("GitHub rejected GITHUB_TOKEN (401); reading anonymously instead. "
                           "That works for public repositories only.")
            return _load(deps.github_factory(None), args)
        if error.status == 404:
            raise CliError(f"{args.repo} {'PR #' + str(args.pr) if args.pr else 'commit ' + str(args.commit)} was "
                           "not found. For a private repository, GITHUB_TOKEN needs access to it.") from error
        if error.status in (403, 429):
            raise CliError(f"GitHub refused the request ({error.status}): likely a rate limit. Anonymous reading "
                           "allows 60 requests per hour; a valid GITHUB_TOKEN raises that.") from error
        raise


def _load(github: Github, args: argparse.Namespace) -> Source:
    if args.pr is not None:
        pull_request = fetch_pull_request(github, args.repo, args.pr)
        return Source(github, pull_request.head.sha, fetch_changed_files(pull_request),
                      f"PR #{args.pr} of {args.repo}", True)
    sha, files = fetch_commit_files(github, args.repo, args.commit)
    return Source(github, sha, files, f"commit {sha[:8]} of {args.repo}", False)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _at_least(finding: Finding, level: str) -> bool:
    return level != "none" and finding.severity.rank <= Severity(level).rank


def render_report(state: Mapping[str, Any], source: Source, usage: Any, providers: str,
                  posted: bool, post_requested: bool, seconds: float, verbose: bool) -> str:
    findings: list[Finding] = list(state.get("findings", []))
    lines = [
        f"Reviewed {source.label} at {source.head_sha[:8]}",
        f"  hunks: {len(state.get('hunks', []))}   reviewed: {len(state.get('reviewable', []))}   "
        f"skipped as trivial: {state.get('skipped_trivial', 0)}   review passes: {state.get('attempts', 0)}",
        f"  providers: {providers}",
        "",
    ]
    if verbose:
        for item in state.get("items", []):
            lines.append(f"  context for {item.hunk.filename}:")
            lines += [f"    note: {note}" for note in item.notes]
            lines += [f"    {c.filename}:{c.start_line}-{c.end_line}  ({c.reason})" for c in item.context]
        lines.append("")
    if findings:
        lines.append(f"{len(findings)} finding(s):")
        for f in findings:
            lines += ["", f"  [{f.severity.value.upper()} / {f.category.value}] {f.filename}:{f.line}",
                      f"      {f.message}"]
    else:
        lines.append("No findings.")
    refuted: list[str] = list(state.get("refuted", []))
    if refuted:
        lines += ["", f"verification rejected {len(refuted)} finding(s):"] + [f"  {why}" for why in refuted]
    lines += ["", f"usage: {usage.calls} model call(s), {usage.total_tokens} tokens "
                  f"({usage.prompt_tokens} prompt + {usage.completion_tokens} reply), {seconds:.0f}s"]
    if post_requested:
        lines.append(f"posted: {'yes (' + state.get('post_mode', '') + ')' if posted else 'no (' + state.get('post_mode', 'nothing to post') + ')'}")
    else:
        lines.append("DRY RUN: nothing was posted (add --post to post the review).")
    return "\n".join(lines)


def _report_json(state: Mapping[str, Any], source: Source, usage: Any, providers: str, seconds: float) -> str:
    return json.dumps({
        "target": source.label, "head_sha": source.head_sha, "providers": providers,
        "hunks": len(state.get("hunks", [])), "reviewed": len(state.get("reviewable", [])),
        "skipped_trivial": state.get("skipped_trivial", 0), "review_passes": state.get("attempts", 0),
        "findings": [f.model_dump(mode="json") for f in state.get("findings", [])],
        "refuted_by_verification": list(state.get("refuted", [])),
        "posted": state.get("posted", False), "post_mode": state.get("post_mode"),
        "usage": {"calls": usage.calls, "prompt_tokens": usage.prompt_tokens,
                  "completion_tokens": usage.completion_tokens},
        "seconds": round(seconds, 1),
    }, indent=2)


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------


def _client_environment(env: Mapping[str, str], args: argparse.Namespace) -> dict[str, str]:
    merged = dict(env)
    if args.model and not args.provider:
        raise CliError("--model needs --provider, to say which provider the model belongs to.")
    if args.provider:
        if args.provider not in PROVIDERS:
            raise CliError(f"unknown provider {args.provider!r}. Known: {', '.join(PROVIDERS)}.")
        merged["LLM_PROVIDER"] = args.provider
        if args.model:
            merged[PROVIDERS[args.provider].model_env] = args.model
    return merged


def run_review(args: argparse.Namespace, env: Mapping[str, str], deps: Dependencies) -> int:
    if args.post and args.commit:
        raise CliError("--post needs a pull request (--pr): a commit has nowhere to post a review.")
    token = (env.get("GITHUB_TOKEN") or "").strip() or None
    if args.post and not token:
        raise CliError("--post needs GITHUB_TOKEN with permission to write to pull requests. Set it in .env.")

    # Validate the LLM configuration BEFORE any network use, so a typo fails in a second.
    llm_env = _client_environment(env, args)
    providers = describe(select_providers(llm_env))

    started = time.perf_counter()
    source = _open_source(deps, args, token)
    logger.info("Fetched %s: %d changed file(s)", source.label, len(source.changed_files))

    index: CodeIndex | None = None
    with deps.tempdir() as tmp:
        if not args.no_retrieval:
            if args.root:
                root = args.root
            else:
                repository = source.github.get_repo(args.repo)
                root = deps.snapshot(repository, source.head_sha, Path(tmp))
            index = build_index(root, deps.embedder_factory())
            logger.info("Indexed %d chunks from %d files", len(index.chunks), len({c.filename for c in index.chunks}))

        client = deps.client_factory(llm_env, reasoning_effort=args.effort, max_tokens=args.reply_tokens)
        reviewer = LLMReviewer(client, max_context_chars=args.context_chars)
        # The verifier re-reads the code each finding depends on, so it needs the index. Its replies
        # are one short verdict, so it reserves little and thinks little.
        verify_client = None
        verifier = None
        if index is not None and not args.no_verify:
            verify_client = deps.client_factory(llm_env, reasoning_effort="low", max_tokens=800)
            verifier = Verifier(verify_client, index.symbols)
        checkpoint: AbstractContextManager[Any] = (
            sqlite_checkpointer(args.checkpoint) if args.checkpoint else nullcontext(None))
        with checkpoint as saver:
            graph = build_review_graph(github=source.github, review_step=llm_review_step(reviewer), index=index,
                                       post_requested=args.post, checkpointer=saver, verifier=verifier)
            if source.is_pull_request:
                state = review_pull_request(graph, args.repo, args.pr, thread_id_for(args.repo, args.pr, source.head_sha))
            else:
                state = review_pull_request(graph, args.repo, 0, f"{args.repo}@commit-{source.head_sha}",
                                            supplied={"changed_files": source.changed_files})

    seconds = time.perf_counter() - started
    usage = Usage()
    for used in (getattr(c, "usage", None) for c in (client, verify_client) if c is not None):
        if used is not None and used is not usage:
            usage.calls += used.calls
            usage.prompt_tokens += used.prompt_tokens
            usage.completion_tokens += used.completion_tokens
    if args.json:
        print(_report_json(state, source, usage, providers, seconds))
    else:
        print(render_report(state, source, usage, providers, bool(state.get("posted")), args.post, seconds, args.verbose))
    findings: list[Finding] = list(state.get("findings", []))
    return EXIT_FINDINGS if any(_at_least(f, args.fail_on) for f in findings) else EXIT_OK


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None,
         deps: Dependencies | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if env is None:
        load_env()
        env = os.environ
    try:
        return run_review(args, env, deps or Dependencies())
    except (CliError, LLMConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    except BudgetExceededError as error:
        print(f"stopped by a spending cap: {error}. Raise LLM_MAX_CALLS_PER_RUN / LLM_MAX_TOKENS_PER_RUN in .env "
              "if this is expected, and use --checkpoint to resume without repeating finished work.", file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:  # noqa: BLE001 - a provider or network failure: one clear line, not a traceback
        if args.verbose:
            raise
        print(f"error: {type(error).__name__}: {error}\n(run with -v for the full traceback)", file=sys.stderr)
        return EXIT_ERROR
