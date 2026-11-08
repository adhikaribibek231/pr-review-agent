import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from github import GithubException
from test_search import FakeEmbedder

from pr_agent.chat_client import BudgetExceededError
from pr_agent.cli import Dependencies, main
from pr_agent.llm import FakeLLMClient, findings_reply
from pr_agent.models import Category, Finding, Severity

SHA = "c" * 40
PATCH = "@@ -0,0 +1,3 @@\n+def total(items):\n+    t = sum(items)\n+    return t / len(items)"
FILE = SimpleNamespace(filename="app.py", status="added", additions=3, deletions=0, patch=PATCH)
ENV = {"GROQ_API_KEY": "gsk_test"}


def finding(line: int = 3, severity: Severity = Severity.WARNING) -> Finding:
    return Finding(filename="app.py", line=line, severity=severity, category=Category.BUG, message="Divides by zero on empty input.")


def make_github(files: list[Any] | None = None) -> Mock:
    github = Mock()
    pr = github.get_repo.return_value.get_pull.return_value
    pr.head.sha = SHA
    pr.get_files.return_value = files if files is not None else [FILE]
    pr.get_reviews.return_value = []
    pr.create_review.return_value = Mock(html_url="https://github.com/o/r/pull/1#r")
    github.get_repo.return_value.get_commit.return_value = Mock(sha=SHA, files=files if files is not None else [FILE])
    return github


def snapshot(repo: object, sha: str, tmp: Path) -> Path:
    root = tmp / "repo"
    root.mkdir(exist_ok=True)
    (root / "app.py").write_text("def total(items):\n    t = sum(items)\n    return t / len(items)\n")
    return root


def deps(github: Mock | None = None, client: Any = None, **overrides: Any) -> Dependencies:
    github = github or make_github()
    client = client if client is not None else FakeLLMClient([findings_reply([finding()])])
    return Dependencies(
        github_factory=overrides.pop("github_factory", lambda token: github),
        embedder_factory=lambda: FakeEmbedder(),
        client_factory=overrides.pop("client_factory", lambda env, **options: client),
        snapshot=overrides.pop("snapshot", snapshot),
        **overrides,
    )


def run(argv: list[str], d: Dependencies | None = None, env: dict[str, str] | None = None, verify: bool = False) -> int:
    extra = [] if verify else ["--no-verify"]  # most tests are about everything else: keep them to one model call
    return main(["review", "--repo", "o/r", *argv, *extra], env=ENV if env is None else env, deps=d or deps())


# ---- the normal path -------------------------------------------------------------------------------------------


def test_a_dry_run_prints_the_findings_and_posts_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    github = make_github()
    assert run(["--pr", "1"], deps(github)) == 0
    out = capsys.readouterr().out
    assert "Reviewed PR #1 of o/r at cccccccc" in out
    assert "[WARNING / bug] app.py:3" in out and "Divides by zero" in out
    assert "DRY RUN: nothing was posted" in out
    github.get_repo.return_value.get_pull.return_value.create_review.assert_not_called()


def test_post_publishes_the_review(capsys: pytest.CaptureFixture[str]) -> None:
    github = make_github()
    assert run(["--pr", "1", "--post"], deps(github), env={**ENV, "GITHUB_TOKEN": "ghp_x"}) == 0
    github.get_repo.return_value.get_pull.return_value.create_review.assert_called_once()
    assert "posted: yes (inline)" in capsys.readouterr().out


def test_a_pr_with_nothing_worth_reviewing_costs_no_model_calls(capsys: pytest.CaptureFixture[str]) -> None:
    trivial = SimpleNamespace(filename="app.py", status="modified", additions=1, deletions=0,
                              patch="@@ -1,0 +1,1 @@\n+# a note")
    client = FakeLLMClient()
    assert run(["--pr", "1"], deps(make_github([trivial]), client)) == 0
    assert client.calls == [] and "No findings." in capsys.readouterr().out


def test_the_report_includes_what_the_run_cost(capsys: pytest.CaptureFixture[str]) -> None:
    _ = run(["--pr", "1"])
    assert "usage: 1 model call(s)" in capsys.readouterr().out


def test_verbose_shows_why_each_context_chunk_was_retrieved(capsys: pytest.CaptureFixture[str]) -> None:
    _ = run(["--pr", "1", "-v"])
    assert "context for app.py:" in capsys.readouterr().out


def test_json_output_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--pr", "1", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["head_sha"] == SHA and data["posted"] is False
    assert data["findings"][0]["severity"] == "warning" and data["findings"][0]["line"] == 3


# ---- exit codes for CI -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(("severity", "fail_on", "expected"), [
    (Severity.CRITICAL, "critical", 1), (Severity.WARNING, "critical", 0), (Severity.WARNING, "warning", 1),
    (Severity.INFO, "warning", 0), (Severity.INFO, "info", 1), (Severity.CRITICAL, "none", 0),
])
def test_fail_on_sets_the_exit_code_from_the_worst_finding(severity: Severity, fail_on: str, expected: int) -> None:
    client = FakeLLMClient([findings_reply([finding(3, severity)])])
    assert run(["--pr", "1", "--fail-on", fail_on], deps(client=client)) == expected


# ---- refusing to start --------------------------------------------------------------------------------------------


def test_post_without_a_token_stops_before_touching_the_network(capsys: pytest.CaptureFixture[str]) -> None:
    factory = Mock()
    assert run(["--pr", "1", "--post"], deps(github_factory=factory)) == 2
    assert "GITHUB_TOKEN" in capsys.readouterr().err
    factory.assert_not_called()


def test_a_commit_cannot_be_posted_to(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--commit", SHA, "--post"], env={**ENV, "GITHUB_TOKEN": "ghp_x"}) == 2
    assert "needs a pull request" in capsys.readouterr().err


def test_exactly_one_of_pr_and_commit_is_required() -> None:
    with pytest.raises(SystemExit) as neither:
        run([])
    with pytest.raises(SystemExit) as both:
        run(["--pr", "1", "--commit", "abc"])
    assert neither.value.code == both.value.code == 2


def test_missing_llm_configuration_fails_before_any_network_use(capsys: pytest.CaptureFixture[str]) -> None:
    factory = Mock()
    assert run(["--pr", "1"], deps(github_factory=factory), env={}) == 2
    err = capsys.readouterr().err
    assert "No LLM provider is configured" in err and "GROQ_API_KEY" in err
    factory.assert_not_called()


def test_model_requires_provider(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--pr", "1", "--model", "x"]) == 2
    assert "--model needs --provider" in capsys.readouterr().err


def test_an_unknown_provider_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--pr", "1", "--provider", "nonesuch"]) == 2
    assert "unknown provider" in capsys.readouterr().err


def test_provider_and_model_flags_reach_the_client_factory() -> None:
    seen: dict[str, Any] = {}

    def factory(env: dict[str, str], **options: Any) -> FakeLLMClient:
        seen.update(env=env, options=options)
        return FakeLLMClient()

    _ = run(["--pr", "1", "--provider", "groq", "--model", "openai/gpt-oss-20b", "--effort", "low"],
            deps(client_factory=factory))
    assert seen["env"]["LLM_PROVIDER"] == "groq" and seen["env"]["GROQ_MODEL"] == "openai/gpt-oss-20b"
    assert seen["options"]["reasoning_effort"] == "low"


# ---- GitHub problems ----------------------------------------------------------------------------------------------


def test_a_rejected_token_falls_back_to_anonymous_reading_for_a_dry_run(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    tokens_used: list[str | None] = []
    good = make_github()
    bad = Mock()
    bad.get_repo.side_effect = GithubException(401, {"message": "Bad credentials"}, {})

    def factory(token: str | None) -> Mock:
        tokens_used.append(token)
        return bad if token else good

    with caplog.at_level(logging.WARNING):
        code = run(["--pr", "1"], deps(github_factory=factory), env={**ENV, "GITHUB_TOKEN": "expired"})
    assert code == 0 and tokens_used == ["expired", None]
    assert "reading anonymously" in caplog.text and "app.py:3" in capsys.readouterr().out


def test_a_rejected_token_is_an_error_when_posting(capsys: pytest.CaptureFixture[str]) -> None:
    bad = Mock()
    bad.get_repo.side_effect = GithubException(401, {"message": "Bad credentials"}, {})
    assert run(["--pr", "1", "--post"], deps(github_factory=lambda t: bad), env={**ENV, "GITHUB_TOKEN": "expired"}) == 2
    assert "rejected GITHUB_TOKEN" in capsys.readouterr().err


def test_a_missing_pr_is_reported_plainly(capsys: pytest.CaptureFixture[str]) -> None:
    missing = Mock()
    missing.get_repo.return_value.get_pull.side_effect = GithubException(404, {"message": "Not Found"}, {})
    assert run(["--pr", "99"], deps(github_factory=lambda t: missing)) == 2
    assert "PR #99 was not found" in capsys.readouterr().err


def test_a_github_rate_limit_is_explained(capsys: pytest.CaptureFixture[str]) -> None:
    limited = Mock()
    limited.get_repo.side_effect = GithubException(403, {"message": "API rate limit exceeded"}, {})
    assert run(["--pr", "1"], deps(github_factory=lambda t: limited)) == 2
    assert "rate limit" in capsys.readouterr().err


# ---- what gets indexed ----------------------------------------------------------------------------------------------


def test_a_local_root_is_indexed_instead_of_downloading(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def total(items):\n    return sum(items)\n")
    snap = Mock(side_effect=snapshot)
    assert run(["--pr", "1", "--root", str(tmp_path)], deps(snapshot=snap)) == 0
    snap.assert_not_called()


def test_no_retrieval_skips_both_the_download_and_the_embedding_model() -> None:
    snap, embedder = Mock(), Mock()
    d = deps(snapshot=snap)
    d.embedder_factory = embedder
    assert run(["--pr", "1", "--no-retrieval"], d) == 0
    snap.assert_not_called()
    embedder.assert_not_called()


# ---- reviewing history ----------------------------------------------------------------------------------------------


def test_a_past_commit_can_be_reviewed_read_only(capsys: pytest.CaptureFixture[str]) -> None:
    github = make_github()
    assert run(["--commit", SHA], deps(github)) == 0
    out = capsys.readouterr().out
    assert "commit cccccccc of o/r" in out and "app.py:3" in out
    github.get_repo.return_value.get_pull.assert_not_called()


# ---- failures during the run -----------------------------------------------------------------------------------------------


def test_a_budget_cap_stops_the_run_with_advice(capsys: pytest.CaptureFixture[str]) -> None:
    broken = Mock()
    broken.complete_json.side_effect = BudgetExceededError("groq: token cap of 100 reached")
    assert run(["--pr", "1"], deps(client=broken)) == 2
    err = capsys.readouterr().err
    assert "spending cap" in err and "LLM_MAX_TOKENS_PER_RUN" in err and "--checkpoint" in err


def test_a_provider_failure_is_one_clear_line_not_a_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    broken = Mock()
    broken.complete_json.side_effect = TimeoutError("model timed out")
    assert run(["--pr", "1"], deps(client=broken)) == 2
    err = capsys.readouterr().err
    assert "TimeoutError: model timed out" in err and "Traceback" not in err


def test_verbose_lets_the_traceback_through() -> None:
    broken = Mock()
    broken.complete_json.side_effect = TimeoutError("model timed out")
    with pytest.raises(TimeoutError):
        run(["--pr", "1", "-v"], deps(client=broken))


# ---- resuming -----------------------------------------------------------------------------------------------------------------


def test_rerunning_the_same_commit_with_a_checkpoint_reuses_the_result(tmp_path: Path) -> None:
    client = FakeLLMClient([findings_reply([finding()])])
    github = make_github()
    args = ["--pr", "1", "--post", "--checkpoint", str(tmp_path / "cp.sqlite")]
    env = {**ENV, "GITHUB_TOKEN": "ghp_x"}

    assert run(args, deps(github, client), env=env) == 0
    assert run(args, deps(github, client), env=env) == 0

    assert len(client.calls) == 1  # no second model call
    github.get_repo.return_value.get_pull.return_value.create_review.assert_called_once()  # and no second review


# ---- the verification pass --------------------------------------------------------------------------------------------------------


def verdict(kind: str) -> str:
    return f'{{"verdict": "{kind}", "reason": "the callee creates the directory first"}}'


def test_verification_is_on_by_default_and_drops_a_refuted_finding(capsys: pytest.CaptureFixture[str]) -> None:
    client = FakeLLMClient([findings_reply([finding()]), verdict("refuted")])
    assert run(["--pr", "1"], deps(client=client), verify=True) == 0
    out = capsys.readouterr().out
    assert "No findings." in out and "verification rejected 1 finding(s):" in out
    assert "refuted: the callee creates the directory first" in out
    assert len(client.calls) == 2


def test_a_confirmed_finding_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    client = FakeLLMClient([findings_reply([finding()]), verdict("confirmed")])
    assert run(["--pr", "1"], deps(client=client), verify=True) == 0
    out = capsys.readouterr().out
    assert "app.py:3" in out and "verification rejected" not in out


def test_no_verify_skips_the_second_pass() -> None:
    client = FakeLLMClient([findings_reply([finding()])])
    _ = run(["--pr", "1"], deps(client=client))
    assert len(client.calls) == 1


def test_verification_needs_the_index_so_no_retrieval_means_no_verification() -> None:
    client = FakeLLMClient([findings_reply([finding()])])
    _ = run(["--pr", "1", "--no-retrieval"], deps(client=client), verify=True)
    assert len(client.calls) == 1


def test_the_verifier_gets_its_own_cheap_client() -> None:
    options: list[dict[str, Any]] = []

    def factory(env: dict[str, str], **kw: Any) -> FakeLLMClient:
        options.append(kw)
        return FakeLLMClient([findings_reply([finding()]), verdict("confirmed")])

    _ = run(["--pr", "1"], deps(client_factory=factory), verify=True)
    assert options[0]["reasoning_effort"] == "medium" and options[1] == {"reasoning_effort": "low", "max_tokens": 800}


def test_the_report_counts_the_verification_calls_in_the_usage(capsys: pytest.CaptureFixture[str]) -> None:
    clients = [FakeLLMClient([findings_reply([finding()])]), FakeLLMClient([verdict("confirmed")])]
    _ = run(["--pr", "1"], deps(client_factory=lambda env, **kw: clients.pop(0)), verify=True)
    assert "usage: 2 model call(s)" in capsys.readouterr().out


def test_json_output_lists_what_verification_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    client = FakeLLMClient([findings_reply([finding()]), verdict("refuted")])
    _ = run(["--pr", "1", "--json"], deps(client=client), verify=True)
    data = json.loads(capsys.readouterr().out)
    assert data["findings"] == [] and data["refuted_by_verification"][0].startswith("app.py:3: refuted")
