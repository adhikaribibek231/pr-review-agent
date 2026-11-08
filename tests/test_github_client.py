"""Posting: what happens when GitHub accepts, rejects, or has already seen the review."""

from typing import Any
from unittest.mock import Mock

import pytest
from github import GithubException

from pr_agent.github_client import fetch_commit_files, get_github_client, post_review
from pr_agent.models import Category, Finding, Severity
from pr_agent.renderer import MARKER_PREFIX, render_summary

SHA = "a" * 40


def finding(line: int = 3, severity: Severity = Severity.WARNING) -> Finding:
    return Finding(filename="app.py", line=line, severity=severity, category=Category.BUG, message="Something is wrong.")


def pull_request(existing_bodies: list[str] | None = None) -> Mock:
    pr = Mock()
    pr.head.sha = SHA
    pr.get_reviews.return_value = [Mock(body=b) for b in (existing_bodies or [])]
    pr.create_review.return_value = Mock(html_url="https://github.com/o/r/pull/1#review")
    return pr


def rejection(status: int = 422) -> GithubException:
    return GithubException(status, {"message": "Line could not be resolved"}, {})


def test_a_normal_review_is_one_call_with_a_summary_and_an_inline_comment_per_finding() -> None:
    pr = pull_request()
    result = post_review(pr, [finding(3), finding(7)])

    assert result.mode == "inline" and result.posted and result.url
    pr.create_review.assert_called_once()
    kwargs = pr.create_review.call_args.kwargs
    assert kwargs["event"] == "COMMENT"
    assert [(c["path"], c["line"], c["side"]) for c in kwargs["comments"]] == [("app.py", 3, "RIGHT"), ("app.py", 7, "RIGHT")]
    assert "Found **2** issue(s)" in kwargs["body"]


def test_the_review_carries_a_hidden_marker_for_the_commit_it_covers() -> None:
    pr = pull_request()
    _ = post_review(pr, [finding()])
    assert f"{MARKER_PREFIX} head={SHA} -->" in pr.create_review.call_args.kwargs["body"]


def test_no_findings_posts_nothing() -> None:
    pr = pull_request()
    result = post_review(pr, [])
    assert result.mode == "nothing-to-post" and not result.posted
    pr.create_review.assert_not_called()


def test_a_commit_that_already_has_a_review_is_not_reviewed_twice() -> None:
    pr = pull_request([f"old text {MARKER_PREFIX} head={SHA} -->"])
    result = post_review(pr, [finding()])
    assert result.mode == "skipped-duplicate" and not result.posted
    pr.create_review.assert_not_called()


def test_a_review_of_an_earlier_commit_does_not_block_a_new_one() -> None:
    pr = pull_request([f"{MARKER_PREFIX} head={'b' * 40} -->"])
    assert post_review(pr, [finding()]).mode == "inline"


def test_reviews_without_a_body_are_tolerated() -> None:
    pr = pull_request()
    pr.get_reviews.return_value = [Mock(body=None)]
    assert post_review(pr, [finding()]).mode == "inline"


def test_one_rejected_line_falls_back_to_a_summary_listing_every_finding() -> None:
    pr = pull_request()
    pr.create_review.side_effect = [rejection(422), Mock(html_url="u")]
    result = post_review(pr, [finding(3), finding(99)])

    assert result.mode == "summary-only" and result.posted
    assert pr.create_review.call_count == 2
    second = pr.create_review.call_args.kwargs
    assert "comments" not in second
    assert "`app.py:3`" in second["body"] and "`app.py:99`" in second["body"]
    assert f"{MARKER_PREFIX} head={SHA} -->" in second["body"]


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_errors_other_than_a_rejected_review_are_not_papered_over(status: int) -> None:
    pr = pull_request()
    pr.create_review.side_effect = rejection(status)
    with pytest.raises(GithubException):
        post_review(pr, [finding()])
    assert pr.create_review.call_count == 1


def test_the_review_is_attached_to_the_commit_where_it_lives_for_fork_prs() -> None:
    pr = pull_request()
    commit = object()
    pr.head.repo.get_commit.return_value = commit
    _ = post_review(pr, [finding()])
    pr.head.repo.get_commit.assert_called_once_with(SHA)
    pr.base.repo.get_commit.assert_not_called()  # the old code asked the base repo: wrong for forks
    assert pr.create_review.call_args.kwargs["commit"] is commit


def test_a_deleted_fork_does_not_break_posting() -> None:
    pr = pull_request()
    pr.head.repo = None
    assert post_review(pr, [finding()]).mode == "inline"
    assert "commit" not in pr.create_review.call_args.kwargs


def test_a_commit_lookup_failure_does_not_break_posting() -> None:
    pr = pull_request()
    pr.head.repo.get_commit.side_effect = rejection(404)
    assert post_review(pr, [finding()]).mode == "inline"


# ---- the summary text ------------------------------------------------------------------------------------


def test_summary_counts_findings_by_severity_worst_first() -> None:
    text = render_summary([finding(1, Severity.INFO), finding(2, Severity.CRITICAL), finding(3, Severity.CRITICAL)], SHA)
    assert "Found **3** issue(s): 2 critical, 1 info." in text
    assert f"`{SHA[:8]}`" in text


def test_summary_without_details_points_to_the_inline_comments() -> None:
    assert "inline comments" in render_summary([finding()])
    assert "`app.py:3`" not in render_summary([finding()])


def test_summary_markdown_is_balanced() -> None:
    assert render_summary([finding()], SHA, with_details=True).count("**") % 2 == 0


# ---- reading --------------------------------------------------------------------------------------------


def test_commit_files_come_back_as_changed_files() -> None:
    github: Any = Mock()
    item = Mock(filename="a.py", status="modified", additions=1, deletions=0, patch="@@ -1 +1 @@\n-x\n+y")
    github.get_repo.return_value.get_commit.return_value = Mock(sha="f" * 40, files=[item])
    sha, files = fetch_commit_files(github, "o/r", "ffff")
    assert sha == "f" * 40 and files[0].filename == "a.py" and files[0].patch.startswith("@@")


def test_the_client_can_be_built_with_or_without_a_token() -> None:
    assert get_github_client("ghp_x") is not None
    assert get_github_client(None) is not None
    assert get_github_client("") is not None
