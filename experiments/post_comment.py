from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from github import Auth, Github
from github.GithubException import GithubException

from pr_agent.github_client import fetch_changed_files, fetch_pull_request, get_github_client


ISSUE_COMMENT_BODY: str = """
## PR Review Agent

This is a general PR conversation comment.
""".strip()

INLINE_COMMENT_BODY: str = """
This is an inline comment attached to this changed line.
""".strip()

REVIEW_BODY: str = """
## PR Review Agent

This is an overall formal PR review.
""".strip()


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: python experiments/post_comment.py OWNER/REPOSITORY PR_NUMBER",
            file=sys.stderr,
        )
        return 1

    _ = load_dotenv()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN is missing.", file=sys.stderr)
        return 1

    repository_name = sys.argv[1]

    try:
        pr_number = int(sys.argv[2])
    except ValueError:
        print("Error: PR_NUMBER must be an integer.", file=sys.stderr)
        return 1

    github = get_github_client(token=token)

    try:
        pull_request = fetch_pull_request(github=github, repository_name=repository_name,pr_number=pr_number)
        issue_comment = pull_request.create_issue_comment(
            ISSUE_COMMENT_BODY,
        )
        print(f"Issue comment created: {issue_comment.id}")

        commits = pull_request.get_commits()
        latest_commit = commits[pull_request.commits - 1]

        inline_comment = pull_request.create_review_comment(
            body=INLINE_COMMENT_BODY,
            commit=latest_commit,
            path="main.py",
            line=36,
            side="RIGHT",
        )
        print(f"Inline comment created: {inline_comment.id}")

        review = pull_request.create_review(
            body=REVIEW_BODY,
            event="COMMENT",
        )
        print(f"Formal review created: {review.id}")

    except GithubException as exc:
        print(
            f"GitHub API error ({exc.status}): {exc.data}",
            file=sys.stderr,
        )
        return 1
    finally:
        github.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
