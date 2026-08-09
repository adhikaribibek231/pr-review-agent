import os
import sys

from dotenv import load_dotenv
from github.GithubException import GithubException
from pr_agent.github_client import get_github_client
from pr_agent.pipeline import run_review



def main() -> int:
    if len(sys.argv) not in {3, 4}:
        print(
            "Usage: uv run python experiments/run_fake_review_on_pr.py OWNER/REPOSITORY PR_NUMBER [--post]",
            file=sys.stderr,
        )
        return 1

    if len(sys.argv) == 4 and sys.argv[3] != "--post":
        print("Error: optional argument must be --post", file=sys.stderr)
        return 1

    should_post = len(sys.argv) == 4

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
        validated_findings = run_review(github=github,repository_name=repository_name, pr_number=pr_number,post=should_post)
        assert len(validated_findings) ==1
        assert validated_findings[0].line ==36
        print("Review completed.")
        print(f"Validated findings: {len(validated_findings)}")
        if should_post:
            print("Review posted to Github")
        else:
            print("Dry run -- Nothing posted")

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
