import os
import sys

from dotenv import load_dotenv
from github.GithubException import GithubException

from pr_agent.diff_parser import DiffHunk, is_noise_file, parse_hunks
from pr_agent.github_client import fetch_changed_files, fetch_pull_request, get_github_client, post_review
from pr_agent.models import Finding
from pr_agent.validator import validate_findings


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
        pull_request = fetch_pull_request(github=github, repository_name=repository_name, pr_number=pr_number)
        changed_files = fetch_changed_files(pull_request=pull_request)
        print(f"PR #{pull_request.number}: {pull_request.title}")
        print(f"State: {pull_request.state}")
        print(f"Base branch: {pull_request.base.ref}")
        print(f"Source branch: {pull_request.head.ref}")
        print()

        all_hunks: list[DiffHunk] = []
        for changed_file in changed_files:
            if is_noise_file(changed_file.filename):
                print(f"Skipping noise file: {changed_file.filename}")
                continue

            all_hunks.extend(
                parse_hunks(
                    filename=changed_file.filename,
                    patch=changed_file.patch,
                )
            )

        findings: list[Finding] = [
            Finding(
                filename="main.py",
                line=36,
                severity="error",
                category="type-error",
                message="The annotation does not match the function return type.",
            ),
            Finding(
                filename="main.py",
                line=999,
                severity="warning",
                category="bug",
                message="This finding points to a nonexistent changed line.",
            ),
        ]

        validated_findings = validate_findings(
            findings=findings,
            hunks=all_hunks,
        )

        print("Parsed hunks:")
        for hunk in all_hunks:
            print(
                f"- {hunk.filename}: "
                f"added={sorted(hunk.added_lines)}, "
                f"deleted={sorted(hunk.deleted_lines)}"
            )

        print("\nProposed findings:")
        for finding in findings:
            print(
                f"- {finding.filename}:{finding.line} "
                f"[{finding.severity}/{finding.category}] "
                f"{finding.message}"
            )

        print("\nValidated findings:")
        for finding in validated_findings:
            print(
                f"- {finding.filename}:{finding.line} "
                f"[{finding.severity}/{finding.category}] "
                f"{finding.message}"
            )

        assert len(validated_findings) == 1
        assert validated_findings[0].line == 36

        print("\nValidation passed.")

        if should_post and validated_findings:
            post_review(pull_request=pull_request, findings=validated_findings)

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
