import os
import sys

from dotenv import load_dotenv
from github import Auth, Github, GithubException

from pr_agent.diff_parser import is_noise_file, parse_hunks


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: uv run python experiments/parse_pr.py OWNER/REPO PR_NUMBER"
        )

    repository_name = sys.argv[1]

    try:
        pr_number = int(sys.argv[2])
    except ValueError as exc:
        raise SystemExit("PR number must be an integer") from exc

    load_dotenv()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is not set")

    github = Github(auth=Auth.Token(token))

    try:
        repository = github.get_repo(repository_name)
        pull_request = repository.get_pull(pr_number)

        all_hunks = []

        for changed_file in pull_request.get_files():
            if is_noise_file(changed_file.filename):
                print(f"Skipping noise file: {changed_file.filename}")
                continue

            hunks = parse_hunks(
                filename=changed_file.filename,
                patch=changed_file.patch,
            )

            all_hunks.extend(hunks)

        for index, hunk in enumerate(all_hunks, start=1):
            print("=" * 80)
            print(f"Hunk: {index}")
            print(f"File: {hunk.filename}")
            print(f"Added lines: {sorted(hunk.added_lines)}")
            print(f"Deleted lines: {sorted(hunk.deleted_lines)}")
            print("Patch:")
            print(hunk.patch)

        print("=" * 80)
        print(f"Total parsed hunks: {len(all_hunks)}")

    except GithubException as exc:
        raise SystemExit(
            f"GitHub API error: {exc.status} {exc.data}"
        ) from exc
    finally:
        github.close()


if __name__ == "__main__":
    main()
