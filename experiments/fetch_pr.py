
from dotenv import load_dotenv
import os
import sys
from github import Auth, Github
from github.GithubException import GithubException

def main() -> int:
    if len(sys.argv) !=3:
        print("Usage: python experiments/fetch_pr.py OWNER/REPOSITORY PR_NUMBER", file=sys.stderr)
        return 1

    loaded = load_dotenv()
    if not loaded:
        print("Warning: .env file was not found")
    token = os.getenv("GITHUB_TOKEN")

    if not token:
        print("Error: GITHUB_TOKEN is missing.", file=sys.stderr)
        return 1

    auth = Auth.Token(token)

    repository_name = sys.argv[1]
    try:
        pr_number = int(sys.argv[2])
    except ValueError:
        print("Error: PR_NUMBER must be an integer.",file = sys.stderr)
        return 1

    github = Github(auth = auth)
    try:
        repository = github.get_repo(repository_name)
        pull_request = repository.get_pull(pr_number)
        
        print(f"PR #{pull_request.number} : {pull_request.title}")
        print(f"State: {pull_request.state}")
        print(f"Base Branch: {pull_request.base.ref}")
        print(f"Source branch: {pull_request.head.ref}")
        print()

        for changed_file in pull_request.get_files():
            print("="*80)
            print(f"File: {changed_file.filename}")
            print(f"Status: {changed_file.status}")
            print(f"Additions: {changed_file.additions}")
            print(f"Deletions: {changed_file.deletions}")
            print("Patch:")

            print(changed_file.patch)

            print()

    except GithubException as exc:
        print(f"Github API error: status={exc.status},data={exc.data}",file = sys.stderr)
        return 1

    finally:
        github.close()
    return 0

if __name__=="__main__":
    raise SystemExit(main())

