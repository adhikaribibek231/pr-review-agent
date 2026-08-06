from dotenv import load_dotenv
from github import Auth, Github
from github.GithubException import GithubException
import sys
import os

from pr_agent.diff_parser import DiffHunk, is_noise_file, parse_hunks
from pr_agent.models import Finding
from pr_agent.validator import validate_findings

def main()->int:
    if len(sys.argv)!=3:
        print("Usage: python experiments/run_fake_review_on_pr.py OWNER/REPOSITORY PR_NUMBER", file=sys.stderr)
        return 1
    _= load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN is missing.", file=sys.stderr)
        return 1

    repository_name = sys.argv[1]
    try:
        pr_number = int(sys.argv[2])
    except ValueError:
        print("Error: PR_NUMBER must be an integer",file=sys.stderr)
        return 1

    github = Github(auth = Auth.Token(token))

    try: 
        repository = github.get_repo(repository_name)
        pull_request = repository.get_pull(pr_number)
        print(f"pr #{pull_request.number}: {pull_request.title}")
        print(f"state: {pull_request.state}")
        print(f"base branch: {pull_request.base.ref}")
        print(f"source branch: {pull_request.head.ref}")
        print()
        all_hunks : list[DiffHunk]=[]
        for changed_file in pull_request.get_files():
            if is_noise_file(changed_file.filename):
                continue
            all_hunks.extend(parse_hunks(filename=changed_file.filename, patch=changed_file.patch))
        findings:list[Finding] = [
            Finding(
                filename='main.py',
                line=36,
                severity= "error",
                category= 'type-error',
                message='The annotation does not match the function return type',
                ),
            Finding(
                filename='main.py',
                line=999,
                severity='warning',
                category="bug",
                message="This finding points to a nonexistent changed line",
                ),
            ]
        validated_findings = validate_findings(findings=findings, hunks= all_hunks)
        print("\nValidated findings:")
        for finding in validated_findings:
            print(f"- {finding.filename}: {finding.line} -- {finding.message}")
        assert len(validated_findings)==1
        assert validated_findings[0].line ==36
        print("\nValidation passes.")
    except GithubException as exc:
        print(f"GitHub API error ({exc.status}): {exc.data}",file=sys.stderr)
        return 1
    finally:
        github.close()
    return 0

if __name__=="__main__":
    raise SystemExit(main())
