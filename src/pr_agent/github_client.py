from github import Auth, Github
from github.PullRequest import PullRequest, ReviewComment
from pr_agent.models import ChangedFile, Finding
from pr_agent.renderer import render_inline_comment

def get_github_client(token:str)->Github:
    return Github(auth=Auth.Token(token))

def fetch_pull_request(github:Github, repository_name:str, pr_number:int)->PullRequest:
    repository = github.get_repo(repository_name)
    return repository.get_pull(pr_number)

def fetch_changed_files(pull_request:PullRequest)->list[ChangedFile]:
    return [
            ChangedFile(
                filename= changed_file.filename,
                status= changed_file.status,
                additions=changed_file.additions,
                deletions=changed_file.deletions,
                patch=changed_file.patch,
                )
            for changed_file in pull_request.get_files()
            ]

def post_review(pull_request:PullRequest, findings: list[Finding])->None:
    if not findings:
        return
    head_commit = pull_request.base.repo.get_commit(pull_request.head.sha)

    
    comments: list[ReviewComment] = [
                ReviewComment(
                    path=finding.filename,
                    line=finding.line,
                    side="RIGHT",
                    body=render_inline_comment(finding),
                )
                for finding in findings
            ]
    _=pull_request.create_review(
                commit=head_commit,
                body=f"Automated review found {len(comments)} issue(s).",
                event="COMMENT",
                comments=comments,
            )

