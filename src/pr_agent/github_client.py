from github import Auth, Github
from github.PullRequest import PullRequest

from pr_agent.models import ChangedFile

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
