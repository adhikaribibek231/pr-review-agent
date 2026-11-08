"""Everything that talks to GitHub.

Reading works anonymously for public repositories (a lower rate limit, no token), which is
what makes a dry run possible without credentials. Writing needs a token that can write to
pull requests.
"""

import logging
from dataclasses import dataclass

from github import Auth, Github, GithubException, GithubRetry
from github.PullRequest import PullRequest, ReviewComment

from pr_agent.models import ChangedFile, Finding
from pr_agent.renderer import MARKER_PREFIX, render_inline_comment, render_summary

logger = logging.getLogger(__name__)


def get_github_client(token: str | None = None) -> Github:
    """A client that backs off and retries on rate limits and transient failures.

    With no token it reads public repositories anonymously (60 requests per hour).
    """
    retry = GithubRetry(total=5, backoff_factor=2)
    return Github(auth=Auth.Token(token), retry=retry) if token else Github(retry=retry)


def fetch_pull_request(github: Github, repository_name: str, pr_number: int) -> PullRequest:
    repository = github.get_repo(repository_name)
    return repository.get_pull(pr_number)


def _changed_file(item: object) -> ChangedFile:
    return ChangedFile(
        filename=getattr(item, "filename"),
        status=getattr(item, "status"),
        additions=getattr(item, "additions"),
        deletions=getattr(item, "deletions"),
        patch=getattr(item, "patch"),
    )


def fetch_changed_files(pull_request: PullRequest) -> list[ChangedFile]:
    return [_changed_file(item) for item in pull_request.get_files()]


def fetch_commit_files(github: Github, repository_name: str, sha: str) -> tuple[str, list[ChangedFile]]:
    """(full SHA, changed files) of one commit: lets the agent review real history read-only."""
    commit = github.get_repo(repository_name).get_commit(sha)
    return commit.sha, [_changed_file(item) for item in commit.files]


# --------------------------------------------------------------------------
# posting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishResult:
    mode: str  # inline | summary-only | skipped-duplicate | nothing-to-post
    url: str | None = None

    @property
    def posted(self) -> bool:
        return self.mode in ("inline", "summary-only")


def _already_reviewed(pull_request: PullRequest, head_sha: str) -> bool:
    """Has this bot already reviewed exactly this commit?

    Every review carries a hidden marker with the commit it covers. Re-running on the same
    commit (a retried Action, a re-run job) must not stack a second review; a new commit
    gets a fresh one.
    """
    marker = f"{MARKER_PREFIX} head={head_sha} -->"
    return any(marker in (review.body or "") for review in pull_request.get_reviews())


def _commit_of(pull_request: PullRequest, head_sha: str) -> object | None:
    """The commit object to attach the review to, found where it actually lives.

    For a fork PR the commit exists in the fork, not in the base repository, so asking the
    base repo (as the first version did) fails. None means "let GitHub use the PR head".
    """
    head_repo = pull_request.head.repo
    if head_repo is None:
        return None
    try:
        return head_repo.get_commit(head_sha)
    except GithubException:
        return None


def post_review(pull_request: PullRequest, findings: list[Finding]) -> PublishResult:
    """Post the findings as one review. Degrades instead of failing, and never duplicates.

      1. nothing to say             -> post nothing
      2. already reviewed this SHA  -> post nothing
      3. normal                     -> one review: a summary plus an inline comment per finding
      4. GitHub rejects the review  -> (422: a line it cannot anchor, a force-push in
                                       between...) one summary review listing every finding,
                                       so a single bad line never costs the whole review
    """
    if not findings:
        return PublishResult("nothing-to-post")
    head_sha = pull_request.head.sha
    if _already_reviewed(pull_request, head_sha):
        logger.info("Commit %s already has a review from this agent; not posting again", head_sha[:8])
        return PublishResult("skipped-duplicate")

    marker = f"{MARKER_PREFIX} head={head_sha} -->"
    commit = _commit_of(pull_request, head_sha)
    options: dict[str, object] = {"commit": commit} if commit is not None else {}
    comments = [
        ReviewComment(path=f.filename, line=f.line, side="RIGHT", body=render_inline_comment(f))
        for f in findings
    ]
    try:
        review = pull_request.create_review(
            body=f"{render_summary(findings, head_sha)}\n\n{marker}", event="COMMENT",
            comments=comments, **options)  # type: ignore[arg-type]
        return PublishResult("inline", getattr(review, "html_url", None))
    except GithubException as error:
        if error.status != 422:
            raise
        logger.warning("GitHub rejected the inline review (%s); posting a summary instead",
                       (error.data or {}).get("message", error.status) if isinstance(error.data, dict) else error.status)

    review = pull_request.create_review(
        body=f"{render_summary(findings, head_sha, with_details=True)}\n\n{marker}", event="COMMENT", **options)  # type: ignore[arg-type]
    return PublishResult("summary-only", getattr(review, "html_url", None))
