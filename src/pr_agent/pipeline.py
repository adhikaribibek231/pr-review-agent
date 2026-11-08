import logging
from github import Github

from pr_agent.diff_parser import DiffHunk, is_noise_file, parse_hunks
from pr_agent.github_client import fetch_changed_files, fetch_pull_request, post_review
from pr_agent.models import ChangedFile, Finding, HunkContext, Reviewer
from pr_agent.search import CodeIndex, retrieve_for_hunk
from pr_agent.validator import validate_findings

logger = logging.getLogger(__name__)


def gather_context(hunks: list[DiffHunk], index: CodeIndex | None) -> list[HunkContext]:
    """Retrieve context for every hunk; with no index, hunks carry no context."""
    return [
        HunkContext(hunk, tuple(retrieve_for_hunk(index, hunk)) if index is not None else ())
        for hunk in hunks
    ]


def build_hunks(changed_files:list[ChangedFile])->list[DiffHunk]:
    all_hunks:list[DiffHunk]=[]
    for changed_file in changed_files:
        if is_noise_file(changed_file.filename):
            logger.info(f"Skipping noise file: {changed_file.filename}")
            continue

        all_hunks.extend(
                parse_hunks(
                    filename=changed_file.filename,
                    patch=changed_file.patch,
                    )
                )
    return all_hunks

def generate_fake_findings(hunks:list[DiffHunk])->list[Finding]:
    _=hunks #for later
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
    logger.info(f"Generated {len(findings)} findings.")
    return findings

def aggregate_findings(findings: list[Finding])->list[Finding]:
    severity_rank = {
            'error':0,
            'warning':1,
            'info':2
            }
    sorted_findings = sorted(findings, key=lambda x: (x.filename, x.line,severity_rank.get(x.severity,999)))
    return sorted_findings

def triage_findings(findings:list[Finding])->bool:
    return bool(findings)

def fake_reviewer(items: list[HunkContext]) -> list[Finding]:
    return generate_fake_findings([item.hunk for item in items])


def run_review(
    github: Github,
    repository_name: str,
    pr_number: int,
    post_requested: bool,
    reviewer: Reviewer = fake_reviewer,
    index: CodeIndex | None = None,
) -> list[Finding]:
    logger.info(f"Fetching PR #{pr_number}")
    pull_request = fetch_pull_request(github=github, repository_name=repository_name, pr_number=pr_number)
    changed_files = fetch_changed_files(pull_request=pull_request)
    logger.info(f"Fetched {len(changed_files)} changed files")
    hunks = build_hunks(changed_files=changed_files)
    findings = reviewer(gather_context(hunks, index))
    validated_findings = validate_findings(findings = findings, hunks=hunks)
    aggregated_findings = aggregate_findings(findings=validated_findings)
    logger.info(f"Validated {len(aggregated_findings)}/{len(findings)}")
    should_post = triage_findings(aggregated_findings)
    if post_requested and should_post:
        logger.info(f"Posting {len(aggregated_findings)} findings")
        post_review(pull_request=pull_request,findings=aggregated_findings)
        logger.info("Review Posted successfully")
    return aggregated_findings
