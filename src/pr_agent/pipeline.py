
import logging

from github import Github

from pr_agent.diff_parser import DiffHunk, is_noise_file, parse_hunks
from pr_agent.github_client import fetch_changed_files, fetch_pull_request, post_review
from pr_agent.models import ChangedFile, Finding
from pr_agent.validator import validate_findings


def build_hunks(changed_files:list[ChangedFile])->list[DiffHunk]:
    all_hunks:list[DiffHunk]=[]
    for changed_file in changed_files:
        if is_noise_file(changed_file.filename):
            logging.info(f"Skipping noise file: {changed_file.filename}")
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
    logging.info(f"Generated {len(findings)} findings.")
    return findings

def run_review(github:Github, repository_name:str, pr_number:int,post:bool)->list[Finding]:
    logging.info(f"Fetching PR #1")
    pull_request = fetch_pull_request(github=github, repository_name=repository_name, pr_number=pr_number)
    changed_files = fetch_changed_files(pull_request=pull_request)
    logging.info(f"Fetched {len(changed_files)} changed files")
    hunks = build_hunks(changed_files=changed_files)
    findings = generate_fake_findings(hunks=hunks)
    validated_findings = validate_findings(findings = findings, hunks=hunks)
    logging.info(f"Validated {len(validated_findings)}/{len(findings)}")
    if post and validated_findings:
        logging.info(f"Posting {len(validated_findings)} finding")
        post_review(pull_request=pull_request,findings=validated_findings)
        logging.info("Review Posted successfully")
    return validated_findings
