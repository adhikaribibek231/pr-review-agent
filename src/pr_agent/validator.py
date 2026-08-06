from pr_agent.diff_parser import DiffHunk
from pr_agent.models import Finding

def validate_findings(findings: list[Finding],hunks:list[DiffHunk])->list[Finding]:
    valid_locations = {(hunk.filename, line) for hunk in hunks for line in hunk.added_lines}
    return [finding for finding in findings if (finding.filename, finding.line) in valid_locations]
