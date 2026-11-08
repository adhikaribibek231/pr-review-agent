from dataclasses import dataclass

from pr_agent.diff_parser import DiffHunk
from pr_agent.models import Finding


@dataclass(frozen=True)
class ValidationResult:
    """What grounding kept and what it threw away.

    The survivors alone are not enough: the self-correction loop has to know
    *that* something was dropped (to decide whether to retry) and *which*
    findings (to retry only the files that went wrong).
    """

    kept: tuple[Finding, ...]
    dropped: tuple[Finding, ...]

    @property
    def has_unsupported(self) -> bool:
        return bool(self.dropped)


def check_findings(findings: list[Finding], hunks: list[DiffHunk]) -> ValidationResult:
    """Split findings into grounded and ungrounded.

    A finding is grounded only if it points at a (file, line) the diff added.
    This is a deliberate precision-over-recall choice: GitHub also accepts
    comments on unchanged context lines inside a hunk, but we don't, because a
    comment on code the author didn't touch is far more likely to be noise.
    """
    valid_locations = {(hunk.filename, line) for hunk in hunks for line in hunk.added_lines}
    kept: list[Finding] = []
    dropped: list[Finding] = []
    for finding in findings:
        (kept if (finding.filename, finding.line) in valid_locations else dropped).append(finding)
    return ValidationResult(kept=tuple(kept), dropped=tuple(dropped))


def validate_findings(findings: list[Finding], hunks: list[DiffHunk]) -> list[Finding]:
    """Just the grounded findings. Use `check_findings` when you need the drops too."""
    return list(check_findings(findings, hunks).kept)
