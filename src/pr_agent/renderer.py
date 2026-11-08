from pr_agent.models import Finding, Severity

# Hidden in every review body (an HTML comment, invisible when rendered) so the agent can
# recognise its own reviews and not post the same commit twice.
MARKER_PREFIX = "<!-- pr-review-agent:"


def render_inline_comment(finding: Finding) -> str:
    severity = finding.severity.value.upper()
    return (
            f"**{severity} -- {finding.category.value}**\n\n"
            f"{finding.message}"
            )


def render_summary(findings: list[Finding], head_sha: str = "", with_details: bool = False) -> str:
    """The review's top-level body.

    Normally a short count (the details are in the inline comments). With `with_details`,
    used when inline comments could not be posted, every finding is listed here.
    """
    counts = [
        f"{sum(1 for f in findings if f.severity is level)} {level.value}"
        for level in Severity
        if any(f.severity is level for f in findings)
    ]
    reviewed = f" (commit `{head_sha[:8]}`)" if head_sha else ""
    lines = [f"### Automated review{reviewed}", "", f"Found **{len(findings)}** issue(s): {', '.join(counts)}."]
    if with_details:
        lines += ["", "These could not be attached to specific lines, so they are listed here:", ""]
        lines += [
            f"- **{f.severity.value.upper()} / {f.category.value}** `{f.filename}:{f.line}`: {f.message}"
            for f in findings
        ]
    else:
        lines += ["", "Details are in the inline comments."]
    return "\n".join(lines)
