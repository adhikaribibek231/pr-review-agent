from pr_agent.models import Finding

def render_inline_comment(finding: Finding)->str:
    severity = finding.severity.upper()
    return (
            f"**{severity} -- {finding.category}**\n\n"
            f"**{finding.message}"
            )
