from pr_agent.models import Finding

def render_inline_comment(finding: Finding)->str:
    severity = finding.severity.value.upper()
    return (
            f"**{severity} -- {finding.category.value}**\n\n"
            f"{finding.message}"
            )
