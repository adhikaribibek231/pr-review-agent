from collections.abc import Callable
from dataclasses import dataclass

from pr_agent.diff_parser import DiffHunk

@dataclass(frozen=True)
class Finding:
    filename: str
    line:int
    severity:str
    category:str
    message:str

@dataclass(frozen=True)
class ChangedFile:
    filename: str
    status: str
    additions: int
    deletions: int
    patch: str | None

@dataclass(frozen=True)
class RetrievedContext:
    filename: str
    start_line:int
    end_line:int
    content: str
    score: float


@dataclass(frozen=True)
class HunkContext:
    """A changed hunk plus the repository code retrieved for it.

    This is what a reviewer receives: the change, and what it touches.
    """

    hunk: DiffHunk
    context: tuple[RetrievedContext, ...]


# Anything that turns hunks-with-context into candidate findings: the hardcoded
# fake, an LLM-backed reviewer, a rule-based linter. Injected into run_review so
# the pipeline and its tests never depend on which one is plugged in.
Reviewer = Callable[[list[HunkContext]], list[Finding]]
