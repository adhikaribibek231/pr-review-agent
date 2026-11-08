from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from pr_agent.diff_parser import DiffHunk


class _LenientStrEnum(StrEnum):
    """String enum that tolerates the case/whitespace noise LLMs add.

    "Critical", " CRITICAL" and "critical" all become Severity.CRITICAL. That is
    formatting drift, which is safe to absorb. It deliberately does NOT map
    synonyms ("error" -> critical, "high" -> critical): guessing at meaning would
    hide the model drifting off the vocabulary. An unknown word fails validation
    loudly instead.
    """

    @classmethod
    def _missing_(cls, value: object) -> "_LenientStrEnum | None":
        # Python calls this when `cls(value)` finds no exact match.
        # Returning None means "really not a member" -> ValueError -> Pydantic error.
        if isinstance(value, str):
            cleaned = value.strip().lower()
            for member in cls:
                if member.value == cleaned:
                    return member
        return None


class Severity(_LenientStrEnum):
    """How much a finding matters. Declared most severe first: that order is
    what `rank` (and therefore sorting) is built on, so don't reorder casually."""

    CRITICAL = "critical"  # will break, corrupt data, or is exploitable
    WARNING = "warning"  # likely a bug or a real risk; should be looked at
    INFO = "info"  # worth knowing; no action strictly required

    @property
    def rank(self) -> int:
        """0 for the most severe. Sort ascending to put critical findings first."""
        return list(Severity).index(self)


class Category(_LenientStrEnum):
    """What kind of problem a finding describes."""

    BUG = "bug"  # logic error, wrong result, crash
    SECURITY = "security"  # injection, leaked secret, missing auth check
    STYLE = "style"  # readability and formatting
    CONVENTION = "convention"  # violates this repository's own conventions


class Finding(BaseModel):
    """One review comment: where it is and what is wrong.

    This is the shape the LLM must produce, so it is a Pydantic model: untrusted
    text enters here and is either validated into a well-typed object or rejected
    with a precise error. Past this point the rest of the code can rely on
    `severity` being one of three values and `line` being a positive int.
    """

    # frozen:  immutable and hashable, like the dataclass this replaced.
    # forbid:  extra keys are an error, so a model that invents "confidence" or
    #          renames "line" to "line_number" fails instead of silently dropping data.
    # strip:   " src/a.py\n" and "src/a.py" are the same file.
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    # The descriptions below are not just documentation: they are copied into
    # `Finding.model_json_schema()`, which is what we show the LLM.
    filename: str = Field(
        min_length=1,
        description="Repository-relative path of the file, exactly as it appears in the diff.",
    )
    line: int = Field(
        ge=1,  # file lines are 1-indexed; 0 and negatives are never valid
        description="1-based line number in the NEW version of the file. Must be a line the diff added.",
    )
    severity: Severity = Field(description="How serious the problem is.")
    category: Category = Field(description="What kind of problem it is.")
    message: str = Field(
        min_length=1,
        description="What is wrong and why, in one or two sentences. Say how to fix it if you can.",
    )


class ReviewOutput(BaseModel):
    """The complete reply expected from the LLM for one review call.

    Wrapping the list in an object (instead of asking for a bare JSON array)
    matters: JSON mode on most providers requires an object at the top level, and
    `findings` being a *required* field means "I found nothing" must be said
    explicitly as {"findings": []}. A reply that forgets the key fails validation
    rather than being mistaken for a clean review.
    """

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding]


# ChangedFile and RetrievedContext stay plain dataclasses on purpose. They are built
# by our own code from GitHub's API and our own chunker, so there is nothing
# untrusted to validate. Pydantic earns its overhead at trust boundaries (LLM
# output, config files), not between our own functions.
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
    start_line: int
    end_line: int
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
