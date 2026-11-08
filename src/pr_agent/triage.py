"""Decide which hunks are worth the expensive path (retrieve -> LLM review).

Pure heuristics, no model call: triage exists to *save* tokens, so it must be
free. A hunk is trivial only when every changed line is something that cannot
introduce a bug. When in doubt a hunk is NOT trivial: skipping a real change
silently is worse than spending a few tokens on a boring one.
"""

import re
from collections import Counter
from dataclasses import dataclass

from pr_agent.diff_parser import DiffHunk

# The WHOLE line must be an import (optionally with a trailing comment). Matching only
# the start would let `import os; os.system(cmd)` through as "just an import" and skip
# review of real code. Parenthesised and multi-line imports deliberately don't match:
# when in doubt, a hunk stays reviewable.
_NAME = r"[\w.]+(?:\s+as\s+\w+)?"
_IMPORT = re.compile(
    rf"^(?:import\s+{_NAME}(?:\s*,\s*{_NAME})*"
    rf"|from\s+[\w.]+\s+import\s+(?:\*|{_NAME}(?:\s*,\s*{_NAME})*))\s*(?:#.*)?$"
)


@dataclass(frozen=True)
class TriageResult:
    reviewable: tuple[DiffHunk, ...]
    skipped: tuple[DiffHunk, ...]


def _changed_lines(hunk: DiffHunk) -> tuple[list[str], list[str]]:
    """(added, deleted) source text, stripped, markers removed."""
    added: list[str] = []
    deleted: list[str] = []
    for line in hunk.patch.splitlines()[1:]:
        if line.startswith("+"):
            added.append(line[1:].strip())
        elif line.startswith("-"):
            deleted.append(line[1:].strip())
    return added, deleted


def _boring(line: str) -> bool:
    """Blank, a comment, or an import: no behaviour of its own."""
    return not line or line.startswith("#") or bool(_IMPORT.match(line))


def is_trivial_hunk(hunk: DiffHunk) -> bool:
    """True if the hunk cannot plausibly change behaviour.

    Two cases:
    * the same non-blank lines were removed and added (re-indentation,
      whitespace, or reordering): nothing new was written;
    * everything touched is blank, a comment, or an import statement.
    """
    added, deleted = _changed_lines(hunk)
    if Counter(line for line in added if line) == Counter(line for line in deleted if line) and (added or deleted):
        return True
    return all(_boring(line) for line in added + deleted)


def triage_hunks(hunks: list[DiffHunk]) -> TriageResult:
    reviewable = tuple(hunk for hunk in hunks if not is_trivial_hunk(hunk))
    skipped = tuple(hunk for hunk in hunks if is_trivial_hunk(hunk))
    return TriageResult(reviewable=reviewable, skipped=skipped)
