"""Build realistic pull requests for tests: a repo before, a repo after, real diffs."""

import difflib
import textwrap
from dataclasses import dataclass
from pathlib import Path

from pr_agent.diff_parser import DiffHunk, parse_hunks


@dataclass(frozen=True)
class PullRequest:
    root: Path  # the repository as it is AFTER the change (the PR's head)
    hunks: list[DiffHunk]  # every hunk in the PR, in file order

    def hunks_in(self, filename: str) -> list[DiffHunk]:
        return [h for h in self.hunks if h.filename == filename]


def _lines(source: str) -> list[str]:
    return textwrap.dedent(source).lstrip("\n").splitlines()


def write_repo(root: Path, files: dict[str, str]) -> None:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text("\n".join(_lines(source)) + "\n", encoding="utf-8")


def make_pr(root: Path, before: dict[str, str], after: dict[str, str]) -> PullRequest:
    """Write `after` to `root` and diff every file that differs from `before`.

    Files only in `before` are deleted by the PR; files only in `after` are added.
    Hunks come from difflib with 3 lines of context, the same shape git produces.
    """
    write_repo(root, after)
    hunks: list[DiffHunk] = []
    for name in sorted(set(before) | set(after)):
        old = _lines(before[name]) if name in before else []
        new = _lines(after[name]) if name in after else []
        if old == new:
            continue
        diff = list(difflib.unified_diff(old, new, lineterm="", n=3))
        patch = "\n".join(line for line in diff if not line.startswith(("--- ", "+++ ")))
        hunks.extend(parse_hunks(name, patch))
    return PullRequest(root=root, hunks=hunks)
