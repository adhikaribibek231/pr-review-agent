import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

HUNK_HEADER_PATTERN = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)

NOISE_FILENAMES = {
    "package-lock.json",
    "poetry.lock",
    "yarn.lock",
    "pnpm-lock.yaml",
}
@dataclass(frozen=True)
class DiffHunk:
    filename: str
    patch:str
    added_lines: frozenset[int]
    deleted_lines: frozenset[int]
    

def parse_hunk_header(header:str)->tuple[int,int,int,int]:
    match = HUNK_HEADER_PATTERN.match(header)

    if match is None:
        raise ValueError(f"Invalid hunk header: {header!r}")

    old_start = int(match.group("old_start"))
    old_count = int(match.group("old_count") or 1)
    new_start = int(match.group("new_start"))
    new_count = int(match.group("new_count") or 1)

    return old_start, old_count, new_start, new_count

def split_hunks(patch:str)->list[str]:
    hunks : list[list[str]] = []
    current_hunk: list[str]| None = None
    for line in patch.splitlines():
        if line.startswith("@@"):
            current_hunk = [line]
            hunks.append(current_hunk)
            continue

        if current_hunk is not None:
            current_hunk.append(line)
    return ["\n".join(hunk) for hunk in hunks]

def parse_changed_lines(hunk_text:str)-> tuple[frozenset[int],frozenset[int]]:
    lines = hunk_text.splitlines()
    if not lines:
        raise ValueError("Hunk text cannot be empty")
    old_start, _, new_start,_ = parse_hunk_header(lines[0])

    old_line_cursor = old_start
    new_line_cursor = new_start
    added_lines: set[int] = set()
    deleted_lines: set[int] = set()
   
    for line in lines[1:]:
        if line.startswith("\\"):
            continue

        if line.startswith("+"):
            added_lines.add(new_line_cursor)
            new_line_cursor += 1
        elif line.startswith("-"):
            deleted_lines.add(old_line_cursor)
            old_line_cursor+=1
        elif line.startswith(" ") or line=="":
            old_line_cursor+=1
            new_line_cursor += 1
        else:
            raise ValueError(f"Invalid diff body line: {line!r}")

    return frozenset(added_lines), frozenset(deleted_lines)



def parse_hunks(filename:str, patch:str | None)->list[DiffHunk]:
    if patch is None:
        return []
    parsed_hunks: list[DiffHunk]=[]
    for hunk_text in split_hunks(patch):
        added_lines, deleted_lines= parse_changed_lines(hunk_text)
        if not added_lines and not deleted_lines:
            continue
        parsed_hunks.append(
                DiffHunk(
                    filename = filename,
                    patch = hunk_text,
                    added_lines=added_lines,
                    deleted_lines=deleted_lines,
                    )
                )
    logger.info("Parsed %d hunks", len(parsed_hunks))
    return parsed_hunks

def is_noise_file(filename:str)->bool:
    basename = filename.rsplit("/", maxsplit=1)[-1]
    return basename in NOISE_FILENAMES


def changed_code(hunk: DiffHunk) -> list[str]:
    """Source text of the hunk's changed lines, markers removed (added, else deleted)."""
    body = [line for line in hunk.patch.splitlines()[1:] if not line.startswith("\\")]
    added = [line[1:] for line in body if line.startswith("+")]
    return added or [line[1:] for line in body if line.startswith("-")]


def changed_new_lines(hunk: DiffHunk) -> frozenset[int]:
    """New-file line numbers the hunk touches (a pure deletion maps to its position)."""
    if hunk.added_lines:
        return hunk.added_lines
    _, _, new_start, _ = parse_hunk_header(hunk.patch.splitlines()[0])
    return frozenset({new_start})


def deletions(hunk: DiffHunk) -> list[tuple[int, str]]:
    """(new-file line just BEFORE the deletion, deleted text) for each deleted line.

    A deleted line has no new-file number, so "where" means the last surviving line
    above it. Unlike `changed_new_lines`, which for a pure deletion falls back to the
    hunk's first line (up to three lines of context away), this is exact.
    """
    lines = hunk.patch.splitlines()
    _, _, new_cursor, _ = parse_hunk_header(lines[0])
    found: list[tuple[int, str]] = []
    for line in lines[1:]:
        if line.startswith("\\"):
            continue
        if line.startswith("-"):
            found.append((max(new_cursor - 1, 1), line[1:]))
        else:
            new_cursor += 1
    return found
