import logging
import re
from dataclasses import dataclass

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
        elif line.startswith(" "):
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
    logging.info(f"Parsed {len(parsed_hunks)} hunks")
    return parsed_hunks

def is_noise_file(filename:str)->bool:
    basename = filename.rsplit("/", maxsplit=1)[-1]
    return basename in NOISE_FILENAMES
