import ast
import logging
from collections.abc import Callable, Collection
from dataclasses import replace
from pathlib import Path
import tokenize
from pr_agent.models import RetrievedContext

logger = logging.getLogger(__name__)

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "site-packages",
    "node_modules",
}

MAX_CHUNK_LINES = 40
CHUNK_OVERLAP_LINES = 10

def _start_line(node: ast.stmt) -> int:
    """Line where the chunk should begin, including any decorators.

    node.lineno points at `def` or `class`, so decorators above it are
    excluded unless we look them up explicitly.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if node.decorator_list:
            return min(decorator.lineno for decorator in node.decorator_list)
    return node.lineno

def _end_line(node: ast.stmt)-> int:
    return node.end_lineno if node.end_lineno is not None else node.lineno


def _make_context(filename: str, source_lines: list[str], start_line: int, end_line:int)->list[RetrievedContext]:
    """Build one context, or several windows if the span is too long"""
    span = end_line-start_line+1
    if span<= MAX_CHUNK_LINES:
        content = "\n".join(source_lines[start_line -1:end_line]) #ast line numbers are 1-indexed and source_lines is 0-indexed. line 5 sits at index 4
        if not content.strip(): #drop whitespace only chunks
            return []
        return [
                RetrievedContext(
                    filename=filename,
                    start_line=start_line,
                    end_line=end_line,
                    content = content,
                    score=0.0,
                    )
                ]
    contexts: list[RetrievedContext] = []
    step = MAX_CHUNK_LINES - CHUNK_OVERLAP_LINES
    cursor = start_line -1 #0-indexed
    while cursor <=end_line:
        window_end = min(cursor + MAX_CHUNK_LINES, end_line)
        content = "\n".join(source_lines[cursor:window_end])
        if content.strip():
            contexts.append(
                    RetrievedContext(
                        filename=filename,
                        start_line=cursor+1, #back to 1-indexed
                        end_line=window_end,
                        content = content,
                        score=0.0,
                        )
                    )
        if window_end == end_line:
            break
        cursor +=step #window 1 is 1-100 lines. window 2 starts from 91
    return contexts


def _chunk_class(filename: str, source_lines: list[str], node:ast.ClassDef)-> list[RetrievedContext]:
    """Whole class if it fits, otherwise one chunk per definition plus the gaps.

    Never both: emitting the class and its methods would double-index the same
    lines and let one method win two retrieval slots.

    Gap segments (attributes, TYPE_CHECKING blocks, anything between
    definitions) are flushed before the next definition and after the last one,
    so no line inside the class is dropped.
    """
    class_start = _start_line(node)
    class_end = _end_line(node)

    if class_end - class_start + 1 <= MAX_CHUNK_LINES:
        return _make_context(filename, source_lines, class_start, class_end)

    contexts: list[RetrievedContext] = []
    
    #header: decorators, class line, docstring - up to the first body statement
    cursor = class_start
    for child in node.body:
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  #picked up by the gap flush before the next definition
        child_start = _start_line(child)
        if child_start > cursor:
            #gap since the last definition: docstring, attributes, TYPE_CHECKING
            contexts.extend(_make_context(filename, source_lines, cursor, child_start -1))

        contexts.extend(_make_context(filename,source_lines, child_start, _end_line(child)))
        cursor = _end_line(child)+1

    if cursor <=class_end:
        #trailing attributes after the last definition
        contexts.extend(_make_context(filename, source_lines, cursor, class_end))

    return contexts

def discover_python_files(
    repository_root: Path, excluded_dirs: Collection[str] = EXCLUDED_DIRS
) -> list[Path]:
    
    files:list[Path] = []

    for path in repository_root.rglob("*.py"):
        relative_parts = path.relative_to(repository_root).parts
        if any(part in excluded_dirs for part in relative_parts):
            continue
        if path.is_file():
            files.append(path)

    return sorted(files)



def chunk_python_file(repository_root:Path, file_path: Path)-> list[RetrievedContext]:
    try:
        with tokenize.open(file_path) as handle:
            source = handle.read()
        tree =ast.parse(source)
        relative_filename = str(file_path.relative_to(repository_root))
    except (SyntaxError, UnicodeDecodeError, OSError, ValueError) as error:
        logger.warning("Skipping %s: %s", file_path,error)
        return []
    source_lines = source.split("\n")
    contexts:list[RetrievedContext]=[]
    
    #runs of consecutive top-level statements that aren't defs: imports, constants, config.
    #these carry the repo's conventions and were previously never indexed at all.
    module_run_start: int | None = None
    module_run_end: int | None = None

    def flush_module_run()->None:
        nonlocal module_run_start, module_run_end
        if module_run_start is not None and module_run_end is not None:
            contexts.extend(_make_context(relative_filename, source_lines, module_run_start, module_run_end))
        module_run_start = None
        module_run_end = None

    for node in tree.body:
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
             flush_module_run()
             contexts.extend(_make_context(relative_filename, source_lines, _start_line(node), _end_line(node)))

         elif isinstance(node, ast.ClassDef):
             flush_module_run()
             contexts.extend(_chunk_class(relative_filename, source_lines, node))
         else:
             if module_run_start is None:
                 module_run_start = _start_line(node)
             module_run_end = _end_line(node)
    flush_module_run()

    return contexts



def passage_text(chunk: RetrievedContext) -> str:
    """Exact text that gets embedded for a chunk.

    The token budget must be checked against this, not against `chunk.content`:
    the filename header is part of what the model reads.
    """
    return f"File: {chunk.filename}\n{chunk.content}"


def _split_point(lines: list[str]) -> int:
    """Index to cut `lines` at, so the left half is lines[:i] and the right lines[i:].

    Prefers the blank line closest to the middle (within the middle half) so a
    cut tends to fall between statements rather than inside one; otherwise cuts
    at the midpoint. Always returns 1 <= i <= len(lines) - 1.
    """
    count = len(lines)
    middle = count // 2
    lower, upper = max(1, count // 4), min(count - 1, (3 * count) // 4)
    blanks = [i for i in range(lower, upper + 1) if not lines[i - 1].strip()]
    if blanks:
        return min(blanks, key=lambda i: abs(i - middle))
    return max(1, min(count - 1, middle))


def _truncate_to_budget(
    chunk: RetrievedContext,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> list[RetrievedContext]:
    """Last resort for a single line that alone exceeds the budget."""
    low, high = 0, len(chunk.content)
    while low < high:  # largest prefix length that still fits
        mid = (low + high + 1) // 2
        if count_tokens(passage_text(replace(chunk, content=chunk.content[:mid]))) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    logger.warning(
        "Truncated a single %d-character line at %s:%d to fit %d tokens",
        len(chunk.content), chunk.filename, chunk.start_line, max_tokens,
    )
    if low == 0:
        return []
    return [replace(chunk, content=chunk.content[:low])]


def fit_to_token_budget(
    chunk: RetrievedContext,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> list[RetrievedContext]:
    """Split `chunk` until every piece's embedded text fits in `max_tokens`.

    The line cap in `_make_context` is only a proxy for a token cap: real code
    with long docstrings or dense lines blows through it (measured: 4-22% of
    40-line chunks exceed 512 tokens on PyGithub / sentence-transformers /
    numpy). An embedding model truncates silently, so an over-budget chunk would
    be indexed by its first half only, with no error.

    Splitting is by recursive halving on line boundaries, so every original line
    stays covered exactly once and line numbers stay exact.
    """
    if count_tokens(passage_text(chunk)) <= max_tokens:
        return [chunk]

    lines = chunk.content.split("\n")
    if len(lines) == 1:
        return _truncate_to_budget(chunk, count_tokens, max_tokens)

    cut = _split_point(lines)
    pieces = [
        replace(
            chunk,
            content="\n".join(lines[:cut]),
            end_line=chunk.start_line + cut - 1,
        ),
        replace(
            chunk,
            content="\n".join(lines[cut:]),
            start_line=chunk.start_line + cut,
        ),
    ]
    fitted: list[RetrievedContext] = []
    for piece in pieces:
        if piece.content.strip():  # a half that is only blank lines isn't worth indexing
            fitted.extend(fit_to_token_budget(piece, count_tokens, max_tokens))
    return fitted
