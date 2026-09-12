import ast
import logging
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

def discover_python_files(repository_root: Path) -> list[Path]:
    
    files:list[Path] = []

    for path in repository_root.rglob("*.py"):
        relative_parts = path.relative_to(repository_root).parts
        if any(part in EXCLUDED_DIRS for part in relative_parts):
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
