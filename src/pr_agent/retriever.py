import ast
from pathlib import Path
from pr_agent.models import RetrievedContext

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
     source = file_path.read_text(encoding="utf-8")
     source_lines = source.splitlines()

     tree = ast.parse(source)

     results:list[RetrievedContext] = []
     relative_filename = str(file_path.relative_to(repository_root))

     for node in tree.body:
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = node.end_lineno if node.end_lineno is not None else start_line

            chunk_lines = source_lines[start_line -1: end_line]
            content = "\n".join(chunk_lines)

            context = RetrievedContext(
                    filename=relative_filename,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    score = 0.0,
                    )
            results.append(context)
     return results
