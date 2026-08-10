from pathlib import Path


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
    """Return Python source files that are suitable for indexing."""
    files = []

    for path in repository_root.rglob("*.py"):
        relative_parts = path.relative_to(repository_root).parts
        if any(part in EXCLUDED_DIRS for part in relative_parts):
            continue
        if path.is_file():
            files.append(path)

    return sorted(files)
