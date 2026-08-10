from pathlib import Path

import pytest

from pr_agent.retriever import discover_python_files


def test_discovers_only_repository_python_files_in_sorted_order(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    package = src / "package"
    package.mkdir(parents=True)

    main_file = src / "main.py"
    utils_file = package / "utils.py"
    # Create files in reverse lexical order so the assertion also checks sorting.
    utils_file.write_text("def foo(): pass\n", encoding="utf-8")
    main_file.write_text("print('hello')\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text("# Project\n", encoding="utf-8")

    ignored_file = tmp_path / ".venv" / "dependency.py"
    ignored_file.parent.mkdir()
    ignored_file.write_text("def dependency(): pass\n", encoding="utf-8")

    result = discover_python_files(tmp_path)

    assert result == [
        main_file,
        utils_file,
    ]


def test_discovers_python_files_in_deterministic_order(tmp_path: Path) -> None:
    (tmp_path / "z_module.py").touch()
    package = tmp_path / "package"
    package.mkdir()
    (package / "a_module.py").touch()
    (tmp_path / "README.md").touch()

    assert discover_python_files(tmp_path) == [
        package / "a_module.py",
        tmp_path / "z_module.py",
    ]


@pytest.mark.parametrize(
    "excluded_dir",
    [
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
    ],
)
def test_ignores_python_files_in_excluded_directories(
    tmp_path: Path,
    excluded_dir: str,
) -> None:
    excluded_path = tmp_path / "nested" / excluded_dir
    excluded_path.mkdir(parents=True)
    (excluded_path / "ignored.py").touch()
    included_path = tmp_path / "included.py"
    included_path.touch()

    assert discover_python_files(tmp_path) == [included_path]
