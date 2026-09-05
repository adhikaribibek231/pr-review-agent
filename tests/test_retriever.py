from pathlib import Path

import pytest

from pr_agent import retriever
from pr_agent.models import RetrievedContext
from pr_agent.retriever import chunk_python_file, discover_python_files

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


@pytest.fixture
def small_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the chunk cap so split paths are reachable without huge fixtures.

    monkeypatch restores the originals after each test, so ordering can't leak.
    """
    monkeypatch.setattr(retriever, "MAX_CHUNK_LINES", 6)
    monkeypatch.setattr(retriever, "CHUNK_OVERLAP_LINES", 2)


def write_file(tmp_path: Path, name: str, source: str) -> Path:
    file_path = tmp_path / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    _ = file_path.write_text(source, encoding="utf-8")
    return file_path


def covered_lines(chunks: list[RetrievedContext]) -> set[int]:
    covered: set[int] = set()
    for chunk in chunks:
        covered |= set(range(chunk.start_line, chunk.end_line + 1))
    return covered


def first_lines(chunks: list[RetrievedContext]) -> list[str]:
    return [chunk.content.splitlines()[0].strip() for chunk in chunks]


# --------------------------------------------------------------------------
# discover_python_files
# --------------------------------------------------------------------------


def test_discovers_only_repository_python_files_in_sorted_order(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    package = src / "package"
    package.mkdir(parents=True)

    main_file = src / "main.py"
    utils_file = package / "utils.py"
    # Create files in reverse lexical order so the assertion also checks sorting.
    _ = utils_file.write_text("def foo(): pass\n", encoding="utf-8")
    _ = main_file.write_text("print('hello')\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    _ = readme.write_text("# Project\n", encoding="utf-8")

    ignored_file = tmp_path / ".venv" / "dependency.py"
    ignored_file.parent.mkdir()
    _ = ignored_file.write_text("def dependency(): pass\n", encoding="utf-8")

    result = discover_python_files(tmp_path)

    assert result == [main_file, utils_file]


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


def test_empty_repository_returns_empty_list(tmp_path: Path) -> None:
    assert discover_python_files(tmp_path) == []


def test_exclusion_matches_whole_directory_name_only(tmp_path: Path) -> None:
    """'buildings' must not be excluded just because 'build' is in the set."""
    kept = write_file(tmp_path, "buildings/model.py", "x = 1\n")
    also_kept = write_file(tmp_path, "distribution/util.py", "y = 2\n")

    assert discover_python_files(tmp_path) == sorted([kept, also_kept])


def test_exclusion_applies_relative_to_repository_root(tmp_path: Path) -> None:
    """A repo cloned into a path containing 'build' must not exclude itself."""
    repository_root = tmp_path / "build" / "myproject"
    kept = write_file(repository_root, "main.py", "x = 1\n")

    assert discover_python_files(repository_root) == [kept]


def test_non_python_files_are_ignored(tmp_path: Path) -> None:
    write_file(tmp_path, "notes.md", "# hi\n")
    write_file(tmp_path, "config.yaml", "a: 1\n")
    write_file(tmp_path, "script.pyc", "junk\n")
    kept = write_file(tmp_path, "real.py", "x = 1\n")

    assert discover_python_files(tmp_path) == [kept]


# --------------------------------------------------------------------------
# chunk_python_file — coverage and line-number invariants
# --------------------------------------------------------------------------


def test_large_class_indexes_every_line(tmp_path: Path, small_cap: None) -> None:
    """No line inside a class may be silently dropped.

    Attributes and nested classes between methods are the regression this
    guards: they are neither the skeleton nor a method.
    """
    file_path = write_file(
        tmp_path,
        "service.py",
        '''class Service:
    """Docstring."""

    table = "users"

    def first(self) -> int:
        return 1

    SECRET_KEY = "hunter2"
    LIMIT = 100

    class Nested:
        pass

    def second(self) -> int:
        return 2
''',
    )

    chunks = chunk_python_file(tmp_path, file_path)
    covered = covered_lines(chunks)
    source_lines = file_path.read_text(encoding="utf-8").splitlines()

    missing = [
        number
        for number, text in enumerate(source_lines, start=1)
        if text.strip() and number not in covered
    ]

    assert missing == [], f"lines never indexed: {missing}"


def test_module_level_lines_are_all_indexed(tmp_path: Path, small_cap: None) -> None:
    file_path = write_file(
        tmp_path,
        "mixed.py",
        '''"""Module docstring."""

import os

CONSTANT = 1


def alpha() -> None:
    return None


OTHER = 2


def beta() -> None:
    return None


if __name__ == "__main__":
    alpha()
''',
    )

    chunks = chunk_python_file(tmp_path, file_path)
    covered = covered_lines(chunks)
    source_lines = file_path.read_text(encoding="utf-8").splitlines()

    missing = [
        number
        for number, text in enumerate(source_lines, start=1)
        if text.strip() and number not in covered
    ]

    assert missing == [], f"lines never indexed: {missing}"


def test_oversized_span_reports_correct_line_numbers(tmp_path: Path) -> None:
    """start_line/end_line must point at the text actually stored.

    Guards the 1-indexed/0-indexed conversion in _make_context, which fails
    silently: content stays correct while metadata drifts.
    """
    body = "\n".join(f"    x = {index}" for index in range(200))
    file_path = write_file(tmp_path, "big.py", f"def big():\n{body}\n")

    chunks = chunk_python_file(tmp_path, file_path)
    source_lines = file_path.read_text(encoding="utf-8").splitlines()

    assert len(chunks) > 1
    for chunk in chunks:
        chunk_lines = chunk.content.splitlines()
        assert chunk_lines[0] == source_lines[chunk.start_line - 1]
        assert chunk_lines[-1] == source_lines[chunk.end_line - 1]


def test_line_numbers_match_content_for_every_chunk(
    tmp_path: Path, small_cap: None
) -> None:
    """The same invariant, but across classes, methods and module runs."""
    file_path = write_file(
        tmp_path,
        "wide.py",
        '''import os

VALUE = 1


@decorator
class Thing:
    """Doc."""

    attribute = 2

    @property
    def prop(self) -> int:
        return 1

    async def work(self) -> None:
        return None


def free() -> None:
    return None
''',
    )

    chunks = chunk_python_file(tmp_path, file_path)
    source_lines = file_path.read_text(encoding="utf-8").splitlines()

    for chunk in chunks:
        expected = "\n".join(source_lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.content == expected


def test_no_chunk_exceeds_the_cap(tmp_path: Path, small_cap: None) -> None:
    body = "\n".join(f"    x = {index}" for index in range(50))
    file_path = write_file(tmp_path, "long.py", f"def long_one():\n{body}\n")

    for chunk in chunk_python_file(tmp_path, file_path):
        span = chunk.end_line - chunk.start_line + 1
        assert span <= retriever.MAX_CHUNK_LINES


def test_windows_overlap_so_no_boundary_is_lost(
    tmp_path: Path, small_cap: None
) -> None:
    """Consecutive windows must share lines, or a split can orphan a construct."""
    body = "\n".join(f"    x = {index}" for index in range(30))
    file_path = write_file(tmp_path, "long.py", f"def long_one():\n{body}\n")

    chunks = chunk_python_file(tmp_path, file_path)

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start_line <= previous.end_line


def test_chunks_are_returned_in_ascending_line_order(
    tmp_path: Path, small_cap: None
) -> None:
    file_path = write_file(
        tmp_path,
        "ordered.py",
        '''import os

CONSTANT = 1


class Big:
    """Doc."""

    attribute = 1

    def one(self) -> None:
        return None

    def two(self) -> None:
        return None


def last() -> None:
    return None
''',
    )

    starts = [chunk.start_line for chunk in chunk_python_file(tmp_path, file_path)]

    assert starts == sorted(starts)


# --------------------------------------------------------------------------
# chunk_python_file — chunk shape
# --------------------------------------------------------------------------


def test_chunk_starts_at_decorator_not_def(tmp_path: Path) -> None:
    file_path = write_file(
        tmp_path,
        "routes.py",
        '''@app.route("/health")
@requires_auth
def health() -> str:
    return "ok"
''',
    )

    chunks = chunk_python_file(tmp_path, file_path)

    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert '@app.route("/health")' in chunks[0].content
    assert "@requires_auth" in chunks[0].content


def test_decorated_class_chunk_includes_the_decorator(tmp_path: Path) -> None:
    file_path = write_file(
        tmp_path,
        "models.py",
        '''@dataclass
class Point:
    x: int
    y: int
''',
    )

    chunks = chunk_python_file(tmp_path, file_path)

    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].content.startswith("@dataclass")


def test_method_decorator_belongs_to_the_method_not_the_header(
    tmp_path: Path, small_cap: None
) -> None:
    file_path = write_file(
        tmp_path,
        "svc.py",
        '''class Service:
    """Doc."""

    attribute = 1

    @property
    def value(self) -> int:
        return 1

    def other(self) -> None:
        return None
''',
    )

    chunks = chunk_python_file(tmp_path, file_path)
    property_chunks = [c for c in chunks if "def value" in c.content]

    assert len(property_chunks) == 1
    assert property_chunks[0].content.lstrip().startswith("@property")


def test_async_function_is_chunked(tmp_path: Path) -> None:
    file_path = write_file(
        tmp_path,
        "async_mod.py",
        '''async def fetch() -> None:
    return None
''',
    )

    assert first_lines(chunk_python_file(tmp_path, file_path)) == [
        "async def fetch() -> None:"
    ]


def test_small_class_stays_whole(tmp_path: Path) -> None:
    file_path = write_file(
        tmp_path,
        "small.py",
        '''class Small:
    def a(self) -> int:
        return 1

    def b(self) -> int:
        return 2
''',
    )

    chunks = chunk_python_file(tmp_path, file_path)

    assert len(chunks) == 1
    assert chunks[0].content.startswith("class Small:")


def test_large_class_splits_into_methods(tmp_path: Path, small_cap: None) -> None:
    file_path = write_file(
        tmp_path,
        "large.py",
        '''class Large:
    """Doc."""

    def authenticate(self, user: str) -> str:
        return "SELECT * FROM users WHERE u='" + user + "'"

    def logout(self) -> None:
        return None
''',
    )

    starts = first_lines(chunk_python_file(tmp_path, file_path))

    assert "def authenticate(self, user: str) -> str:" in starts
    assert "def logout(self) -> None:" in starts


def test_large_class_is_not_also_emitted_whole(
    tmp_path: Path, small_cap: None
) -> None:
    """Whole-or-parts, never both: double-indexing wastes retrieval slots."""
    file_path = write_file(
        tmp_path,
        "large.py",
        '''class Large:
    """Doc."""

    def one(self) -> None:
        return None

    def two(self) -> None:
        return None
''',
    )

    chunks = chunk_python_file(tmp_path, file_path)
    class_span = (1, len(file_path.read_text(encoding="utf-8").splitlines()))

    assert all((c.start_line, c.end_line) != class_span for c in chunks)


def test_nested_class_inside_large_class_is_chunked(
    tmp_path: Path, small_cap: None
) -> None:
    file_path = write_file(
        tmp_path,
        "nested.py",
        '''class Outer:
    """Doc."""

    attribute = 1

    class Meta:
        ordering = ["name"]
        verbose_name = "thing"

    def method(self) -> None:
        return None
''',
    )

    chunks = chunk_python_file(tmp_path, file_path)

    assert any("class Meta:" in chunk.content for chunk in chunks)


def test_methodless_long_class_is_windowed(tmp_path: Path, small_cap: None) -> None:
    """A settings or enum class has no methods but must still be indexed."""
    fields = "\n".join(f"    FIELD_{index} = {index}" for index in range(20))
    file_path = write_file(tmp_path, "config.py", f"class Config:\n{fields}\n")

    chunks = chunk_python_file(tmp_path, file_path)

    assert len(chunks) > 1
    assert all(
        chunk.end_line - chunk.start_line + 1 <= retriever.MAX_CHUNK_LINES
        for chunk in chunks
    )
    assert any("FIELD_19" in chunk.content for chunk in chunks)


def test_module_level_run_is_grouped_into_one_chunk(tmp_path: Path) -> None:
    """Imports and constants carry repo conventions; a lone import is useless."""
    file_path = write_file(
        tmp_path,
        "settings.py",
        '''import os

MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30


def run() -> None:
    return None
''',
    )

    chunks = chunk_python_file(tmp_path, file_path)
    module_chunks = [c for c in chunks if "MAX_RETRIES" in c.content]

    assert len(module_chunks) == 1
    assert "import os" in module_chunks[0].content
    assert "DEFAULT_TIMEOUT" in module_chunks[0].content


def test_trailing_module_code_after_last_function_is_indexed(
    tmp_path: Path,
) -> None:
    """The final flush_module_run() call is what makes this pass."""
    file_path = write_file(
        tmp_path,
        "entry.py",
        '''def main() -> None:
    return None


if __name__ == "__main__":
    main()
''',
    )

    chunks = chunk_python_file(tmp_path, file_path)

    assert any("__main__" in chunk.content for chunk in chunks)


# --------------------------------------------------------------------------
# chunk_python_file — metadata and failure handling
# --------------------------------------------------------------------------


def test_filename_is_relative_to_repository_root(tmp_path: Path) -> None:
    file_path = write_file(tmp_path, "src/package/module.py", "x = 1\n")

    chunks = chunk_python_file(tmp_path, file_path)

    assert chunks[0].filename == str(Path("src/package/module.py"))


def test_score_is_zero_until_retrieval_populates_it(tmp_path: Path) -> None:
    file_path = write_file(tmp_path, "mod.py", "def f() -> None:\n    return None\n")

    assert all(chunk.score == 0.0 for chunk in chunk_python_file(tmp_path, file_path))


def test_unparseable_file_is_skipped_not_raised(tmp_path: Path) -> None:
    """One bad file must not take down the whole index build."""
    file_path = write_file(tmp_path, "broken.py", "def oops(:\n")

    assert chunk_python_file(tmp_path, file_path) == []

def test_bom_prefixed_file_is_indexed(tmp_path: Path) -> None:
    """Windows editors write a UTF-8 BOM; the file is still valid Python."""
    file_path = tmp_path / "bom.py"
    _ = file_path.write_bytes(b"\xef\xbb\xbfdef f() -> int:\n    return 1\n")

    chunks = chunk_python_file(tmp_path, file_path)

    assert len(chunks) == 1
    assert chunks[0].content.startswith("def f()")


def test_declared_encoding_is_honoured(tmp_path: Path) -> None:
    """A coding declaration is valid Python and must not drop the file."""
    file_path = tmp_path / "latin.py"
    _ = file_path.write_bytes(
        "# -*- coding: latin-1 -*-\nNAME = 'caf\xe9'\n".encode("latin-1")
    )

    chunks = chunk_python_file(tmp_path, file_path)

    assert len(chunks) == 1
    assert "café" in chunks[0].content


def test_undecodable_file_is_skipped(tmp_path: Path) -> None:
    """Bytes that match no declared or detected encoding are still skipped."""
    file_path = tmp_path / "binary.py"
    _ = file_path.write_bytes(b"\xff\xfe\x00\x01garbage\n")

    assert chunk_python_file(tmp_path, file_path) == []


def test_missing_file_is_skipped(tmp_path: Path) -> None:
    assert chunk_python_file(tmp_path, tmp_path / "gone.py") == []


def test_empty_file_produces_no_chunks(tmp_path: Path) -> None:
    file_path = write_file(tmp_path, "empty.py", "")

    assert chunk_python_file(tmp_path, file_path) == []


def test_whitespace_only_file_produces_no_chunks(tmp_path: Path) -> None:
    file_path = write_file(tmp_path, "blank.py", "\n\n\n")

    assert chunk_python_file(tmp_path, file_path) == []


def test_comment_only_file_produces_no_chunks(tmp_path: Path) -> None:
    """ast.parse yields an empty body; comments are not AST nodes."""
    file_path = write_file(tmp_path, "comments.py", "# just a note\n# and another\n")

    assert chunk_python_file(tmp_path, file_path) == []


def test_chunking_is_deterministic(tmp_path: Path, small_cap: None) -> None:
    """Reproducible indexing means a retrieval change implies a code change."""
    file_path = write_file(
        tmp_path,
        "stable.py",
        '''import os

CONSTANT = 1


class Thing:
    """Doc."""

    attribute = 1

    def one(self) -> None:
        return None

    def two(self) -> None:
        return None
''',
    )

    first = chunk_python_file(tmp_path, file_path)
    second = chunk_python_file(tmp_path, file_path)

    assert first == second


def test_class_chunks_do_not_nest_or_duplicate(
    tmp_path: Path, small_cap: None
) -> None:
    """Overlapping class chunks waste embeddings and retrieval slots."""
    file_path = write_file(
        tmp_path,
        "c.py",
        '''class C:
    """Doc."""

    a = 1

    @deco
    def m(self) -> int:
        return 1

    b = 2
''',
    )

    chunks = chunk_python_file(tmp_path, file_path)
    spans = [(c.start_line, c.end_line) for c in chunks]

    assert len(spans) == len(set(spans))
    for outer in spans:
        for inner in spans:
            if outer != inner:
                assert not (
                    outer[0] <= inner[0] and inner[1] <= outer[1]
                ), f"{inner} nested inside {outer}"

