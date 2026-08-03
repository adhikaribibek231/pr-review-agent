import pytest
from pr_agent.diff_parser import (DiffHunk, is_noise_file, parse_changed_lines, parse_hunk_header, parse_hunks, split_hunks)


def test_parse_hunk_header_with_counts() -> None:
    result = parse_hunk_header("@@ -10,3 +12,4 @@")

    assert result == (10, 3, 12, 4)


def test_parse_hunk_header_without_counts() -> None:
    result = parse_hunk_header("@@ -10 +12 @@")

    assert result == (10, 1, 12, 1)


def test_parse_hunk_header_with_function_context() -> None:
    result = parse_hunk_header(
        "@@ -33,7 +33,7 @@ def main() -> None:"
    )

    assert result == (33, 7, 33, 7)


def test_invalid_hunk_header() -> None:
    with pytest.raises(ValueError, match="Invalid hunk header"):
        parse_hunk_header("not a hunk header")


def test_single_added_line() -> None:
    hunk = """@@ -10,2 +10,3 @@
 context
+added
 context"""

    added_lines, deleted_lines = parse_changed_lines(hunk)

    assert added_lines == frozenset({11})
    assert deleted_lines == frozenset()


def test_replacement() -> None:
    hunk = """@@ -4,2 +4,2 @@
-old
+new
 context"""

    added_lines, deleted_lines = parse_changed_lines(hunk)

    assert added_lines == frozenset({4})
    assert deleted_lines == frozenset({4})


def test_consecutive_added_lines() -> None:
    hunk = """@@ -20,1 +20,3 @@
 context
+first
+second"""

    added_lines, deleted_lines = parse_changed_lines(hunk)

    assert added_lines == frozenset({21, 22})
    assert deleted_lines == frozenset()


def test_consecutive_deleted_lines() -> None:
    hunk = """@@ -20,3 +20,1 @@
 context
-first
-second"""

    added_lines, deleted_lines = parse_changed_lines(hunk)

    assert added_lines == frozenset()
    assert deleted_lines == frozenset({21, 22})


def test_no_newline_metadata_is_ignored() -> None:
    hunk = """@@ -1 +1 @@
-old
+new
\\ No newline at end of file"""

    added_lines, deleted_lines = parse_changed_lines(hunk)

    assert added_lines == frozenset({1})
    assert deleted_lines == frozenset({1})


def test_split_multiple_hunks() -> None:
    patch = """@@ -1,2 +1,2 @@
-old
+new
 context
@@ -20,2 +20,3 @@
 context
+another
 context"""

    hunks = split_hunks(patch)

    assert len(hunks) == 2
    assert hunks[0].startswith("@@ -1,2 +1,2 @@")
    assert hunks[1].startswith("@@ -20,2 +20,3 @@")


def test_parse_hunks_returns_structured_hunks() -> None:
    patch = """@@ -4,2 +4,2 @@
-old
+new
 context"""

    result = parse_hunks("main.py", patch)

    assert result == [
        DiffHunk(
            filename="main.py",
            patch=patch,
            added_lines=frozenset({4}),
            deleted_lines=frozenset({4}),
        )
    ]


def test_parse_hunks_with_missing_patch() -> None:
    assert parse_hunks("main.py", None) == []


def test_parse_real_pr_patch() -> None:
    patch = '''@@ -33,7 +33,7 @@ def main() -> None:
 
     print("1. Extracting text from PDFs...")
     for pdf_path in input_pdfs:
-        text_path = extract_text_from_pdf(pdf_path)
+        text_path: list[str] = extract_text_from_pdf(pdf_path)
         extracted_text_paths.append(text_path)
 
     extracted_facts_path = output_dir / "extracted_facts.json"'''

    result = parse_hunks("main.py", patch)

    assert len(result) == 1
    assert result[0].added_lines == frozenset({36})
    assert result[0].deleted_lines == frozenset({36})


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("package-lock.json", True),
        ("frontend/package-lock.json", True),
        ("poetry.lock", True),
        ("src/main.py", False),
        ("README.md", False),
    ],
)
def test_is_noise_file(filename: str, expected: bool) -> None:
    assert is_noise_file(filename) is expected

