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


def test_blank_context_line_without_leading_space() -> None:
    hunk = "@@ -1,3 +1,4 @@\n ctx\n\n+new"

    added_lines, deleted_lines = parse_changed_lines(hunk)

    assert added_lines == frozenset({3})
    assert deleted_lines == frozenset()


# ---- where a finding about a pure removal may be attached ----------------------------------------------------------------------------


def _anchors(patch: str) -> set[int]:
    from pr_agent.diff_parser import deletion_anchor_lines

    return set(deletion_anchor_lines(parse_hunks("a.py", patch)[0]))


def test_a_deletion_between_context_lines_anchors_to_both_neighbours() -> None:
    assert _anchors("@@ -1,5 +1,3 @@\n keep1\n keep2\n-gone1\n-gone2\n keep3\n keep4") == {2, 3}


def test_a_deletion_at_the_end_anchors_to_the_line_above_only() -> None:
    assert _anchors("@@ -1,3 +1,1 @@\n keep\n-gone1\n-gone2") == {1}


def test_a_deletion_at_the_start_anchors_to_the_line_below_only() -> None:
    assert _anchors("@@ -1,3 +1,1 @@\n-gone1\n-gone2\n keep") == {1}


def test_two_separate_deletions_each_get_their_own_anchors() -> None:
    assert _anchors("@@ -1,7 +1,5 @@\n a\n-x\n b\n c\n-y\n d\n e") == {1, 2, 3, 4}


def test_a_hunk_that_adds_anything_has_no_deletion_anchors() -> None:
    assert _anchors("@@ -1,3 +1,3 @@\n keep\n-old\n+new\n tail") == set()


def test_a_blank_line_stripped_of_its_space_still_counts_as_context() -> None:
    assert _anchors("@@ -1,4 +1,3 @@\n a\n-gone\n\n tail") == {1, 2}


def test_a_hunk_with_no_patch_text_has_no_anchors() -> None:
    from pr_agent.diff_parser import deletion_anchor_lines

    empty = DiffHunk(filename="a.py", patch="", added_lines=frozenset(), deleted_lines=frozenset({3}))
    assert deletion_anchor_lines(empty) == frozenset()
