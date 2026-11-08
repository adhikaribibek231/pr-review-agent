import re
import zlib
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from pr_agent.diff_parser import DiffHunk, parse_hunks
from pr_agent.retriever import passage_text
from pr_agent.search import (
    CodeIndex,
    build_hunk_query,
    build_index,
    changed_new_lines,
    referenced_definitions,
    retrieve_for_hunk,
)

# --------------------------------------------------------------------------
# a deterministic stand-in for the embedding model
# --------------------------------------------------------------------------

DIM = 128


class FakeEmbedder:
    """Bag-of-words hashing: texts sharing words get similar vectors.

    Tokens are whitespace-separated words, so token budgets are easy to reason
    about in tests, and nothing is downloaded or loaded.
    """

    def __init__(self, max_tokens: int = 10_000) -> None:
        self._max_tokens = max_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vector(self, text: str) -> NDArray[np.float32]:
        vector = np.zeros(DIM, dtype=np.float32)
        for word in re.findall(r"[a-z]+", text.lower().replace("_", " ")):
            vector[zlib.crc32(word.encode()) % DIM] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._vector(text)


def write_file(root: Path, name: str, source: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(source, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# build_index
# --------------------------------------------------------------------------


def test_build_index_guarantees_every_chunk_fits_the_budget(tmp_path: Path) -> None:
    # Few lines (under MAX_CHUNK_LINES) but far too many "tokens": the case a
    # line cap alone cannot catch.
    long_docstring = " ".join(f"word{i}" for i in range(300))
    write_file(tmp_path, "dense.py", f'def f():\n    """{long_docstring}"""\n    return 1\n')
    embedder = FakeEmbedder(max_tokens=50)

    index = build_index(tmp_path, embedder)

    assert len(index.chunks) >= 1
    for chunk in index.chunks:
        assert embedder.count_tokens(passage_text(chunk)) <= 50


def test_build_index_embeds_one_vector_per_chunk(tmp_path: Path) -> None:
    write_file(tmp_path, "a.py", "def one():\n    return 1\n\ndef two():\n    return 2\n")
    index = build_index(tmp_path, FakeEmbedder())
    assert index.embeddings.shape == (len(index.chunks), DIM)


def test_build_index_on_empty_repository_is_empty_and_searchable(tmp_path: Path) -> None:
    index = build_index(tmp_path, FakeEmbedder())
    assert index.chunks == ()
    assert index.search("anything") == []


# --------------------------------------------------------------------------
# CodeIndex.search
# --------------------------------------------------------------------------


@pytest.fixture
def index(tmp_path: Path) -> CodeIndex:
    write_file(
        tmp_path,
        "validation.py",
        "def validate_findings(findings, hunks):\n    return [f for f in findings if f.valid]\n",
    )
    write_file(
        tmp_path,
        "client.py",
        "def post_review(pull_request, comments):\n    pull_request.create_review(comments)\n",
    )
    write_file(
        tmp_path,
        "parser.py",
        "def parse_hunks(patch):\n    return patch.split('@@')\n",
    )
    return build_index(tmp_path, FakeEmbedder())


def test_search_ranks_the_matching_chunk_first(index: CodeIndex) -> None:
    hits = index.search("validate findings hunks", top_k=3)
    assert hits[0].filename == "validation.py"


def test_search_populates_scores_in_descending_order(index: CodeIndex) -> None:
    hits = index.search("post review pull request", top_k=3)
    assert hits[0].filename == "client.py"
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0.0


def test_search_respects_top_k(index: CodeIndex) -> None:
    assert len(index.search("review", top_k=2)) == 2
    assert len(index.search("review", top_k=50)) == len(index.chunks)
    assert index.search("review", top_k=0) == []


def test_search_does_not_mutate_the_index(index: CodeIndex) -> None:
    _ = index.search("validate findings")
    assert all(chunk.score == 0.0 for chunk in index.chunks)


def test_exclude_is_applied_before_the_cut(index: CodeIndex) -> None:
    hits = index.search(
        "validate findings hunks",
        top_k=2,
        exclude=lambda chunk: chunk.filename == "validation.py",
    )
    assert len(hits) == 2
    assert all(hit.filename != "validation.py" for hit in hits)


# --------------------------------------------------------------------------
# build_hunk_query
# --------------------------------------------------------------------------

PATCH = """@@ -10,4 +10,6 @@ def validate_findings(findings, hunks):
 context_before = 1
+added_one = compute(findings)
-removed_line = 2
+added_two = filter_hunks(hunks)
 context_after = 3"""


def make_hunk(patch: str = PATCH, filename: str = "src/validator.py") -> DiffHunk:
    return parse_hunks(filename, patch)[0]


def words(text: str) -> int:
    return FakeEmbedder().count_tokens(text)


def test_hunk_query_strips_diff_markers_and_deleted_lines() -> None:
    query = build_hunk_query(make_hunk(), words, 1000)

    assert "added_one = compute(findings)" in query
    assert "added_two = filter_hunks(hunks)" in query
    assert "context_before = 1" in query
    assert "removed_line" not in query
    assert "\n+" not in query
    assert "@@" not in query


def test_hunk_query_names_the_file_and_enclosing_function() -> None:
    query = build_hunk_query(make_hunk(), words, 1000)
    assert "File: src/validator.py" in query
    assert "In: def validate_findings(findings, hunks):" in query


def test_hunk_query_drops_context_before_changed_lines() -> None:
    full = build_hunk_query(make_hunk(), words, 1000)
    tight = build_hunk_query(make_hunk(), words, words(full) - 1)

    assert "context_before" not in tight
    assert "added_one = compute(findings)" in tight
    assert words(tight) <= words(full) - 1


def test_hunk_query_always_fits_the_budget() -> None:
    patch = "@@ -1,0 +1,40 @@\n" + "\n".join(f"+value_{i} = call({i}, a, b, c)" for i in range(40))
    for budget in (10, 25, 60, 500):
        assert words(build_hunk_query(make_hunk(patch), words, budget)) <= budget


def test_pure_deletion_hunk_queries_with_the_deleted_lines() -> None:
    patch = "@@ -5,3 +5,1 @@ def f():\n keep = 1\n-gone_one = 1\n-gone_two = 2"
    query = build_hunk_query(make_hunk(patch), words, 1000)
    assert "gone_one = 1" in query
    assert "gone_two = 2" in query


# --------------------------------------------------------------------------
# retrieve_for_hunk
# --------------------------------------------------------------------------


@pytest.fixture
def repo_index(tmp_path: Path) -> CodeIndex:
    write_file(
        tmp_path,
        "validator.py",
        "def validate_findings(findings, hunks):\n"
        "    valid = compute_valid_locations(hunks)\n"
        "    return [f for f in findings if f in valid]\n"
        "\n\n"
        "def helper_unrelated():\n"
        "    return 'nothing to see'\n",
    )
    write_file(
        tmp_path,
        "locations.py",
        "def compute_valid_locations(hunks):\n"
        "    return {line for hunk in hunks for line in hunk.added_lines}\n",
    )
    return build_index(tmp_path, FakeEmbedder())


def test_retrieval_excludes_the_chunk_being_changed(repo_index: CodeIndex) -> None:
    patch = (
        "@@ -0,0 +1,3 @@\n"
        "+def validate_findings(findings, hunks):\n"
        "+    valid = compute_valid_locations(hunks)\n"
        "+    return [f for f in findings if f in valid]"
    )
    hits = retrieve_for_hunk(repo_index, make_hunk(patch, "validator.py"), top_k=5)

    assert hits, "expected related context"
    assert not any(h.filename == "validator.py" and h.start_line <= 1 <= h.end_line for h in hits)
    assert hits[0].filename == "locations.py"


def test_retrieval_keeps_other_chunks_from_the_same_file(repo_index: CodeIndex) -> None:
    patch = "@@ -0,0 +1,1 @@\n+    valid = compute_valid_locations(hunks)"
    hunk = make_hunk(patch.replace("+1,1", "+2,1"), "validator.py")
    hits = retrieve_for_hunk(repo_index, hunk, top_k=5)

    assert any(h.filename == "validator.py" and "helper_unrelated" in h.content for h in hits)


def test_retrieval_can_include_the_changed_code_when_asked(repo_index: CodeIndex) -> None:
    patch = "@@ -0,0 +1,2 @@\n+def validate_findings(findings, hunks):\n+    valid = compute_valid_locations(hunks)"
    hits = retrieve_for_hunk(
        repo_index, make_hunk(patch, "validator.py"), top_k=5, exclude_changed_code=False
    )
    # Included again (not necessarily first: the called function's definition leads).
    assert any(h.filename == "validator.py" and h.start_line == 1 for h in hits)


def test_changed_new_lines_for_additions_and_pure_deletions() -> None:
    assert changed_new_lines(make_hunk()) == frozenset({11, 12})
    deletion = make_hunk("@@ -5,3 +5,1 @@\n keep = 1\n-gone_one = 1\n-gone_two = 2")
    assert changed_new_lines(deletion) == frozenset({5})


# --------------------------------------------------------------------------
# definition lookup (deterministic callee retrieval)
# --------------------------------------------------------------------------


@pytest.fixture
def callee_index(tmp_path: Path) -> CodeIndex:
    write_file(tmp_path, "models.py", "class Finding:\n    line: int\n")
    write_file(tmp_path, "renderer.py", "def render_inline_comment(finding):\n    return str(finding)\n")
    write_file(tmp_path, "noise.py", "def unrelated_helper():\n    return 'zebra giraffe lion'\n")
    write_file(
        tmp_path,
        "client.py",
        "def post_review(findings):\n    return [render_inline_comment(f) for f in findings]\n",
    )
    return build_index(tmp_path, FakeEmbedder())


def callee_hunk() -> DiffHunk:
    patch = "@@ -0,0 +1,2 @@\n+def post_review(findings):\n+    return [render_inline_comment(f) for f in findings]"
    return make_hunk(patch, "client.py")


def test_index_maps_names_to_the_chunks_that_define_them(callee_index: CodeIndex) -> None:
    definitions = callee_index.definitions
    assert set(definitions) == {"Finding", "render_inline_comment", "unrelated_helper", "post_review"}
    position = definitions["Finding"][0]
    assert callee_index.chunks[position].filename == "models.py"


def test_called_function_is_retrieved_by_name(callee_index: CodeIndex) -> None:
    hits = retrieve_for_hunk(callee_index, callee_hunk(), top_k=3)
    assert hits[0].filename == "renderer.py"
    assert hits[0].score != 0.0


def test_definition_lookup_can_be_switched_off(callee_index: CodeIndex) -> None:
    with_lookup = retrieve_for_hunk(callee_index, callee_hunk(), top_k=1)
    without = retrieve_for_hunk(callee_index, callee_hunk(), top_k=3, lookup_definitions=False)
    assert with_lookup[0].filename == "renderer.py"
    assert len(without) == 3  # semantic search alone still fills every slot


def test_results_contain_no_duplicates_and_respect_top_k(callee_index: CodeIndex) -> None:
    hits = retrieve_for_hunk(callee_index, callee_hunk(), top_k=3)
    keys = [(h.filename, h.start_line) for h in hits]
    assert len(keys) == len(set(keys)) <= 3


def test_attribute_names_are_not_resolved_as_definitions(tmp_path: Path) -> None:
    write_file(tmp_path, "tool.py", "def search(x):\n    return x\n")
    write_file(tmp_path, "other.py", "def caller():\n    return index.search('q')\n")
    index = build_index(tmp_path, FakeEmbedder())
    patch = "@@ -0,0 +1,1 @@\n+    return index.search('q')"
    # `search` is only used as an attribute here, so it must not be looked up by name
    assert referenced_definitions(index, make_hunk(patch, "other.py"), lambda _c: False) == []


def test_generic_names_defined_in_many_places_are_ignored(tmp_path: Path) -> None:
    for name in ("a.py", "b.py", "c.py"):
        write_file(tmp_path, name, "def main():\n    return 1\n")
    write_file(tmp_path, "caller.py", "def run():\n    return main()\n")
    index = build_index(tmp_path, FakeEmbedder())
    hunk = make_hunk("@@ -0,0 +1,1 @@\n+    return main()", "caller.py")
    assert referenced_definitions(index, hunk, lambda _c: False) == []


def test_changed_code_is_never_returned_as_its_own_callee(callee_index: CodeIndex) -> None:
    patch = "@@ -0,0 +1,2 @@\n+def render_inline_comment(finding):\n+    return str(finding)"
    hits = retrieve_for_hunk(callee_index, make_hunk(patch, "renderer.py"), top_k=5)
    assert not any(h.filename == "renderer.py" for h in hits)


def test_extra_excluded_dirs_are_left_out_of_the_index(tmp_path: Path) -> None:
    write_file(tmp_path, "src/keep.py", "def keep():\n    return 1\n")
    write_file(tmp_path, "experiments/scratch.py", "def scratch():\n    return 2\n")
    index = build_index(tmp_path, FakeEmbedder(), extra_excluded_dirs={"experiments"})
    assert {c.filename for c in index.chunks} == {"src/keep.py"}


def test_a_local_variable_named_like_a_function_is_not_looked_up(tmp_path: Path) -> None:
    write_file(tmp_path, "helpers.py", "def finding():\n    return 1\n")
    write_file(tmp_path, "client.py", "def post(findings):\n    pass\n")
    index = build_index(tmp_path, FakeEmbedder())
    hunk = make_hunk("@@ -0,0 +1,2 @@\n+    for finding in findings:\n+        print(finding.line)", "client.py")
    assert referenced_definitions(index, hunk, lambda _c: False) == []


def test_a_call_is_looked_up_even_though_a_bare_name_is_not(tmp_path: Path) -> None:
    write_file(tmp_path, "helpers.py", "def finding():\n    return 1\n")
    index = build_index(tmp_path, FakeEmbedder())
    hunk = make_hunk("@@ -0,0 +1,1 @@\n+    value = finding()", "client.py")
    assert [index.chunks[p].filename for p in referenced_definitions(index, hunk, lambda _c: False)] == ["helpers.py"]


def test_a_camelcase_type_is_looked_up_without_a_call(tmp_path: Path) -> None:
    write_file(tmp_path, "models.py", "class Finding:\n    line: int\n")
    index = build_index(tmp_path, FakeEmbedder())
    hunk = make_hunk("@@ -0,0 +1,1 @@\n+def run(items: list[Finding]) -> None:", "client.py")
    assert [index.chunks[p].filename for p in referenced_definitions(index, hunk, lambda _c: False)] == ["models.py"]


def test_a_name_the_hunk_defines_is_not_a_reference(tmp_path: Path) -> None:
    write_file(tmp_path, "old.py", "def compute(x):\n    return x\n")
    index = build_index(tmp_path, FakeEmbedder())
    hunk = make_hunk("@@ -0,0 +1,2 @@\n+def compute(x):\n+    return x * 2", "new.py")
    assert referenced_definitions(index, hunk, lambda _c: False) == []


def test_called_functions_rank_ahead_of_type_mentions(tmp_path: Path) -> None:
    write_file(tmp_path, "models.py", "class Finding:\n    line: int\n")
    write_file(tmp_path, "client.py", "def fetch():\n    return 1\n")
    index = build_index(tmp_path, FakeEmbedder())
    # The type appears first (in the signature), the call after: the call must lead.
    hunk = make_hunk("@@ -0,0 +1,2 @@\n+def run(items: list[Finding]) -> None:\n+    fetch()", "app.py")
    found = [index.chunks[p].filename for p in referenced_definitions(index, hunk, lambda _c: False)]
    assert found == ["client.py", "models.py"]
