"""Semantic search over a repository's Python code.

    index = build_index(repo_root, embedder)       # embeds every chunk once
    hits  = index.search("where are findings validated?", top_k=5)
    hits  = retrieve_for_hunk(index, hunk)         # PR-aware retrieval

Plain NumPy: embeddings are L2-normalised, so a dot product is cosine
similarity. At repository scale (thousands of chunks) a vector database buys
persistence, not speed; add one when you need the former.
"""

import logging
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pr_agent.diff_parser import DiffHunk, parse_hunk_header
from pr_agent.embedding import Embedder
from pr_agent.models import RetrievedContext
from pr_agent.retriever import (
    chunk_python_file,
    EXCLUDED_DIRS,
    discover_python_files,
    fit_to_token_budget,
    passage_text,
)

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

# A chunk "defines" the first def/class it contains (for a split piece with no
# def line, nothing). Decorators precede it, so match the first definition line.
_DEFINITION = re.compile(r"^[ \t]*(?:async[ \t]+def|def|class)[ \t]+(\w+)", re.MULTILINE)
# A bare name (not `obj.name`) with an optional "(" after it. Attribute names can't
# be resolved statically, and common ones (`search`, `get`) would match unrelated
# definitions.
_NAME_USE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)(\s*\()?")
# `def foo` / `class Foo` in the hunk *introduces* a name; it isn't a use of one.
_INTRODUCES_NAME = re.compile(r"\b(?:def|class)\s+\w+")
# A name defined in more chunks than this is too generic to be a useful lookup.
_MAX_DEFINITIONS_PER_NAME = 2


@dataclass(frozen=True, eq=False)
class CodeIndex:
    """Chunks plus their vectors. Build once per repository state, query often."""

    chunks: tuple[RetrievedContext, ...]
    embeddings: NDArray[np.float32]  # (len(chunks), dim), L2-normalised
    embedder: Embedder

    @cached_property
    def definitions(self) -> dict[str, tuple[int, ...]]:
        """name -> positions of the chunks whose first def/class has that name."""
        found: dict[str, list[int]] = {}
        for position, chunk in enumerate(self.chunks):
            match = _DEFINITION.search(chunk.content)
            if match:
                found.setdefault(match.group(1), []).append(position)
        return {name: tuple(positions) for name, positions in found.items()}

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        exclude: Callable[[RetrievedContext], bool] | None = None,
        query_vector: NDArray[np.float32] | None = None,
    ) -> list[RetrievedContext]:
        """Top-k chunks by cosine similarity, best first, with `score` filled in.

        `exclude` drops candidates *before* the cut, so you still get `top_k`
        results when some of the best matches are filtered out.
        """
        if not self.chunks or top_k <= 0:
            return []
        vector = query_vector if query_vector is not None else self.embedder.embed_query(query)
        scores = self.embeddings @ vector
        hits: list[RetrievedContext] = []
        for position in np.argsort(-scores):
            chunk = self.chunks[int(position)]
            if exclude is not None and exclude(chunk):
                continue
            hits.append(replace(chunk, score=float(scores[position])))
            if len(hits) == top_k:
                break
        return hits


def build_index(
    repository_root: Path,
    embedder: Embedder,
    extra_excluded_dirs: Collection[str] = (),
) -> CodeIndex:
    """Chunk every Python file under `repository_root` and embed the chunks.

    Every chunk is guaranteed to fit the embedder's token limit (oversized ones
    are split, never silently truncated).
    """
    chunks: list[RetrievedContext] = []
    for file_path in discover_python_files(repository_root, {*EXCLUDED_DIRS, *extra_excluded_dirs}):
        for chunk in chunk_python_file(repository_root, file_path):
            chunks.extend(
                fit_to_token_budget(chunk, embedder.count_tokens, embedder.max_tokens)
            )
    logger.info("Embedding %d chunks", len(chunks))
    if not chunks:
        return CodeIndex((), np.zeros((0, 0), dtype=np.float32), embedder)
    embeddings = embedder.embed_passages([passage_text(chunk) for chunk in chunks])
    return CodeIndex(tuple(chunks), embeddings, embedder)


# --------------------------------------------------------------------------
# PR-aware retrieval
# --------------------------------------------------------------------------


def _hunk_section_heading(header: str) -> str:
    """Text after the closing `@@` -- git puts the enclosing function there."""
    return header.split("@@", 2)[2].strip() if header.count("@@") >= 2 else ""


def build_hunk_query(hunk: DiffHunk, count_tokens: Callable[[str], int], max_tokens: int) -> str:
    """Turn a hunk into a retrieval query that fits the embedder's token limit.

    The query is the changed code with diff markers stripped (a `+` prefix is
    noise to an embedding model), under the file path and git's enclosing-
    function heading. Deleted lines are left out unless the hunk is a pure
    deletion. If it's too long, context lines go first, then trailing changed
    lines: the added code is the last thing dropped.
    """
    body_lines = hunk.patch.splitlines()[1:]
    marked = [(line[:1], line[1:]) for line in body_lines if not line.startswith("\\")]
    wanted = "+" if any(mark == "+" for mark, _ in marked) else "-"
    with_context = [text for mark, text in marked if mark in (wanted, " ")]
    changed_only = [text for mark, text in marked if mark == wanted]

    heading = _hunk_section_heading(hunk.patch.splitlines()[0])
    header = f"File: {hunk.filename}" + (f"\nIn: {heading}" if heading else "")

    def fits(lines: list[str]) -> bool:
        return count_tokens(header + "\n" + "\n".join(lines)) <= max_tokens

    for lines in (with_context, changed_only):
        if fits(lines):
            return header + "\n" + "\n".join(lines)
    while changed_only and not fits(changed_only):
        _ = changed_only.pop()
    return header + "\n" + "\n".join(changed_only)


def changed_code(hunk: DiffHunk) -> list[str]:
    """Source text of the hunk's changed lines, markers removed (added, else deleted)."""
    body = [line for line in hunk.patch.splitlines()[1:] if not line.startswith("\\")]
    added = [line[1:] for line in body if line.startswith("+")]
    return added or [line[1:] for line in body if line.startswith("-")]


def referenced_definitions(
    index: CodeIndex, hunk: DiffHunk, exclude: Callable[[RetrievedContext], bool]
) -> list[int]:
    """Chunk positions defining a name the hunk calls or instantiates, calls first, then types, each in order of first use.

    Deterministic callee lookup: if the changed code calls `render_inline_comment`
    or builds a `Finding`, the reviewer should see those definitions whether or
    not an embedding model happens to link them (measured: it mostly doesn't).
    """
    code = _INTRODUCES_NAME.sub("", "\n".join(changed_code(hunk)))
    # name -> was it ever called? (dict keeps first-use order)
    uses: dict[str, bool] = {}
    for name, call in _NAME_USE.findall(code):
        # Only a call (`render(...)`) or a CamelCase type (`Finding`) is a real
        # reference. A bare lowercase name is usually a local variable: in
        # `for finding in findings`, `finding` must not pull in some unrelated
        # `def finding()` elsewhere in the repository.
        if call or name[0].isupper():
            uses[name] = uses.get(name, False) or bool(call)
    # Calls before type mentions: what the change *does* matters more than what
    # its signature mentions, and the cap on definitions is small.
    used = [n for n, called in uses.items() if called] + [n for n, called in uses.items() if not called]
    positions: list[int] = []
    for name in used:
        candidates = index.definitions.get(name, ())
        if not candidates or len(candidates) > _MAX_DEFINITIONS_PER_NAME:
            continue
        positions.extend(p for p in candidates if not exclude(index.chunks[p]) and p not in positions)
    return positions


def changed_new_lines(hunk: DiffHunk) -> frozenset[int]:
    """New-file line numbers the hunk touches (a pure deletion maps to its position)."""
    if hunk.added_lines:
        return hunk.added_lines
    _, _, new_start, _ = parse_hunk_header(hunk.patch.splitlines()[0])
    return frozenset({new_start})


def retrieve_for_hunk(
    index: CodeIndex,
    hunk: DiffHunk,
    top_k: int = DEFAULT_TOP_K,
    exclude_changed_code: bool = True,
    symmetric: bool = False,
    lookup_definitions: bool = True,
) -> list[RetrievedContext]:
    """Repository context relevant to reviewing `hunk`.

    Two sources, merged and in this order: definitions of names the changed code
    calls or instantiates (exact; capped at about half of `top_k`), then semantic
    search over the hunk's text, best match first, to fill the remaining slots.
    So the list is priority order, not purely descending by score.

    The index must be built from the PR's *head* state so chunk line numbers
    line up with the hunk's new-file line numbers. With `exclude_changed_code`
    (default) chunks overlapping the changed lines are dropped: the most
    similar chunk to a hunk is the code it modifies, and the reviewer is already
    shown that.

    `symmetric=True` embeds the query exactly like an indexed passage (no query
    instruction) -- see experiments/retrieval_eval.py for why it's not the default.
    """
    changed = changed_new_lines(hunk)

    def is_the_changed_code(chunk: RetrievedContext) -> bool:
        return chunk.filename == hunk.filename and any(
            chunk.start_line <= line <= chunk.end_line for line in changed
        )

    exclude = is_the_changed_code if exclude_changed_code else (lambda _chunk: False)
    query = build_hunk_query(hunk, index.embedder.count_tokens, index.embedder.max_tokens)
    vector = (
        index.embedder.embed_passages([query])[0]
        if symmetric
        else index.embedder.embed_query(query)
    )

    hits: list[RetrievedContext] = []
    if lookup_definitions and index.chunks:
        for position in referenced_definitions(index, hunk, exclude)[: (top_k + 1) // 2]:
            hits.append(replace(index.chunks[position], score=float(index.embeddings[position] @ vector)))

    seen = {(hit.filename, hit.start_line) for hit in hits}
    for hit in index.search(query, top_k=top_k + len(hits), exclude=exclude, query_vector=vector):
        if len(hits) == top_k:
            break
        if (hit.filename, hit.start_line) not in seen:
            hits.append(hit)
    return hits
