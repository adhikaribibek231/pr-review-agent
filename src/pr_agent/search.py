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
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pr_agent.diff_parser import DiffHunk, changed_code, changed_new_lines
from pr_agent.embedding import Embedder
from pr_agent.impact import analyze_impact
from pr_agent.models import RetrievedContext
from pr_agent.symbols import SymbolIndex
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
    symbols: SymbolIndex = field(default_factory=SymbolIndex)  # who defines and uses what

    @cached_property
    def _position(self) -> dict[tuple[str, int], int]:
        return {(chunk.filename, chunk.start_line): i for i, chunk in enumerate(self.chunks)}

    @cached_property
    def _chunks_by_file(self) -> dict[str, list[int]]:
        by_file: dict[str, list[int]] = {}
        for i, chunk in enumerate(self.chunks):
            by_file.setdefault(chunk.filename, []).append(i)
        return by_file

    def chunk_position_at(self, filename: str, line: int) -> int | None:
        """Position of the chunk holding `line`; the narrowest one if windows overlap."""
        covering = [i for i in self._chunks_by_file.get(filename, [])
                    if self.chunks[i].start_line <= line <= self.chunks[i].end_line]
        if not covering:
            return None
        return min(covering, key=lambda i: self.chunks[i].end_line - self.chunks[i].start_line)

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
    files = discover_python_files(repository_root, {*EXCLUDED_DIRS, *extra_excluded_dirs})
    symbols = SymbolIndex.build(repository_root, files)
    for file_path in files:
        for chunk in chunk_python_file(repository_root, file_path):
            chunks.extend(
                fit_to_token_budget(chunk, embedder.count_tokens, embedder.max_tokens)
            )
    logger.info("Embedding %d chunks", len(chunks))
    if not chunks:
        return CodeIndex((), np.zeros((0, 0), dtype=np.float32), embedder, symbols)
    embeddings = embedder.embed_passages([passage_text(chunk) for chunk in chunks])
    return CodeIndex(tuple(chunks), embeddings, embedder, symbols)


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


def _referenced_definitions_named(
    index: CodeIndex, hunk: DiffHunk, exclude: Callable[[RetrievedContext], bool]
) -> list[tuple[int, str, bool]]:
    """(chunk position, name, was it called?) for definitions the hunk calls or instantiates.

    Calls first, then type mentions, each in order of first use. Deterministic callee
    lookup: if the changed code calls `render_inline_comment` or builds a `Finding`,
    the reviewer should see those definitions whether or not an embedding model
    happens to link them (measured: it mostly doesn't).
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
    ordered = [n for n, called in uses.items() if called] + [n for n, called in uses.items() if not called]
    found: list[tuple[int, str, bool]] = []
    taken: set[int] = set()
    for name in ordered:
        candidates = index.definitions.get(name, ())
        if not candidates or len(candidates) > _MAX_DEFINITIONS_PER_NAME:
            continue
        for position in candidates:
            if position not in taken and not exclude(index.chunks[position]):
                taken.add(position)
                found.append((position, name, uses[name]))
    return found


def referenced_definitions(
    index: CodeIndex, hunk: DiffHunk, exclude: Callable[[RetrievedContext], bool]
) -> list[int]:
    """Chunk positions defining a name the hunk calls or instantiates (calls first)."""
    return [position for position, _name, _called in _referenced_definitions_named(index, hunk, exclude)]


def _imports_only(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return bool(lines) and all(
        line.startswith("#") or line.startswith(("import ", "from ")) or line in (")", "(") or line.endswith(",")
        for line in lines
    ) and any(line.startswith(("import ", "from ")) for line in lines)


@dataclass(frozen=True)
class Retrieval:
    """What the reviewer is given for one hunk."""

    context: tuple[RetrievedContext, ...]
    notes: tuple[str, ...]


def retrieve(
    index: CodeIndex,
    hunk: DiffHunk,
    pr_hunks: list[DiffHunk] | None = None,
    top_k: int = DEFAULT_TOP_K,
    exclude_changed_code: bool = True,
    symmetric: bool = False,
    lookup_definitions: bool = True,
    analyze_dependents: bool = True,
) -> Retrieval:
    """Everything worth showing a reviewer of `hunk`, in priority order:

      1. DEPENDENTS: code outside the diff that this change could break (callers,
         instantiations, subclasses, dangling references...), worst first, each with the
         reason it is shown. Needs `pr_hunks`, every hunk in the PR, to tell callers the
         PR already updated from the ones it left untouched.
      2. DEFINITIONS of the functions and types the changed code uses.
      3. SEMANTIC matches for the changed code, to fill the remaining `top_k` slots.

    The index must be built from the PR's *head* state so chunk line numbers line up
    with the hunk's. With `exclude_changed_code` (default) chunks overlapping the changed
    lines are dropped from (2) and (3): the most similar chunk to a hunk is the code it
    modifies, and the reviewer is already shown that.
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
    ) if index.chunks else None

    def with_score(position: int, reason: str) -> RetrievedContext:
        assert vector is not None
        return replace(index.chunks[position], score=float(index.embeddings[position] @ vector), reason=reason)

    hits: list[RetrievedContext] = []
    notes: tuple[str, ...] = ()
    taken: set[int] = set()

    if analyze_dependents and index.chunks:
        report = analyze_impact(index.symbols, hunk, pr_hunks or [hunk])
        notes = report.notes
        for item in report.items:
            position = index.chunk_position_at(item.file, item.line)
            if position is not None and position not in taken:
                taken.add(position)
                hits.append(with_score(position, item.reason))

    callees: list[RetrievedContext] = []
    if lookup_definitions and index.chunks:
        for position, name, called in _referenced_definitions_named(index, hunk, exclude)[: (top_k + 1) // 2]:
            if position not in taken:
                taken.add(position)
                verb = "calls" if called else "mentions"
                callees.append(with_score(position, f"defines `{name}`, which the changed code {verb}"))
    hits += callees

    semantic = index.search(query, top_k=top_k + len(taken), exclude=exclude, query_vector=vector)
    added = len(callees)
    for hit in semantic:
        if added >= top_k:
            break
        position = index._position.get((hit.filename, hit.start_line))
        if position is not None and position in taken:
            continue
        if _imports_only(hit.content):
            continue  # a block of imports says which modules exist, not what the code does
        hits.append(replace(hit, reason="semantically similar to the changed code"))
        added += 1
    return Retrieval(tuple(hits), notes)


def retrieve_for_hunk(
    index: CodeIndex,
    hunk: DiffHunk,
    top_k: int = DEFAULT_TOP_K,
    exclude_changed_code: bool = True,
    symmetric: bool = False,
    lookup_definitions: bool = True,
    pr_hunks: list[DiffHunk] | None = None,
    analyze_dependents: bool = True,
) -> list[RetrievedContext]:
    """Just the context chunks of `retrieve`. Kept for callers that don't need the notes."""
    return list(retrieve(index, hunk, pr_hunks, top_k, exclude_changed_code, symmetric,
                         lookup_definitions, analyze_dependents).context)
