"""Measure retrieval quality instead of eyeballing it.

    python experiments/retrieval_eval.py            # whole repo, real model
    python experiments/retrieval_eval.py -v         # also print every miss

Two suites over this repository. tests/ is indexed and acts as a realistic
distractor (it repeats src/'s vocabulary); experiments/ is excluded because it
contains these very queries, which would leak the answers into the corpus:

  queries  natural-language question  -> the function that answers it
  hunks    a changed hunk             -> code the reviewer should also see

Expectations name a (file, symbol), not line numbers, so they survive edits.
A result is a hit if any returned chunk overlaps the symbol's line span.

Metrics: Recall@k = share of cases with a hit in the top k;
MRR = mean of 1/rank of the first hit (0 if none in the top 5).
Use these numbers to decide whether a retrieval change helped.
"""

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pr_agent.diff_parser import DiffHunk, parse_hunks  # noqa: E402
from pr_agent.embedding import BGE_QUERY_INSTRUCTION, SentenceTransformerEmbedder  # noqa: E402
from pr_agent.models import RetrievedContext  # noqa: E402
from pr_agent.search import CodeIndex, build_index, retrieve_for_hunk  # noqa: E402

TOP_K = 5

# (file, symbol); "Class.method" for methods.
Target = tuple[str, str]

QUERY_CASES: list[tuple[str, list[Target]]] = [
    ("Where are invalid review findings filtered out?", [("src/pr_agent/validator.py", "validate_findings")]),
    ("Where are review comments posted to GitHub?", [("src/pr_agent/github_client.py", "post_review")]),
    ("How are Python files divided into chunks?", [("src/pr_agent/retriever.py", "chunk_python_file")]),
    ("Where are pull-request hunks built from changed files?", [("src/pr_agent/pipeline.py", "build_hunks")]),
    ("How are added and deleted line numbers calculated?", [("src/pr_agent/diff_parser.py", "parse_changed_lines")]),
    ("How is a hunk header like @@ -1,3 +1,4 @@ parsed?", [("src/pr_agent/diff_parser.py", "parse_hunk_header")]),
    ("Which files are skipped as lockfiles or noise?", [("src/pr_agent/diff_parser.py", "is_noise_file")]),
    ("How do we fetch the files changed in a pull request?", [("src/pr_agent/github_client.py", "fetch_changed_files")]),
    ("Where are findings sorted by severity?", [("src/pr_agent/pipeline.py", "aggregate_findings")]),
    ("What turns a finding into markdown text for a comment?", [("src/pr_agent/renderer.py", "render_inline_comment")]),
    ("How are virtualenv and cache directories excluded when scanning for source files?", [("src/pr_agent/retriever.py", "discover_python_files")]),
    ("How are very long classes split up for indexing?", [("src/pr_agent/retriever.py", "_chunk_class")]),
    ("How do we authenticate to GitHub with an access token?", [("src/pr_agent/github_client.py", "get_github_client")]),
    ("Why do chunks start at the decorator line instead of the def line?", [("src/pr_agent/retriever.py", "_start_line")]),
    ("What data structure represents an issue the reviewer found?", [("src/pr_agent/models.py", "Finding")]),
    ("How is an oversized chunk split to fit the embedding model's token limit?", [("src/pr_agent/retriever.py", "fit_to_token_budget")]),
]

# (file, symbol being "changed", related code a reviewer should be shown)
HUNK_CASES: list[tuple[Target, list[Target]]] = [
    (("src/pr_agent/github_client.py", "post_review"),
     [("src/pr_agent/renderer.py", "render_inline_comment"), ("src/pr_agent/models.py", "Finding")]),
    (("src/pr_agent/pipeline.py", "run_review"),
     [("src/pr_agent/github_client.py", "fetch_changed_files"), ("src/pr_agent/pipeline.py", "build_hunks"),
      ("src/pr_agent/validator.py", "validate_findings"), ("src/pr_agent/pipeline.py", "aggregate_findings"),
      ("src/pr_agent/github_client.py", "post_review")]),
    (("src/pr_agent/validator.py", "validate_findings"),
     [("src/pr_agent/diff_parser.py", "DiffHunk"), ("src/pr_agent/models.py", "Finding")]),
    (("src/pr_agent/pipeline.py", "build_hunks"),
     [("src/pr_agent/diff_parser.py", "parse_hunks"), ("src/pr_agent/diff_parser.py", "is_noise_file")]),
    (("src/pr_agent/github_client.py", "fetch_changed_files"),
     [("src/pr_agent/models.py", "ChangedFile")]),
    (("src/pr_agent/diff_parser.py", "parse_hunks"),
     [("src/pr_agent/diff_parser.py", "split_hunks"), ("src/pr_agent/diff_parser.py", "parse_changed_lines"),
      ("src/pr_agent/diff_parser.py", "DiffHunk")]),
]


def symbol_span(root: Path, target: Target) -> tuple[str, int, int]:
    filename, symbol = target
    tree = ast.parse((root / filename).read_text(encoding="utf-8"))
    scope: list[ast.stmt] = tree.body
    node: ast.stmt | None = None
    for part in symbol.split("."):
        node = next(
            (n for n in scope
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part),
            None,
        )
        if node is None:
            raise LookupError(f"{symbol!r} not found in {filename}; update the eval case")
        scope = node.body
    assert node is not None and node.end_lineno is not None
    return filename, node.lineno, node.end_lineno


def first_hit_rank(hits: list[RetrievedContext], spans: list[tuple[str, int, int]]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        for filename, start, end in spans:
            if hit.filename == filename and hit.start_line <= end and hit.end_line >= start:
                return rank
    return None


def hunk_for(root: Path, target: Target) -> DiffHunk:
    """Pretend `target` was just added: a hunk whose + lines are its current source."""
    filename, start, end = symbol_span(root, target)
    lines = (root / filename).read_text(encoding="utf-8").split("\n")[start - 1 : end]
    patch = f"@@ -{start - 1},0 +{start},{len(lines)} @@\n" + "\n".join(f"+{line}" for line in lines)
    return parse_hunks(filename, patch)[0]


@dataclass
class Score:
    name: str
    ranks: list[int | None]

    def line(self) -> str:
        n = len(self.ranks)
        recall = lambda k: sum(1 for r in self.ranks if r is not None and r <= k) / n  # noqa: E731
        mrr = sum(1 / r for r in self.ranks if r is not None) / n
        return f"{self.name:<34} R@1={recall(1):.2f}  R@3={recall(3):.2f}  R@5={recall(5):.2f}  MRR={mrr:.3f}  (n={n})"


def run_queries(index: CodeIndex, root: Path, name: str, verbose: bool) -> Score:
    ranks = []
    for query, targets in QUERY_CASES:
        spans = [symbol_span(root, t) for t in targets]
        hits = index.search(query, top_k=TOP_K)
        rank = first_hit_rank(hits, spans)
        ranks.append(rank)
        if verbose and rank != 1:
            got = ", ".join(f"{h.filename.split('/')[-1]}:{h.start_line}" for h in hits[:3])
            print(f"  [{name}] rank={rank} {query!r}\n      wanted {targets[0][1]}; got {got}")
    return Score(name, ranks)


def run_hunks(index: CodeIndex, root: Path, name: str, verbose: bool, **options: bool) -> Score:
    ranks = []
    for changed, related in HUNK_CASES:
        hunk = hunk_for(root, changed)
        spans = [symbol_span(root, t) for t in related]
        hits = retrieve_for_hunk(index, hunk, top_k=TOP_K, **options)
        rank = first_hit_rank(hits, spans)
        ranks.append(rank)
        if verbose and rank != 1:
            got = ", ".join(f"{h.filename.split('/')[-1]}:{h.start_line}" for h in hits[:3])
            print(f"  [{name}] rank={rank} changed={changed[1]}; wanted any of {[t[1] for t in related]}; got {got}")
    return Score(name, ranks)


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("-v", "--verbose", action="store_true", help="print non-rank-1 cases")
    verbose = parser.parse_args().verbose

    embedder = SentenceTransformerEmbedder()
    index = build_index(PROJECT_ROOT, embedder, extra_excluded_dirs={"experiments"})
    print(f"Indexed {len(index.chunks)} chunks from {len({c.filename for c in index.chunks})} files\n")

    print("== natural-language query -> function ==")
    embedder.query_instruction = ""
    print(run_queries(index, PROJECT_ROOT, "no query prefix", verbose).line())
    embedder.query_instruction = BGE_QUERY_INSTRUCTION
    print(run_queries(index, PROJECT_ROOT, "BGE query prefix", verbose).line())

    print("\n== changed hunk -> related code (changed code excluded) ==")
    embedder.query_instruction = ""
    print(run_hunks(index, PROJECT_ROOT, "semantic only", verbose, lookup_definitions=False).line())
    print(run_hunks(index, PROJECT_ROOT, "semantic + definition lookup", verbose).line())
    print("-- variants of the semantic part (definition lookup on) --")
    print(run_hunks(index, PROJECT_ROOT, "asymmetric, no prefix", verbose).line())
    embedder.query_instruction = BGE_QUERY_INSTRUCTION
    print(run_hunks(index, PROJECT_ROOT, "asymmetric, BGE prefix", verbose).line())
    print(run_hunks(index, PROJECT_ROOT, "symmetric (query as passage)", verbose, symmetric=True).line())
    print(run_hunks(index, PROJECT_ROOT, "symmetric, changed code NOT excluded", verbose,
                    symmetric=True, exclude_changed_code=False).line())


if __name__ == "__main__":
    main()
