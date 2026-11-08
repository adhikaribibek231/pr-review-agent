from pr_agent.models import RetrievedContext
from pr_agent.retriever import fit_to_token_budget, passage_text


def words(text: str) -> int:
    """Stand-in tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def make_chunk(content: str, start_line: int = 1, filename: str = "a.py") -> RetrievedContext:
    end_line = start_line + content.count("\n")
    return RetrievedContext(filename, start_line, end_line, content, 0.0)


def lines_of(chunks: list[RetrievedContext]) -> list[int]:
    return [n for chunk in chunks for n in range(chunk.start_line, chunk.end_line + 1)]


# --------------------------------------------------------------------------


def test_chunk_within_budget_is_returned_unchanged() -> None:
    chunk = make_chunk("def f():\n    return 1")
    assert fit_to_token_budget(chunk, words, 100) == [chunk]


def test_oversized_chunk_is_split_into_pieces_that_all_fit() -> None:
    content = "\n".join(f"value_{i} = compute_something({i}, extra, args)" for i in range(30))
    budget = 40
    pieces = fit_to_token_budget(make_chunk(content, start_line=10), words, budget)

    assert len(pieces) > 1
    for piece in pieces:
        assert words(passage_text(piece)) <= budget


def test_split_covers_every_line_exactly_once_with_exact_line_numbers() -> None:
    source_lines = [f"line_{i} = {i}" for i in range(25)]
    chunk = make_chunk("\n".join(source_lines), start_line=100)
    pieces = fit_to_token_budget(chunk, words, 20)

    assert lines_of(pieces) == list(range(100, 125))
    for piece in pieces:
        expected = source_lines[piece.start_line - 100 : piece.end_line - 100 + 1]
        assert piece.content == "\n".join(expected)


def test_split_prefers_a_blank_line_near_the_middle() -> None:
    first = "\n".join(f"first_{i} = 1" for i in range(5))
    second = "\n".join(f"second_{i} = 2" for i in range(5))
    chunk = make_chunk(f"{first}\n\n{second}")
    pieces = fit_to_token_budget(chunk, words, 30)

    assert len(pieces) == 2
    assert all("second_" not in piece.content for piece in pieces[:1])
    assert all("first_" not in piece.content for piece in pieces[1:])


def test_single_line_over_budget_is_truncated_not_dropped() -> None:
    chunk = make_chunk("data = [" + " ".join(f"item{i}" for i in range(200)) + "]")
    pieces = fit_to_token_budget(chunk, words, 30)

    assert len(pieces) == 1
    assert words(passage_text(pieces[0])) <= 30
    assert pieces[0].start_line == pieces[0].end_line == chunk.start_line


def test_splitting_never_emits_whitespace_only_pieces() -> None:
    chunk = make_chunk("alpha beta gamma\n\n\n\n\n\n\n\n\n\ndelta epsilon zeta")
    pieces = fit_to_token_budget(chunk, words, 6)
    assert pieces
    assert all(piece.content.strip() for piece in pieces)
