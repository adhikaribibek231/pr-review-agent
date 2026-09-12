from pathlib import Path

from pr_agent.retriever import chunk_python_file, discover_python_files
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

root_dir = Path(__file__).resolve().parent.parent
package_dir = root_dir/"src" /"pr_agent"
files = discover_python_files(package_dir)

chunks = []
for file in files:
    file_chunks = chunk_python_file(root_dir, file)
    chunks.extend(file_chunks)
    print(f"{file.relative_to(root_dir)}: {len(file_chunks)} chunks")

print(f"\nFiles: {len(files)}")
print(f"Chunks: {len(chunks)}")

max_limit = model.max_seq_length
for chunk in chunks:
    first = chunk.content.splitlines()[0]
    span = chunk.end_line - chunk.start_line +1
    name = chunk.filename.removeprefix("src/pr_agent/")
    #count tokens for the chunk's content
    tokens =len(model.tokenizer.encode(chunk.content, add_special_tokens=True, truncation=False))

    #Format status label
    if tokens> max_limit:
        status = f"[Exceeded: {tokens}/{max_limit}]"
    else:
        status = f"[OK: {tokens}/{max_limit}]"

    print(f"{name:<18} {chunk.start_line:>4}-{chunk.end_line:<4} ({span:>3}L) {status:<18} {first[:60]}")

"""
Measured on this repo with BAAI/bge-small-en-v1.5 (max_seq_length 512):

    chunk_python_file   39L / 473 tokens   12.1 tok/line
    _chunk_class        36L / 426 tokens   11.8 tok/line
    run_review          16L / 305 tokens   19.1 tok/line
    validate_findings    3L /  88 tokens   29.3 tok/line

Ordinary code runs 10-12 tokens/line; dense lines reach 19-29.

MAX_CHUNK_LINES was 100, which would produce ~1000-1200 token chunks --
over 2x the limit. sentence-transformers truncates silently, so those
chunks would embed only their first ~half with no error and no warning.
Lowered to 40 (512/12, and it fits the longest chunk here at 473).

Caveat: a line cap is a proxy for a token cap. validate_findings is 3
lines and 88 tokens because it is one long set comprehension -- a file
written in that style could still overflow at 40 lines. Capping on
tokens directly would be correct but couples the chunker to a
tokenizer. Revisit if retrieval quality looks bad on dense code.

Re-measure if the embedding model changes: a 8k-context model makes
this cap far too conservative.
"""
