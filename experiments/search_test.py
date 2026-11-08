"""First semantic-search experiment for the PR Review Agent.

Run from your project root with: python experiments/search_test.py
Required: uv add sentence-transformers
"""

from pathlib import Path
import sys

#resolve paths relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PACKAGE_DIR = SRC_DIR / "pr_agent"

if not PACKAGE_DIR.is_dir():
    raise FileNotFoundError(
            f"Expected the agent's Python files at: {PACKAGE_DIR}\n"+
            "If your repository uses a different layout, update SRC_DIR and PACKAGE_DIR."
            )

#supports repositories that put the python package
sys.path.insert(0,str(SRC_DIR))

from pr_agent.retriever import chunk_python_file, discover_python_files
from sentence_transformers import SentenceTransformer

def main()->None:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    max_seq_length = model.max_seq_length
    print(f"Model token limit: {max_seq_length}")

    files = list(discover_python_files(PACKAGE_DIR))
    chunks = []
    for file in files:
        chunks.extend(chunk_python_file(PROJECT_ROOT,file))
    print(f"Found {len(files)} files and {len(chunks)} code chunks")

    valid_chunks = []
    passages = []
    oversized = []
    for chunk in chunks:
        passage = f"File: {chunk.filename}\n {chunk.content}"
        token_count = len(
                model.tokenizer.encode(
                    passage,
                    add_special_tokens = True,
                    truncation = False,
                    )
                )
        if max_seq_length is None:
            raise RuntimeError("The model's maximum sequence length is not set")
        if token_count > max_seq_length:
            oversized.append((chunk.filename, chunk.start_line, token_count))
            continue
        valid_chunks.append(chunk)
        passages.append(passage)

    print(f"Indexing {len(valid_chunks)} chunks; skipped {len(oversized)} oversized chunks")
    for filename, start_line, count in oversized[:5]:
        print(f"Skipped {filename}:{start_line} ({count} tokens)")
        
    if not passages:
        raise RuntimeError("No chunks fit the embedding model's token limit")
    
    embeddings = model.encode(passages, convert_to_numpy = True, normalize_embeddings = True, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    test_cases = [
            {
                "query": "Where are invalid review findings filtered out?",
                "expected_file":"validator.py",
                },
            {
                "query":"Where are review comments posted to GitHub?",
                "expected_file": "github_client.py",
                },
            {
                "query":"How are Python files divided into chunks?",
                "expected_file":"retriever.py",
                },
            ]
    for case in test_cases:
        query, expected_file = case.values()
        query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True,)[0]
        scores = embeddings @query_embedding
        top_indices = scores.argsort()[::-1][:5]
        got_expected = False
        print("################################################################")
        print('------------------------------------------------------------')
        print(f"\nQuestion: {query}")
        for rank, index in enumerate(top_indices, start=1):
            chunk = valid_chunks[index]
            print("\n"+"-"*55)
            print(f"Rank: {rank} | Similarity: {scores[index]:.4f}")
            print(f"File: {chunk.filename}")
            print(f"Lines: {chunk.start_line}-{chunk.end_line}")
            print(chunk.content[:500])
            got_expected = got_expected or (chunk.filename.endswith(expected_file))
        print('------------------------------------------------------------')
        if got_expected:
            print(f"GOT EXPECTED? = YES. GOT THE EXPECTED FILE -- {expected_file}")
        else:
            print(f"DIDN'T GET THE EXPECTED FILE {expected_file}  :(")
        print("################################################################")
        

if __name__ =="__main__":
    main()
