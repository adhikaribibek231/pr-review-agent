You're at a point where you need a small, achievable coding task, not another big architecture plan.

I suggest we work on this project incrementally. For each feature, you should understand why you're building it, how to implement it, how to test it, and what improvement comes next.

Right now, I would not change your existing architecture.

# Your first task: Make your code chunks searchable

Where you currently are

Chunking completed

Python source files

AST chunking + line metadata

Your next task: Embeddings + Search

Turn chunks into vectors and retrieve relevant code

Later: Connect retrieval to PR diffs

Your `retriever.py` already discovers Python files, uses AST to create chunks, and stores their content and line numbers.&#x20;

retriever.py



But those chunks aren't doing anything useful yet.

You need to make it possible to ask something like:

> Where in the repository are invalid review findings filtered out?

And have your system retrieve the relevant function:

```
def validate_findings(findings, hunks):    ...
```

This is called semantic search, and it is the first working component of your RAG system.

## Step 1 — Understand what you're implementing

You already know how to create chunks. Now imagine you have three of them:

Repository chunks

Chunk A

GitHub API

Chunk B

Diff parsing

Chunk C

Validation

Embedding model

Convert each chunk into a numerical vector

[0.12, -0.37, 0.81, ...]

Compare the query vector against the chunk vectors

Rank chunks by similarity

Retrieve the top results

Most similar code chunks

Return the source code, filename and line ranges

That's all we're implementing for now. No LLM calls, no LangGraph, no vector database.

## Step 2 — Write your first search experiment

Create a new file:

`experiments/search_test.py`

Put this code inside:

```
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")    # Step 1: Discover and chunk Python files    files = discover_python_files(package_dir)    chunks = []    for file in files:        chunks.extend(chunk_python_file(root_dir, file))    print(f"Found {len(chunks)} chunks")    # Step 2: Prepare passages for embedding    valid_chunks = []    passages = []    for chunk in chunks:        text = f"File: {chunk.filename}\n{chunk.content}"        token_count = len(            model.tokenizer.encode(                text,                add_special_tokens=True,                truncation=False,            )        )        if token_count > model.max_seq_length:            print(f"Skipping oversized chunk: {chunk.filename}")            continue        valid_chunks.append(chunk)        passages.append(text)    if not passages:        raise RuntimeError("No valid chunks to index")    # Step 3: Create embeddings    embeddings = model.encode(        passages,        normalize_embeddings=True,        show_progress_bar=True,    )    print("Embeddings shape:", embeddings.shape)    # Step 4: Query the repository    query = "Where are invalid review findings filtered out?"    query_embedding = model.encode(        [query],        normalize_embeddings=True,    )[0]    # Step 5: Compute similarity    scores = embeddings @ query_embedding    # Step 6: Retrieve top 5 chunks    top_indices = scores.argsort()[::-1][:5]    for index in top_indices:        chunk = valid_chunks[index]        print("\n---------------------------")        print(f"Score: {scores[index]:.4f}")        print(f"File: {chunk.filename}")        print(f"Lines: {chunk.start_line}-{chunk.end_line}")        print(chunk.content[:500])if __name__ == "__main__":    main()
```

Run this from your project's root directory in your activated virtual environment:

```
python experiments/search_test.py
```

This uses the same embedding model you already experimented with. Sentence Transformers supports this normalized-embedding and similarity-search workflow.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face

+1



Notice that I temporarily skip chunks that exceed the model's token limit. This avoids silent truncation during this experiment, but skipping chunks is not the final solution. We'll fix that afterward.

## Step 3 — Check whether it works

The first question is:

`Where are invalid review findings filtered out?`

A relevant result should be `validator.py`, because that file implements `validate_findings()`.

Don't expect a particular similarity score. What matters initially is whether the right code ranks highly.

Then try these other queries:

| Query                                              | Expected relevant file |
| -------------------------------------------------- | ---------------------- |
| Where are GitHub PR review comments posted?        | `github_client.py`     |
| How does the agent split Python files into chunks? | `retriever.py`         |
| Where are pull-request hunks built?                | `pipeline.py`          |
| How are added and deleted line numbers calculated? | `diff_parser.py`       |

This is your first retrieval evaluation dataset, even though it's a tiny one.

If the expected function consistently appears among the first few results, that's encouraging. If it doesn't, you have something concrete to investigate.

## Step 4 — What improvements should you make?

Once the first experiment works, I'd prioritize these improvements in this order.

1\. Make chunking token-aware

You've already discovered that 40 lines can sometimes exceed 512 tokens. Improve `_make_context()` so oversized chunks are subdivided until the actual embedding input fits the token limit.

This is more robust than relying on `MAX_CHUNK_LINES` alone.

2\. Test retrieval quality

Instead of manually looking at one result, define 10–20 queries with known relevant functions. Measure how often the expected function appears in the top five.

This will tell you whether changes to chunking or embeddings actually help.

3\. Turn the experiment into reusable functions

You'll eventually want to call something like:

```
index = build_index(repo_path)results = search(index, query, top_k=5)
```

Keep embedding generation separate from search so you don't recreate embeddings for every query.

4\. Connect retrieval to the PR diff

Currently you're manually writing questions. Instead, use each `DiffHunk` to construct the search query and retrieve related repository code.

This is what turns semantic search into the beginning of a PR-aware RAG system.

There are more advanced improvements worth exploring later, such as hybrid keyword/vector search, class and method metadata in embeddings, reranking, and persistent indexes. But implementing them now would make it harder to understand whether your basic retriever works.

## Your short-term goal

I would split your next work into three small milestones.

| Milestone | What you build                     | When it's done                                                              |
| --------- | ---------------------------------- | --------------------------------------------------------------------------- |
| 1         | Working semantic search experiment | You retrieve relevant functions for test queries                            |
| 2         | Reliable retriever                 | You handle oversized chunks, add tests and expose reusable search functions |
| 3         | PR-aware retrieval                 | A changed hunk automatically retrieves useful repository context            |

Only after milestone 3 would I start your real LLM reviewer.

For now, do just Step 2: create `experiments/search_test.py` and run the search.

Don't worry about improving all your files. Look at the top five results and ask yourself: Did my retriever find the code I expected?

That one experiment will give you a concrete starting point. From its output, we can determine whether chunking, the query format, or the embedding approach needs improvement before integrating it with your review pipeline.
