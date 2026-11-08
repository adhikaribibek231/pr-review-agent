# PR Review Agent — Project Review & Roadmap

My assessment: You've already built a meaningful foundation. You're currently at the transition between building the code-review infrastructure and implementing the actual AI-powered retrieval system.

The project is worth continuing. Your overall architecture makes sense, and the work you've done on parsing diffs, tracking changed lines, and chunking Python code is relevant to the final goal.

However, you do not have a functioning AI code reviewer yet. You have much of the infrastructure that will allow one to work.

I'll walk through what the project does, what you've completed, what needs improvement, and the order I'd recommend implementing everything else.

## 1. What exactly are you building?

Imagine a developer opens a pull request on GitHub.

They've changed a function that handles user authentication. The changed code looks reasonable, but the developer has accidentally removed a permission check.

A normal AI reviewer might only look at the changed function. Without understanding the rest of the repository, it could miss the bug.

Your agent is supposed to do something smarter.

Developer opens a GitHub pull request

The agent receives the changed files and diff

Understand changes

Which functions and lines were modified?

Retrieve context

Find related functions elsewhere in the repo

LLM reviews changes using retrieved context

Generate candidate bug and security findings

Validate findings

Check locations, evidence, and unsupported claims

Post grounded review comments on GitHub

Only accepted findings are published

The central idea is:

Diff tells the agent what changed. RAG tells it what related code exists. The LLM reasons about potential problems. Validation controls what gets published.

That's what distinguishes your project from a simple script that sends a Git diff to an LLM.

## 2. What you've already implemented

I reviewed your seven uploaded Python files and the embedding experiment you shared.

| Component               | Current status                       | Assessment                                             |
| ----------------------- | ------------------------------------ | ------------------------------------------------------ |
| GitHub API client       | Implemented, not verified end-to-end | Fetches PRs and changed files; publishing logic exists |
| Diff parser             | Implemented                          | Extracts hunks and tracks added/deleted lines          |
| Noise filtering         | Basic implementation                 | Skips common lockfiles                                 |
| Review pipeline         | Skeleton implemented                 | Connects fetching, parsing, validation and publishing  |
| Finding validator       | Basic implementation                 | Checks that comments target added lines                |
| Comment renderer        | Implemented                          | Formats findings as Markdown                           |
| Python code chunker     | Implemented                          | AST-based chunks with line metadata                    |
| Embedding experiments   | In progress                          | Model loaded and token lengths measured                |
| Semantic vector search  | Not yet implemented                  | Next major task                                        |
| Cross-encoder reranking | Not implemented                      | Later retrieval improvement                            |
| Real LLM reviewer       | Not implemented                      | Currently using fake findings                          |
| LangGraph               | Not implemented                      | Future orchestration                                   |
| GitHub Actions          | Not shown                            | Future automation                                      |
| Evaluation framework    | Not shown                            | Needed to measure accuracy                             |

Statuses refer to the code and descriptions you shared, not a verified test run of your entire repository.

### The most important thing you've completed

Your `retriever.py` is further along than a basic file splitter.

You've already handled several useful problems:

- Parsing Python with `ast` instead of blindly splitting every N lines.
- Keeping function and class definitions together where possible.
- Handling oversized functions with overlapping windows.
- Including decorators and top-level imports/constants.
- Preserving filename and line-number metadata.
- Skipping excluded directories and files that cannot be parsed.

Your latest experiment also identified a real problem: an embedding model with a 512-token limit can silently truncate code chunks.

Reducing `MAX_CHUNK_LINES` from 100 to 40 was a reasonable temporary solution based on your measured token counts.

That experiment is effectively where you left off.

### What your pipeline currently does

Looking at `pipeline.py`, your current flow is:

```
Fetch pull request                 DONE
       ↓
Fetch changed files                DONE
       ↓
Parse changed hunks                DONE
       ↓
Generate fake findings             PLACEHOLDER
       ↓
Validate changed-line locations    DONE (basic)
       ↓
Sort findings                      DONE
       ↓
Optionally publish to GitHub       IMPLEMENTED, needs testing
```

The critical detail is `generate_fake_findings()`.

It currently returns two hardcoded findings for `main.py`, one referencing line 36 and another referencing line 999. It does not inspect the hunks to discover defects.

That's intentional scaffolding, but it means the reviewer cannot yet find actual bugs.&#x20;

pipeline.py



So your immediate goal is not to rebuild the pipeline. It's to replace its placeholders, one step at a time.

## 3. Is your current implementation good enough?

For the stage you're at: yes. For the finished project: no, and that's expected.

I also checked that all seven uploaded Python files parse successfully and ran a small smoke test of chunking and changed-line validation. Those checks passed. This is not a replacement for running your complete test suite.

There are a few things I'd improve, but you don't need to rewrite the project.

### What I would keep

Your choice of Python AST for chunking, explicit `DiffHunk` objects, immutable dataclasses, a separate validator, and a dry-run posting flag are all reasonable foundations.

In particular, preserving line numbers throughout retrieval is important because the final output must reference real GitHub diff lines.

### What I would improve

| Priority | Component           | Issue and recommended change                                                                                                                   |
| -------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| High     | `retriever.py`      | You can generate chunks, but cannot search them yet. Build vector retrieval next.                                                              |
| High     | `validator.py`      | It verifies only filename and added-line location. It does not verify whether an alleged bug is real. Add evidence-based checks later.         |
| High     | Embedding chunks    | The 40-line limit does not guarantee every chunk fits 512 tokens. Add tokenizer-aware splitting before indexing.                               |
| Medium   | `pipeline.py`       | `aggregate_findings()` only sorts results; it doesn't deduplicate them.                                                                        |
| Medium   | `pipeline.py`       | `triage_findings()` checks whether findings exist. It doesn't actually triage changed hunks before review.                                     |
| Medium   | `github_client.py`  | Test posting on real PRs, especially forked PRs. Resolving the head SHA through the base repository may be fragile.                            |
| Medium   | Repository indexing | Your experiment indexes only `src/pr_agent`, whereas a general-purpose reviewer will need to index relevant code across the target repository. |

The distinction in `validator.py` is especially important. It currently accepts a finding whenever `(filename, line)` matches an added line. That proves the comment points somewhere valid; it does not prove the finding's message is correct.&#x20;

validator.py



Your next version should continue doing this deterministic location validation while adding stronger evidence checks.

For publishing, PyGithub supports constructing inline comments with `line` and `side`, and the review creation API permits an optional commit argument. I'd test whether omitting that argument is simpler than resolving the PR head through the base repository.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 4. What you should do next — right now

Next milestone

Repository semantic search

## Turn your code chunks into a working RAG retriever

You were in the middle of your retrieval implementation. I would pick up from exactly there.

Right now you have:

```
Python source files
       ↓
AST-based chunking
       ↓
Chunks with filename, line range and content
       ↓
Embedding token-length checks
       ↓
[CURRENT STOPPING POINT]
```

What comes next is:

```
Chunks
   ↓
Create vector embeddings
   ↓
Store vectors + chunk metadata
   ↓
Accept a search query
   ↓
Embed the query
   ↓
Calculate similarity
   ↓
Return top-k relevant code chunks
```

### Step A — Create embeddings

You've already initialized the model.

The next experiment can be very small. Using the `chunks` and `model` variables from your current script, try this:

```
import numpy as np# Prepare passages for embeddingtexts = [    f"{chunk.filename}\n{chunk.content}"    for chunk in chunks]# Encode and normalizeembeddings = model.encode(    texts,    normalize_embeddings=True,    show_progress_bar=True,)# Search the codebasequery = "Where are GitHub pull request review comments posted?"query_embedding = model.encode(    [query],    normalize_embeddings=True,)[0]# Similarity scoresscores = embeddings @ query_embedding# Retrieve the five closest chunkstop_indices = np.argsort(scores)[::-1][:5]for index in top_indices:    chunk = chunks[index]    print(f"\nScore: {scores[index]:.4f}")    print(f"File: {chunk.filename}")    print(f"Lines: {chunk.start_line}-{chunk.end_line}")    print(chunk.content)
```

With normalized embeddings, the dot product gives cosine similarity. This approach is also consistent with the BGE model's documented retrieval usage.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

BAAI/bge-small-en at main



This is enough for an initial experiment. You do not need a dedicated vector database yet.

For your current repository size, even a plain NumPy-based search is sufficient to prove that retrieval works.

### Step B — Test retrieval quality

Try questions like:

- "Where are changed lines extracted from a Git diff?"
- "Which function publishes review comments?"
- "Where are invalid findings discarded?"
- "How does the project split large Python classes into chunks?"

For each query, manually examine the top five results.

Check whether the correct function appears near the top, whether the chunk contains enough context to understand the function, and whether unrelated chunks consistently outrank relevant ones.

This is important because embeddings existing does not mean retrieval is useful.

You should also test a simple filename/keyword search baseline. For code, exact function names and imports can sometimes retrieve better results than semantic similarity.

### Step C — Make the chunker tokenizer-safe

Your decision to lower the line cap to 40 was reasonable, but now that the embedding model is becoming part of the actual system, I'd add a stronger guarantee.

The chunker should ensure that every passage sent to the model fits its token budget, splitting an oversized chunk further when necessary.

I would keep the AST-based initial grouping and use token-aware splitting only as a fallback. That preserves meaningful function boundaries as much as possible.

You can leave some headroom below 512 tokens for future formatting, and avoid assuming that every 40-line function fits.

### Step D — Turn the experiment into actual retriever functions

Eventually, you want a small interface resembling:

```
index = build_index(repository_root, model)results = search(    index=index,    query="Where is the review published?",    top_k=5,)
```

Each result should provide the existing `RetrievedContext` information:

```
filename
start_line
end_line
content
score
```

This is already a good data structure for the next part of the project.

### Step E — Retrieve context for real PR hunks

Once natural-language search works, the important integration is searching automatically for each changed hunk.

For example:

```
for hunk in hunks:    query = f"""    Find repository code relevant to this change.    File: {hunk.filename}    Patch:    {hunk.patch}    """    context = search(        index=index,        query=query,        top_k=5,    )
```

This is an initial query format, not necessarily the final one. Large patches should be shortened or split to fit the query model's token limit.

Later, you can enrich the query with function names, imported symbols, enclosing classes and summaries of what changed.

Your immediate completion goal: Give the retriever a question or a changed hunk and get back five relevant chunks, each with its source path, line range and similarity score.

Don't start LangGraph before this works.

## 5. Complete roadmap to finish the project

Here's the implementation order I'd use. The key is to create a usable version first and then introduce more sophisticated features.

1. Phase 1 — Complete basic RAG retrieval

   Your current phase
   - Generate embeddings for code chunks.
   - Implement top-k semantic search.
   - Store vectors alongside source metadata.
   - Test whether relevant code is actually retrieved.
   - Add token-safe chunk splitting.
   - Integrate retrieval with changed hunks.
   Deliverable: A working repository search system that accepts PR changes and returns relevant source code.
2. Phase 2 — Build the LLM reviewer

   Replace `generate_fake_findings()` with a real review service.
   - Choose an LLM provider or a locally hosted model.
   - Design a review prompt containing the changed hunk and retrieved context.
   - Request structured findings instead of free-form prose.
   - Use Pydantic to validate the response schema.
   - Support a valid empty response when no problems are found.
   - Handle API failures and malformed output.
   Deliverable: An LLM that can propose real findings from a diff and supporting repository code.
3. Phase 3 — Improve validation and aggregation

   Extend your existing validation logic.
   - Require findings to reference real changed lines.
   - Check filenames and current PR commit.
   - Require an explanation grounded in the diff or retrieved code.
   - Reject vague, unsubstantiated and malformed findings.
   - Deduplicate findings on the same underlying problem.
   - Introduce consistent severity categories.
   - Bound retries instead of infinitely asking the LLM to reconsider.
   Deliverable: A safer pipeline that rejects structurally invalid and poorly supported findings. Semantic correctness still requires evaluation.
4. Phase 4 — Complete an end-to-end local MVP

   Connect retrieval, the LLM, and validation through `pipeline.py`.
   - Fetch a real PR.
   - Parse its changed hunks.
   - Retrieve relevant context for each hunk.
   - Generate structured findings.
   - Validate and aggregate them.
   - Display results without publishing.
   - Test optional GitHub review posting on a controlled repository.
   Deliverable: Your first usable PR review agent.

   This is the first major project milestone. You should aim to reach it before adding LangGraph.
5. Phase 5 — Improve retrieval with reranking
   - Retrieve a broader initial candidate set.
   - Add a cross-encoder to rerank candidates.
   - Experiment with lexical + vector (hybrid) retrieval.
   - Prioritize related functions, callers, imports and relevant tests.
   - Deduplicate overlapping chunks.
   - Compare accuracy and latency with the original retriever.
   Deliverable: Better context quality, supported by retrieval measurements.
6. Phase 6 — Introduce LangGraph orchestration

   Move the established pipeline into graph nodes.
   - Define the shared workflow state.
   - Implement `fetch_pr`, `plan_hunks`, `triage`, `retrieve`, `rerank`, `review`, `validate`, `aggregate`, and `post_review`.
   - Add conditional routing.
   - Support bounded review retries.
   - Track errors and outcomes.
   - Add checkpointing if recovery requirements justify it.
   Deliverable: A stateful, observable agent workflow instead of one large procedural function.
7. Phase 7 — GitHub Actions automation
   - Add PR event triggers.
   - Handle newly opened and updated pull requests.
   - Configure credentials and minimal permissions.
   - Use safe checkout and execution boundaries.
   - Avoid duplicate review comments across runs.
   - Revalidate the head commit before publishing.
   - Add timeouts, logging and failure handling.
   Deliverable: Reviews run automatically on GitHub.
8. Phase 8 — Evaluate and harden the system
   - Create PR test cases containing known bugs.
   - Include clean PRs to measure false positives.
   - Measure finding precision, recall and line accuracy.
   - Measure retrieval Recall\@k.
   - Track latency and token/API cost.
   - Test large diffs, renamed/deleted files and unavailable patches.
   - Test malicious content, prompt injection and API failures.
   Deliverable: Evidence that the agent produces useful reviews reliably.
9. Phase 9 — Optional advanced features
   - Incremental repository indexing.
   - Multi-language parsing with Tree-sitter.
   - Specialized security and performance reviewers.
   - Historical finding tracking.
   - Hosted vector storage.
   - More advanced evaluation and monitoring.
   - GitHub App or webhook-based deployment.
   Deliverable: A more scalable and production-oriented reviewer.

One important security consideration for Phase 7: pull-request code is untrusted. GitHub restricts tokens and secrets for forked `pull_request` workflows, and using `pull_request_target` incorrectly can expose privileged credentials. Keep analysis of untrusted code separate from privileged publishing, and avoid executing PR code with secrets available.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 6. How all the modules will connect

Your final data flow should look approximately like this:

Target architecture — the boxes will be implemented incrementally, not all at once.

I would keep your project organized around these responsibilities, but avoid refactoring every file into the proposed future structure immediately.

Refactor modules when they become large enough to need separation. For example, `retriever.py` can later be split into `chunker.py`, `indexer.py`, and `search.py`, but there's no urgent need to do that today.

## 7. What should qualify as your first complete version?

Before spending time on LangGraph, advanced reranking, or deployment, aim for this specific demonstration.

A developer introduces an intentionally buggy change:

```
def calculate_discount(price, discount):    return price + discount
```

Suppose the intended calculation requires subtraction. A successful run of your agent would:

1. Identify the changed line from the PR diff.
2. Retrieve relevant calculation logic or tests from the repository.
3. Ask the LLM to review the change using that context.
4. Return a finding explaining the likely problem and its evidence.
5. Confirm the finding points to a valid changed line.
6. Show it locally in dry-run mode.
7. Post the comment when explicitly enabled.

The agent must also handle a perfectly valid change by returning zero findings.

That is a meaningful MVP. It demonstrates GitHub integration, RAG, LLM reasoning, structured output, and validation without needing every advanced feature.

## 8. What I'd focus on during your next few coding sessions

Your immediate task checklist

0/6 completed

[] Resume and stabilize the current chunking experiment

[] Run existing tests, confirm token counts, and inspect any oversized chunks.

[] Build normalized vector embeddings

[] Encode the generated chunks and keep their source metadata.

[] Implement top-k semantic search

[] Use cosine similarity to retrieve chunks for a natural-language query.

[] Create a small retrieval test set

[] Write queries with known relevant functions and inspect the results.

[] Connect semantic retrieval to DiffHunk

[] Pass changed hunks into search and return related repository context.

[] Add tokenizer-aware overflow handling

[] Ensure that dense code cannot be silently truncated.

&#x20;Copy checklist

Once these are done, you can start building the LLM reviewer without a big architecture change.

## 9. A few things I would avoid

- Don't introduce LangGraph yet. It will add complexity without fixing missing retrieval or review functionality.
- Don't introduce a vector database just because the architecture mentions one. Start with NumPy, then switch when persistence and scale justify it.
- Don't spend too much time optimizing the embedding model before you have retrieval evaluations. A larger model is not automatically a better model for your repository.
- Don't treat model-generated evidence as proof that an issue is real. Validation and empirical testing are separate responsibilities.
- Don't automatically publish untested LLM findings. Keep dry-run as the default during development.

## 10. My overall assessment

| Area                               | Assessment                                                                    |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| Project idea                       | Strong for learning practical AI engineering                                  |
| Intended architecture              | Sensible, although ambitious for a first implementation                       |
| Current code organization          | Reasonable foundation                                                         |
| Diff parsing and location tracking | Useful groundwork, with room for edge-case testing                            |
| Code chunking                      | Good start; token-safety remains                                              |
| RAG functionality                  | Incomplete                                                                    |
| AI reviewer functionality          | Not started beyond scaffolding                                                |
| Production readiness               | Not yet                                                                       |
| Portfolio potential                | Good if you demonstrate measured accuracy and an end-to-end working PR review |

The strongest part of this project is not that it uses LangGraph or embeddings. It's the combination of context retrieval, grounded findings, controlled publishing, and measurable quality. Those are the parts that can make it more than a generic LLM wrapper.

Your next move is clear: stay in `retriever.py` and your embedding experiment until you can search the codebase successfully. Then replace `generate_fake_findings()` with an LLM that consumes the retrieved context.

That is the most direct path from where you stopped to a genuinely working first version.
