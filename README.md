# PR Review Agent

An autonomous AI-powered GitHub pull-request reviewer that analyzes code changes, retrieves relevant repository context, detects bugs and security issues, validates findings against the actual diff, and posts grounded review comments automatically.

The final project is intended to use **LangGraph**, **Retrieval-Augmented Generation (RAG)**, LLMs, and GitHub Actions to make automated code review more trustworthy.

> **Work in progress:** This repository is being built as a learning project. The current code and folders are evolving and do not yet represent the final architecture. The workflow and project structure documented below describe the intended, production-oriented version of the project; implementation details may change as the project develops.

## Overview

Most AI code-review tools fail for one fundamental reason: they ask an LLM to review code without enough context.

A pull request cannot always be reviewed correctly by looking only at changed lines. A reviewer also needs surrounding code, related functions, dependencies, and repository-specific conventions.

This project aims to combine:

- **LLM reasoning** for identifying potential bugs and issues
- **RAG retrieval** for grounding the model with relevant codebase context
- **LangGraph orchestration** for workflow state, branching, and retries
- **Validation layers** for preventing unsupported review comments

The goal is not simply to generate comments, but to generate comments developers can trust.

## How it works

```text
Pull-request event
        |
        v
Fetch PR metadata and files
        |
        v
Plan and prioritize changed hunks
        |
        v
Retrieve relevant repository context
        |
        v
Rerank the retrieved context
        |
        v
Generate candidate findings with an LLM
        |
        v
Validate findings against the diff and available context
        |
        v
Aggregate validated findings
        |
        v
Publish the review to GitHub
```

The agent can conditionally skip trivial changes and retry review generation when validation identifies unsupported output.

## Agent architecture

The intended completed workflow will use a graph with shared state rather than a single linear prompt.

| Node | Responsibility |
| --- | --- |
| `fetch_pr` | Fetch PR metadata, changed files, and patches from GitHub. |
| `plan_hunks` | Parse a diff into reviewable units. |
| `triage` | Skip trivial or low-value changes. |
| `retrieve` | Retrieve relevant repository context. |
| `rerank` | Improve the relevance of retrieved context. |
| `review` | Generate structured candidate findings. |
| `validate` | Confirm each finding is grounded in the diff and context. |
| `aggregate` | Deduplicate and organize accepted findings. |
| `post_review` | Publish the validated review to GitHub. |

### Retrieval and reranking

For each changed hunk, the future system will retrieve potentially relevant code and documentation with vector search, then rerank those candidates with a cross-encoder. This two-stage approach aims to retain useful context without overwhelming the review model with unrelated code.

### Grounding and validation

Every finding should reference a real file and a changed line. Findings that cannot be verified against the pull-request diff, or that are not supported by the available context, should be discarded or sent back for review rather than published.

Structured findings are preferred over free-form model output so they can be validated consistently:

```json
{
  "file": "database.py",
  "line": 42,
  "severity": "critical",
  "category": "security",
  "message": "User input is directly interpolated into SQL."
}
```

## Proposed future project structure

This is the **ideal future structure** for the completed project. It is included as a design target, not as a description of the present repository or a requirement for the learning implementation to follow exactly.

```text
pr-review-agent/
├── src/
│   └── pr_agent/
│       ├── cli.py                 # Command-line entry points
│       ├── config.py              # Environment and application settings
│       ├── github/
│       │   ├── client.py          # GitHub API integration
│       │   ├── fetcher.py         # Pull-request data collection
│       │   └── publisher.py       # Review/comment publishing
│       ├── diff/
│       │   ├── parser.py          # Unified-diff parsing
│       │   └── models.py          # Hunk and change models
│       ├── retrieval/
│       │   ├── indexer.py         # Repository indexing
│       │   ├── search.py          # Context retrieval
│       │   └── reranker.py        # Optional result reranking
│       ├── review/
│       │   ├── prompts.py         # Review prompts
│       │   ├── models.py          # Structured finding models
│       │   └── service.py         # LLM review generation
│       ├── validation/
│       │   └── grounding.py       # Diff/context validation
│       └── workflow/
│           ├── graph.py           # Agent orchestration
│           └── state.py           # Shared workflow state
├── tests/
├── experiments/                   # Small manual integration experiments
├── docs/
├── .github/
│   └── workflows/                 # Future GitHub Actions workflow
├── pyproject.toml
└── README.md
```

## Technology choices

- Python and Pydantic for application code and structured data
- GitHub REST API and GitHub Actions for pull-request integration
- LangGraph for stateful agent orchestration
- Embedding models, vector search, and reranking for repository-aware retrieval
- An LLM for candidate review findings

## GitHub Actions deployment

The completed agent is intended to run automatically whenever a pull request is opened or updated:

```text
Developer opens or updates a PR
        |
        v
GitHub Actions trigger
        |
        v
Run the review-agent workflow
        |
        v
Post a validated GitHub review
```

## Evaluation

A useful review agent needs to be measured, not judged only by the number of comments it produces. The planned evaluation approach uses pull requests containing known or planted defects and tracks metrics such as:

- **Precision:** `TP / (TP + FP)` — how many reported findings are correct.
- **Recall:** `TP / (TP + FN)` — how many known defects are found.
- False-positive rate, grounded-line accuracy, latency, and review cost.

## Design decisions

### Why retrieval instead of fine-tuning?

Repositories change frequently. Retrieval can provide current project context without retraining a model whenever the codebase changes.

### Why validate rather than trust the LLM?

LLMs are useful for proposing issues but can be wrong or insufficiently grounded. The intended system follows this rule:

```text
LLM proposes a finding
        ↓
Validation verifies it
        ↓
Only grounded findings are published
```

### Why workflow orchestration?

An orchestrated graph makes it practical to share state between steps, skip low-value changes, retry unreliable output, and add checkpointing as the system becomes more capable.

## Current development status

Working and tested (`pytest`):

- PR fetching, diff parsing and changed-line tracking, noise-file filtering
- Deterministic grounding check (a finding must target an added line); the validator reports what it dropped, which drives the retry
- Heuristic triage that skips hunks that cannot change behaviour (comments, imports, reformatting) without spending a model call
- AST-based chunking, with a **token-budget guarantee**: no chunk is ever silently truncated by the embedding model
- Semantic search (`pr_agent.search`): `build_index` once, `index.search(query)` many times
- Change-impact retrieval: static analysis finds code the PR did *not* touch that its change could break (callers, instantiations, subclasses, overrides, dangling references to removed names), separates callers the PR already updated from stale ones, ranks and caps them, and tells the reviewer why each is shown. Semantic search and definition lookup fill the rest. See [docs/retrieval_scenarios.md](docs/retrieval_scenarios.md) for every scenario covered and every known limit
- `Severity`/`Category` enums and a Pydantic `Finding`/`ReviewOutput`: LLM output is validated at the boundary, with the JSON schema generated from the same models
- `LLMReviewer`: builds the prompt (numbered diff lines, retrieved context, untrusted-data rules), validates the reply, retries once with the error attached; talks to the model only through the `LLMClient` protocol
- A LangGraph agent (`pr_agent.graph`) with the triage-skip and validate-retry decisions, a bounded retry governor, and SQLite checkpointing that resumes after a crash and makes re-running a commit a no-op. See [docs/architecture.md](docs/architecture.md)
- A **verification pass**: each surviving finding is re-checked against the code it depends on (the enclosing function and the bodies of what it calls) and only confirmed ones are reported
- A command line (`python -m pr_agent review`) that reads a PR, snapshots its head code safely, runs the graph, and posts a summary plus inline comments that degrade gracefully and never duplicate
- `run_review` accepts an injectable `reviewer` and an optional `index`, so any reviewer receives each hunk together with its retrieved context

Not yet built: the GitHub Action, reranking, and a way to review non-Python code with impact analysis.

Measure retrieval changes with `python experiments/retrieval_eval.py -v` rather than by eye.

## Running locally

Requirements: Python 3.13 or newer, and an API key for at least one LLM provider
(several have free tiers; see `.env.example`).

```bash
uv sync                        # or: python -m venv .venv && pip install -e .
cp .env.example .env           # then fill in at least one provider's key
pytest                         # 500+ offline tests, no network or keys needed
```

Review a pull request. The default is a **dry run**: it prints the findings and posts nothing.

```bash
python -m pr_agent review --repo owner/name --pr 7              # dry run
python -m pr_agent review --repo owner/name --pr 7 --post       # also post the review
python -m pr_agent review --repo owner/name --commit abc1234    # review a past commit, read-only
```

Useful options: `--fail-on warning` (exit 1 for CI), `--json`, `--no-verify` (skip the second
check on each finding), `--no-retrieval` (diff only: faster and cheaper), `--root DIR` (index a
local checkout instead of downloading the PR's head), `--checkpoint FILE` (resume after a crash;
re-running the same commit does nothing), `--provider NAME --model ID`, `-v`.

Reading public repositories needs no GitHub token; posting needs `GITHUB_TOKEN`. A rejected token
falls back to anonymous reading for a dry run. Per-run spending caps (`LLM_MAX_CALLS_PER_RUN`,
`LLM_MAX_TOKENS_PER_RUN`) and the provider choice (`LLM_PROVIDER`) are explained in `.env.example`.

Only source code is reviewed (documentation, data and binaries are skipped), and only Groq has
been tested against a live account.

Live experiments (they call a real model; they say what they will cost before spending it):

```bash
python experiments/llm_eval.py --dry-run            # forecast the cost of the planted-bug suite
python experiments/llm_eval.py --verify             # run it, reporting results before and after verification
python experiments/retrieval_eval.py -v             # retrieval quality, offline apart from the local embedding model
```

## Future improvements

- AST-aware chunking with Tree-sitter
- Specialized reviewers for security, performance, and style
- Historical pull-request learning
- Evaluation datasets with known defects
- Hosted embedding and reranking models
- A production webhook service

## Development notes

- `experiments/` contains scratch scripts and sample pull-request material used while exploring GitHub integration and parsing.
- `docs/understanding.md` captures the broader product vision and design rationale.
- GitHub credentials, an LLM provider, and a vector database are not required for the currently implemented parser and tests. They will be documented when the related integrations are added.

## License

MIT License
