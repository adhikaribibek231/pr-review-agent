
# Chapter 1 – What I'm Building

## Project Summary

The PR Review Agent is an AI-powered code review system that automatically reviews GitHub pull requests. When a pull request is opened or updated, a GitHub Action triggers the workflow. The system analyzes the code changes, retrieves only the relevant parts of the repository needed to understand those changes, asks an LLM to identify potential bugs, security issues, and coding convention violations, validates that each finding is supported by real code, and posts the validated findings as a structured GitHub PR review with inline comments where appropriate.

Unlike simply sending a diff to an LLM, this system attempts to provide repository context before analysis and rejects findings that cannot be verified. The goal is to increase the reliability of automated code reviews rather than maximizing the number of comments generated.

---

## Inputs

The system receives:

* GitHub pull request metadata

* The changed files and diff

* Relevant source code retrieved from the repository

* Coding conventions or project-specific guidance

* Previous workflow state (when required)

---

## Outputs

The system produces:

* A structured GitHub pull request review

* Inline review comments attached to specific lines when appropriate

* A review summary describing the findings

* Zero comments if no validated issues are found

---

## Role of Each Component

### GitHub Actions

* Detects pull request events.

* Starts the review workflow automatically.

* Provides the repository and pull request information.

### LLM

The LLM analyzes the retrieved code and proposes potential:

* Bugs

* Security vulnerabilities

* Logic errors

* Code quality issues

* Convention violations

The LLM generates candidate findings but is **not** considered the source of truth.

### RAG (Retrieval-Augmented Generation)

RAG retrieves repository context relevant to the changed code, such as:

* Surrounding implementation

* Related functions and classes

* Dependencies

* Interfaces

* Project conventions

This allows the LLM to reason using more than the isolated diff.

### LangGraph

LangGraph orchestrates the workflow by managing:

* Execution order

* State shared between steps

* Conditional branching

* Validation stages

* Retries or recovery if needed

It coordinates the overall pipeline rather than performing code analysis itself.

---

## Why an LLM Alone Is Not Enough

A raw git diff often lacks the context necessary for accurate code review.

Without additional context, the LLM may:

* Misunderstand how a function is used elsewhere

* Miss interactions with other modules

* Ignore project-specific conventions

* Produce hallucinated or unsupported findings

Providing relevant repository context reduces these problems but does not eliminate them entirely.

---

## Why This Is More Than a Chatbot Wrapper

A chatbot wrapper simply sends a prompt to an LLM and returns its response.

This project performs several additional steps:

1. Filters and prepares the pull request changes.

2. Retrieves relevant repository context.

3. Performs LLM analysis.

4. Validates whether each finding is supported by actual code.

5. Rejects unsupported findings.

6. Formats and posts the validated review to GitHub automatically.

These additional stages are intended to improve reliability compared with a direct LLM call.

---

## Definition of Done

The project is considered complete when all of the following are true:

1. A pull request containing known bugs triggers the GitHub Action and receives a correct, context-aware review.

2. The system has measurable evaluation results, including metrics such as bug detection rate and false-positive rate on a benchmark of known defects.

3. I can explain every node in the LangGraph workflow, every major design decision, and the trade-offs behind those decisions.

---

## Core Engineering Principle

The objective is not to maximize the number of comments produced.

The objective is to maximize the reliability of the comments that are posted.

Grounding, validation, and evaluation are therefore more important than the LLM call itself. If a finding cannot be supported by the retrieved code or fails validation, it should be discarded rather than presented with unwarranted confidence.


