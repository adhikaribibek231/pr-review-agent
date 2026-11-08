"""LLM-backed reviewer: HunkContext in, validated Findings out.

`LLMReviewer` satisfies the `Reviewer` type that `run_review` accepts, so it
drops into the pipeline with no other change.

Per hunk it does:  build prompt -> call client -> parse + validate
                   -> if invalid: call again ONCE with the error attached
                   -> if still invalid: log it and report nothing for that hunk
"""

import logging
import re

from pydantic import ValidationError

from pr_agent.llm import LLMClient, PromptTooLargeError
from pr_agent.models import Finding, HunkContext, ReviewOutput
from pr_agent.prompts import MAX_CONTEXT_CHARS, Prompt, build_review_prompt

logger = logging.getLogger(__name__)

# One call covers every hunk of a file, up to these limits; a bigger file takes several calls.
MAX_BATCH_DIFF_CHARS = 5_000
MAX_BATCH_HUNKS = 8

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_MAX_ECHOED_REPLY_CHARS = 2_000


def extract_json_object(raw: str) -> str:
    """Peel off what models wrap around JSON even when asked not to.

    Handles ```json fences and chatter before or after the object
    ("Sure! Here is the review: {...} Hope that helps!").
    It does not repair broken JSON: that is left to validation, which fails
    loudly and triggers the retry.
    """
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    if "{" in text and "}" in text:
        # First "{" to last "}" drops chatter on either side. A no-op for clean JSON.
        text = text[text.index("{") : text.rindex("}") + 1]
    return text


def parse_review(raw: str) -> ReviewOutput:
    """Raw model text -> ReviewOutput. Raises `ValidationError` if it can't.

    Bad JSON and a good-JSON-wrong-shape reply both raise the same exception
    type, so callers need a single `except`.
    """
    return ReviewOutput.model_validate_json(extract_json_object(raw))


def describe_validation_error(error: ValidationError) -> str:
    """Compact, model-readable error text: one `path: problem` line per error.

    str(error) also includes documentation URLs and echoed inputs, which is
    noise (and tokens) when the reader is an LLM that should just fix its reply.
    """
    lines = []
    for item in error.errors():
        path = ".".join(str(part) for part in item["loc"]) or "(whole reply)"
        lines.append(f"- {path}: {item['msg']}")
    return "\n".join(lines)


def _repair_request(original_user_prompt: str, bad_reply: str, error: ValidationError) -> str:
    return (
        f"{original_user_prompt}\n\n"
        "Your previous reply was rejected.\n"
        f"Previous reply:\n{bad_reply[:_MAX_ECHOED_REPLY_CHARS]}\n\n"
        f"Problems:\n{describe_validation_error(error)}\n\n"
        "Reply again with ONLY a valid JSON object that fixes these problems."
    )


def group_into_batches(items: list[HunkContext]) -> list[list[HunkContext]]:
    """Group hunks by file, in order of first appearance, splitting a file whose diffs are large.

    Reviewing a file's hunks together fixes two problems of reviewing each alone: the fixed
    instructions are paid once per file instead of once per hunk, and the model can see that
    a variable one hunk adds is used by another hunk (it otherwise reports it as unused).
    """
    by_file: dict[str, list[HunkContext]] = {}
    for item in items:
        by_file.setdefault(item.hunk.filename, []).append(item)
    batches: list[list[HunkContext]] = []
    for file_items in by_file.values():
        current: list[HunkContext] = []
        size = 0
        for item in file_items:
            if current and (size + len(item.hunk.patch) > MAX_BATCH_DIFF_CHARS or len(current) >= MAX_BATCH_HUNKS):
                batches.append(current)
                current, size = [], 0
            current.append(item)
            size += len(item.hunk.patch)
        batches.append(current)
    return batches


class LLMReviewer:
    def __init__(self, client: LLMClient, max_attempts: int = 2,
                 max_context_chars: int = MAX_CONTEXT_CHARS) -> None:
        """`max_attempts=2` means one retry. The cap is a governor: a model that
        keeps returning garbage costs two calls per hunk, never an endless loop."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._client = client
        self._max_attempts = max_attempts
        self._max_context_chars = max_context_chars

    def __call__(self, items: list[HunkContext]) -> list[Finding]:
        return self.review(items)

    def review(self, items: list[HunkContext], stricter: bool = False) -> list[Finding]:
        """Review the hunks, one call per file (see `group_into_batches`), and concatenate the findings.

        `stricter` selects the tightened prompt for the validate->review retry
        edge. Client exceptions (rate limits, network) are deliberately NOT
        caught: a failed call is not "no findings", and silently reporting a
        clean review for a hunk we never reviewed would be worse than failing.
        """
        findings: list[Finding] = []
        for batch in group_into_batches(items):
            findings.extend(self._review_batch(batch, stricter))
        return findings

    def _review_batch(self, batch: list[HunkContext], stricter: bool) -> list[Finding]:
        item = batch[0]  # for log messages: every item in a batch is from the same file
        # If the provider says the prompt is too large, retry with half the context, then a
        # quarter, then none: a review with less context beats no review. This is separate
        # from the invalid-reply retry below, which is about the model's answer.
        budget = self._max_context_chars
        while True:
            prompt = build_review_prompt(batch, stricter=stricter, max_context_chars=budget)
            try:
                return self._ask(item, prompt)
            except PromptTooLargeError as error:
                if budget == 0:
                    logger.warning("Giving up on a batch in %s: too large even without context (%s)",
                                   item.hunk.filename, error)
                    return []
                budget = budget // 2 if budget >= 500 else 0
                logger.warning("Prompt too large for %s; retrying with %d context characters",
                               item.hunk.filename, budget)

    def _ask(self, item: HunkContext, prompt: Prompt) -> list[Finding]:
        user = prompt.user
        for attempt in range(1, self._max_attempts + 1):
            raw = self._client.complete_json(system=prompt.system, user=user)
            try:
                return parse_review(raw).findings
            except ValidationError as error:
                logger.warning(
                    "Invalid reply for %s (attempt %d/%d):\n%s",
                    item.hunk.filename, attempt, self._max_attempts, describe_validation_error(error),
                )
                user = _repair_request(prompt.user, raw, error)
        logger.warning("Giving up on a batch in %s: no valid reply", item.hunk.filename)
        return []
