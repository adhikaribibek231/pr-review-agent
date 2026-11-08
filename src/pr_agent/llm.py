"""The boundary between this project and a language model.

Everything else depends on the tiny `LLMClient` protocol, never on a provider
SDK. Today the only implementation is `FakeLLMClient`; a Groq client later is one
new class with the same single method, and nothing else changes.

Two deliberate choices:

* The client returns the model's reply as a *string*. Parsing and validating it
  is the reviewer's job (pr_agent.reviewer), because "the reply was bad" is a
  review-level concern (retry, degrade), not a transport-level one.
* Transport problems (rate limits, timeouts) are the client's job and should
  surface as exceptions. Retrying those with backoff belongs inside the real
  client, not in the reviewer.
"""

import logging
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pr_agent.models import Finding, ReviewOutput

logger = logging.getLogger(__name__)


class PromptTooLargeError(Exception):
    """The request cannot fit the provider's per-request limit as it stands.

    Raised before anything is sent. The reviewer answers it by shrinking the retrieved
    context and trying again, rather than treating it as a failed review.
    """


class LLMClient(Protocol):
    def complete_json(self, *, system: str, user: str) -> str:
        """Send a system + user prompt; return the model's raw JSON reply text."""
        ...


@dataclass(frozen=True)
class LLMCall:
    """One recorded request, so tests can assert on exactly what was sent."""

    system: str
    user: str


# A scripted reply is either literal text or a function of the request, which
# lets a test make the fake "behave differently on the retry prompt".
Reply = str | Callable[[LLMCall], str]

NOTHING_FOUND = '{"findings": []}'


def findings_reply(findings: Sequence[Finding]) -> str:
    """JSON reply text a well-behaved model would send for these findings."""
    return ReviewOutput(findings=list(findings)).model_dump_json()


class FakeLLMClient:
    """Scripted stand-in for a model: offline, free, deterministic.

    Replies are consumed in order, one per call. When the script runs out it
    keeps returning `default` (an empty review) so a test only has to script the
    calls it cares about. Every request is recorded in `calls`.
    """

    def __init__(self, replies: Sequence[Reply] = (), default: str = NOTHING_FOUND) -> None:
        self._replies: deque[Reply] = deque(replies)
        self._default = default
        self.calls: list[LLMCall] = []

    def complete_json(self, *, system: str, user: str) -> str:
        call = LLMCall(system=system, user=user)
        self.calls.append(call)
        if not self._replies:
            return self._default
        reply = self._replies.popleft()
        return reply(call) if callable(reply) else reply


def is_retriable(error: Exception) -> bool:
    """Is this failure the provider's (or the network's), so another provider may succeed?

    Rate limits, timeouts, connection failures and 5xx are. Bad requests and auth
    failures are not: they would fail the same way, and hiding them behind a fallback
    would leave a broken configuration looking healthy. Duck-typed on `status_code`, so
    this needs no SDK.
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    name = type(error).__name__
    return isinstance(error, (ConnectionError, TimeoutError)) or "Timeout" in name or "Connection" in name \
        or name == "BudgetExceededError"


class FallbackClient:
    """Tries clients in order; a retriable failure moves on to the next.

    This is how several free tiers add up: when the first provider is rate limited or
    out of its daily quota, the next one answers. A non-retriable error (bad key, bad
    request) propagates at once. If every provider fails, the last error is raised, or
    PromptTooLargeError if that is what the providers said.
    """

    def __init__(self, clients: Sequence[LLMClient], names: Sequence[str] | None = None) -> None:
        if not clients:
            raise ValueError("FallbackClient needs at least one client")
        self._clients = list(clients)
        self._names = list(names) if names else [f"client{i}" for i in range(len(clients))]
        self.last_used: str | None = None

    def complete_json(self, *, system: str, user: str) -> str:
        last_error: Exception | None = None
        for client, name in zip(self._clients, self._names, strict=True):
            try:
                reply = client.complete_json(system=system, user=user)
            except PromptTooLargeError as error:
                logger.warning("%s: prompt too large (%s); trying the next provider", name, error)
                last_error = error
            except Exception as error:
                if not is_retriable(error):
                    raise
                logger.warning("%s unavailable (%s: %s); trying the next provider", name, type(error).__name__, error)
                last_error = error
            else:
                self.last_used = name
                return reply
        assert last_error is not None
        raise last_error
