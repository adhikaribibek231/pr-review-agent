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

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pr_agent.models import Finding, ReviewOutput


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
