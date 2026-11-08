"""`LLMClient` for any OpenAI-compatible chat service (Groq, Hugging Face, OpenRouter, ...).

The only module that imports an LLM SDK. A provider is just a `ResolvedProvider` (base
URL, key, model, capability flags), so adding another free service is data, not code.

What it owns (transport concerns):
  * retries with backoff for rate limits (429), timeouts and 5xx: the SDK does this,
    honouring the server's Retry-After header, and we set how many times;
  * a pre-flight size check and pacing under a tokens-per-minute limit, when known;
  * hard session caps on calls and tokens, so a bug cannot silently spend a day's quota;
  * JSON mode, when the provider supports it.

What it does NOT own: deciding whether the JSON has the right shape. That is the
reviewer's job (pydantic validation, one retry with the error attached).
"""

import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from pr_agent.llm import FallbackClient, LLMClient, PromptTooLargeError, Usage
from pr_agent.providers import ResolvedProvider, describe, run_caps, select_providers

logger = logging.getLogger(__name__)

# Safety margin: the estimate below is deliberately rough, so aim under the limit.
_PACING_MARGIN = 0.9
_WINDOW_SECONDS = 60.0


class BudgetExceededError(RuntimeError):
    """A session cap on calls or tokens was reached. Raised BEFORE the call is made."""


def estimate_tokens(text: str) -> int:
    """A cheap, conservative token estimate (code runs about 3 characters per token)."""
    return len(text) // 3 + 1


class _TokenPacer:
    """Keeps requests under a tokens-per-minute limit.

    Groq charges a request its prompt tokens PLUS the full `max_tokens` it reserved
    (measured: a 78-token prompt with max 60 dropped the counter by 138), and the charge
    ages out over the following minute. This mirrors that with a rolling window, so we
    wait instead of earning a 429, and reject up front a request that could never fit.
    """

    def __init__(self, limit: int, clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
        self._limit = int(limit * _PACING_MARGIN)
        self._clock = clock
        self._sleep = sleep
        self._entries: list[list[float]] = []  # [timestamp, cost]

    def reserve(self, cost: int) -> list[float]:
        if cost > self._limit:
            raise PromptTooLargeError(
                f"request needs about {cost} tokens (prompt plus reserved reply) but the per-minute "
                f"limit allows {self._limit}")
        while True:
            now = self._clock()
            self._entries = [e for e in self._entries if now - e[0] < _WINDOW_SECONDS]
            if sum(e[1] for e in self._entries) + cost <= self._limit:
                entry = [now, float(cost)]
                self._entries.append(entry)
                return entry
            wait = self._entries[0][0] + _WINDOW_SECONDS - now
            logger.info("Pacing: waiting %.1fs for token budget", max(wait, 0.0))
            self._sleep(max(wait, 0.05) + 0.05)

    @staticmethod
    def settle(entry: list[float], actual_cost: int) -> None:
        """Replace the estimate with what the provider actually charged."""
        entry[1] = float(actual_cost)


_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_restricted_openai_model(provider: ResolvedProvider) -> bool:
    return provider.name == "openai" and provider.model.lower().startswith(_OPENAI_REASONING_PREFIXES)


class ChatClient:
    def __init__(
        self,
        provider: ResolvedProvider,
        *,
        max_retries: int = 3,
        timeout: float = 60.0,
        temperature: float = 0.0,
        reasoning_effort: str = "medium",
        max_tokens: int = 2000,
        tokens_per_minute: int | None = -1,
        max_calls: int | None = None,
        max_total_tokens: int | None = None,
        sdk: Any = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """`sdk`, `clock` and `sleep` are for tests.

        temperature=0 so the same diff gets the same review, which an evaluation needs.
        `max_tokens` is generous because a reasoning model's hidden reasoning counts against
        it, and a reply cut off mid-JSON is useless. But it is also what the provider may
        reserve against the per-minute limit, so don't make it larger than needed.
        `tokens_per_minute=-1` means "use the provider's known limit"; None disables pacing.
        `max_calls` / `max_total_tokens` are hard session caps (per provider).
        """
        self.provider = provider
        self.model = provider.model
        self.usage = Usage()
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._max_calls = max_calls
        self._max_total_tokens = max_total_tokens
        limit = provider.spec.tokens_per_minute if tokens_per_minute == -1 else tokens_per_minute
        self._pacer = _TokenPacer(limit, clock, sleep) if limit else None
        if sdk is not None:
            self._sdk = sdk
        else:
            from openai import OpenAI  # imported here so tests and offline runs never need the SDK

            self._sdk = OpenAI(base_url=provider.base_url, api_key=provider.api_key,
                               max_retries=max_retries, timeout=timeout)

    def complete_json(self, *, system: str, user: str) -> str:
        """Send one review request and return the model's raw JSON text."""
        if self._max_calls is not None and self.usage.calls >= self._max_calls:
            raise BudgetExceededError(f"{self.provider.name}: call cap of {self._max_calls} reached")
        if self._max_total_tokens is not None and self.usage.total_tokens >= self._max_total_tokens:
            raise BudgetExceededError(f"{self.provider.name}: token cap of {self._max_total_tokens} reached "
                                      f"({self.usage.total_tokens} used)")
        reservation = None
        if self._pacer is not None:
            estimated_prompt = estimate_tokens(system) + estimate_tokens(user)
            reservation = self._pacer.reserve(estimated_prompt + self._max_tokens)

        options: dict[str, Any] = {}
        if _is_restricted_openai_model(self.provider):
            # OpenAI's reasoning models reject `max_tokens` and any non-default temperature.
            options["max_completion_tokens"] = self._max_tokens
        else:
            options["max_tokens"] = self._max_tokens
            options["temperature"] = self._temperature
        if self.provider.spec.json_mode:
            options["response_format"] = {"type": "json_object"}
        if self.provider.spec.reasoning_effort:
            options["extra_body"] = {"reasoning_effort": self._reasoning_effort}
        response = self._sdk.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **options,
        )
        usage = getattr(response, "usage", None)
        self.usage.calls += 1
        if usage is not None:
            prompt = getattr(usage, "prompt_tokens", 0) or 0
            self.usage.prompt_tokens += prompt
            self.usage.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            if reservation is not None and self._pacer is not None:
                self._pacer.settle(reservation, prompt + self._max_tokens)

        choice = response.choices[0]
        if choice.finish_reason == "length":
            # Cut off by the token limit, so the JSON is probably truncated. The reviewer's
            # validation will reject it and retry; this says why.
            logger.warning("%s reply hit max_tokens (%d); the JSON is likely truncated",
                           self.provider.name, self._max_tokens)
        return choice.message.content or ""


def make_client(env: Mapping[str, str] | None = None, **client_options: Any) -> LLMClient:
    """Build the client the environment asks for.

    One configured provider gives a plain `ChatClient`; several give a `FallbackClient`
    that tries them in priority order. Raises `LLMConfigError`, naming the exact variable
    to set, if nothing usable is configured. `client_options` apply to every provider. Per-run
    spending caps come from LLM_MAX_CALLS_PER_RUN / LLM_MAX_TOKENS_PER_RUN unless passed here.
    """
    environment = os.environ if env is None else env
    providers = select_providers(environment)
    max_calls, max_tokens = run_caps(environment)
    client_options.setdefault("max_calls", max_calls)
    client_options.setdefault("max_total_tokens", max_tokens)
    logger.info("LLM providers: %s", describe(providers))
    for provider in providers:
        if provider.spec.cost != "free":
            logger.warning("%s is a %s provider: calls may cost money", provider.name, provider.spec.cost)
    clients = [ChatClient(provider, **client_options) for provider in providers]
    return clients[0] if len(clients) == 1 else FallbackClient(clients, [p.name for p in providers])
