import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from pr_agent.chat_client import BudgetExceededError, ChatClient, estimate_tokens, make_client
from pr_agent.llm import FallbackClient, PromptTooLargeError
from pr_agent.providers import PROVIDERS, LLMConfigError, ResolvedProvider


def provider(name: str = "groq", model: str = "m") -> ResolvedProvider:
    spec = PROVIDERS[name]
    return ResolvedProvider(spec, "secret-key", model, spec.default_base_url or "http://x")


def reply(content: str | None = '{"findings": []}', finish: str = "stop",
          prompt_tokens: int = 10, completion_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def sdk_returning(response: SimpleNamespace) -> Mock:
    sdk = Mock()
    sdk.chat.completions.create.return_value = response
    return sdk


class Clock:
    """A fake clock whose sleep() advances time instantly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def client(prov: ResolvedProvider | None = None, response: SimpleNamespace | None = None, **options: object) -> ChatClient:
    return ChatClient(prov or provider(), sdk=sdk_returning(response or reply()), **options)  # type: ignore[arg-type]


# ---- what is sent ---------------------------------------------------------------------------


def test_sends_system_and_user_prompts_with_the_providers_model() -> None:
    c = client(provider(model="my-model"))
    _ = c.complete_json(system="SYS", user="USR")
    kwargs = c._sdk.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "my-model"
    assert kwargs["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}]
    assert kwargs["temperature"] == 0.0 and kwargs["max_tokens"] == 2000


def test_groq_gets_json_mode_and_reasoning_effort() -> None:
    c = client(provider("groq"))
    _ = c.complete_json(system="s", user="u")
    kwargs = c._sdk.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"] == {"reasoning_effort": "medium"}


@pytest.mark.parametrize("name", ["huggingface", "openrouter", "cerebras", "gemini"])
def test_unverified_providers_get_no_provider_specific_parameters(name: str) -> None:
    c = client(provider(name))
    _ = c.complete_json(system="s", user="u")
    kwargs = c._sdk.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs and "extra_body" not in kwargs


def test_returns_raw_text_and_turns_empty_content_into_an_empty_string() -> None:
    assert client(response=reply('{"findings": []}')).complete_json(system="s", user="u") == '{"findings": []}'
    assert client(response=reply(None)).complete_json(system="s", user="u") == ""


def test_token_usage_accumulates() -> None:
    c = client(response=reply(prompt_tokens=100, completion_tokens=40))
    _ = c.complete_json(system="s", user="u")
    _ = c.complete_json(system="s", user="u")
    assert (c.usage.calls, c.usage.total_tokens) == (2, 280)


def test_a_truncated_reply_is_logged_with_the_provider(caplog: pytest.LogCaptureFixture) -> None:
    c = client(response=reply('{"findings": [', finish="length"))
    with caplog.at_level(logging.WARNING):
        _ = c.complete_json(system="s", user="u")
    assert "groq reply hit max_tokens" in caplog.text


def test_transport_errors_propagate() -> None:
    sdk = Mock()
    sdk.chat.completions.create.side_effect = TimeoutError("slow")
    with pytest.raises(TimeoutError):
        ChatClient(provider(), sdk=sdk).complete_json(system="s", user="u")


# ---- hard session caps ------------------------------------------------------------------------


def test_the_call_cap_stops_the_next_call_before_it_is_made() -> None:
    c = client(max_calls=2)
    _ = c.complete_json(system="s", user="u")
    _ = c.complete_json(system="s", user="u")
    with pytest.raises(BudgetExceededError, match="call cap of 2"):
        c.complete_json(system="s", user="u")
    assert c._sdk.chat.completions.create.call_count == 2


def test_the_token_cap_stops_further_calls() -> None:
    c = client(response=reply(prompt_tokens=90, completion_tokens=20), max_total_tokens=100)
    _ = c.complete_json(system="s", user="u")  # now at 110, over the cap
    with pytest.raises(BudgetExceededError, match="token cap"):
        c.complete_json(system="s", user="u")


# ---- pacing under a tokens-per-minute limit ---------------------------------------------------------


def paced(clock: Clock, **options: object) -> ChatClient:
    return client(clock=clock, sleep=clock.sleep, **options)


def test_requests_that_fit_the_window_are_not_delayed() -> None:
    clock = Clock()
    c = paced(clock, max_tokens=1000)
    for _ in range(3):  # 3 * (1000 + a few) fits 7200
        _ = c.complete_json(system="s", user="u")
    assert clock.slept == []


def test_a_request_that_would_exceed_the_minute_waits_for_the_window_to_roll() -> None:
    clock = Clock()
    c = paced(clock, max_tokens=3000)
    _ = c.complete_json(system="s", user="u")
    _ = c.complete_json(system="s", user="u")  # 2 * ~3000 = 6000 of 7200
    assert clock.slept == []
    _ = c.complete_json(system="s", user="u")  # needs ~3000 more: must wait for the first to age out
    assert clock.slept and 55 <= sum(clock.slept) <= 61
    assert clock.now >= 60


def test_a_request_that_can_never_fit_is_rejected_before_sending() -> None:
    c = client(max_tokens=2000)
    with pytest.raises(PromptTooLargeError):
        c.complete_json(system="x" * 30_000, user="y" * 30_000)  # ~20k estimated tokens
    c._sdk.chat.completions.create.assert_not_called()


def test_pacing_is_corrected_by_the_tokens_the_provider_actually_reports() -> None:
    clock = Clock()
    # The estimate for "s"/"u" is tiny; the provider says the prompt was 4000 tokens.
    c = paced(clock, response=reply(prompt_tokens=4000, completion_tokens=1), max_tokens=1000)
    _ = c.complete_json(system="s", user="u")  # settles at 4000 + 1000 = 5000
    _ = c.complete_json(system="s", user="u")  # 5000 + ~1000 <= 7200: no wait yet
    assert clock.slept == []
    _ = c.complete_json(system="s", user="u")  # 5000 + 5000 would exceed: waits
    assert clock.slept


def test_providers_without_a_known_limit_are_not_paced() -> None:
    clock = Clock()
    c = client(provider("openrouter"), clock=clock, sleep=clock.sleep, max_tokens=7000)
    for _ in range(5):
        _ = c.complete_json(system="s", user="u")
    assert clock.slept == []


def test_pacing_can_be_set_explicitly_for_any_provider() -> None:
    c = client(provider("openrouter"), tokens_per_minute=1000, max_tokens=900)
    with pytest.raises(PromptTooLargeError):
        c.complete_json(system="x" * 3000, user="y")


def test_estimate_tokens_is_conservative_for_code() -> None:
    assert estimate_tokens("x" * 300) >= 100


# ---- building the client from the environment --------------------------------------------------------


def test_one_configured_provider_gives_a_plain_client() -> None:
    with patch("openai.OpenAI") as sdk:
        built = make_client({"GROQ_API_KEY": "gsk_abc"}, max_retries=7, timeout=12.0)
    assert isinstance(built, ChatClient)
    sdk.assert_called_once_with(base_url="https://api.groq.com/openai/v1", api_key="gsk_abc",
                                max_retries=7, timeout=12.0)


def test_several_configured_providers_give_a_fallback_chain_in_priority_order() -> None:
    with patch("openai.OpenAI"):
        built = make_client({"HF_TOKEN": "hf_x", "GROQ_API_KEY": "gsk_x"})
    assert isinstance(built, FallbackClient)
    assert built._names == ["groq", "huggingface"]


def test_options_apply_to_every_provider_in_the_chain() -> None:
    with patch("openai.OpenAI"):
        built = make_client({"HF_TOKEN": "hf_x", "GROQ_API_KEY": "gsk_x"}, max_calls=3)
    assert all(c._max_calls == 3 for c in built._clients)  # type: ignore[attr-defined]


def test_nothing_configured_fails_before_any_network_use() -> None:
    with patch("openai.OpenAI") as sdk:
        with pytest.raises(LLMConfigError):
            make_client({})
    sdk.assert_not_called()
