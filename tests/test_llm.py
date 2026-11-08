import pytest

from pr_agent.llm import (
    NOTHING_FOUND, FakeLLMClient, FallbackClient, LLMCall, PromptTooLargeError, findings_reply, is_retriable,
)
from pr_agent.models import Category, Finding, ReviewOutput, Severity


def ask(client: FakeLLMClient, user: str = "u") -> str:
    return client.complete_json(system="s", user=user)


def test_default_reply_is_an_empty_review() -> None:
    assert ask(FakeLLMClient()) == NOTHING_FOUND
    assert ReviewOutput.model_validate_json(NOTHING_FOUND).findings == []


def test_scripted_replies_are_consumed_in_order_then_fall_back_to_default() -> None:
    client = FakeLLMClient(["first", "second"], default="fallback")
    assert [ask(client) for _ in range(4)] == ["first", "second", "fallback", "fallback"]


def test_a_reply_can_be_a_function_of_the_request() -> None:
    client = FakeLLMClient([lambda call: f"saw {call.user}"])
    assert ask(client, user="hello") == "saw hello"


def test_every_request_is_recorded() -> None:
    client = FakeLLMClient()
    _ = client.complete_json(system="SYS", user="USR")
    assert client.calls == [LLMCall(system="SYS", user="USR")]


def test_findings_reply_round_trips_through_the_schema() -> None:
    finding = Finding(
        filename="a.py", line=3, severity=Severity.WARNING, category=Category.BUG, message="m"
    )
    assert ReviewOutput.model_validate_json(findings_reply([finding])).findings == [finding]


# ---- the fallback chain ---------------------------------------------------------------------------


class HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class Failing:
    def __init__(self, error: Exception) -> None:
        self.error, self.calls = error, 0

    def complete_json(self, *, system: str, user: str) -> str:
        self.calls += 1
        raise self.error


def test_a_rate_limited_provider_hands_over_to_the_next() -> None:
    first, second = Failing(HttpError(429)), FakeLLMClient(["from second"])
    chain = FallbackClient([first, second], ["groq", "openrouter"])
    assert chain.complete_json(system="s", user="u") == "from second"
    assert chain.last_used == "openrouter" and first.calls == 1


def test_the_first_working_provider_wins_and_later_ones_are_untouched() -> None:
    second = FakeLLMClient(["never"])
    chain = FallbackClient([FakeLLMClient(["from first"]), second])
    assert chain.complete_json(system="s", user="u") == "from first"
    assert second.calls == []


@pytest.mark.parametrize("error", [HttpError(429), HttpError(503), TimeoutError("t"), ConnectionError("c")])
def test_provider_side_failures_are_retriable(error: Exception) -> None:
    assert is_retriable(error)


@pytest.mark.parametrize("error", [HttpError(401), HttpError(400), HttpError(404), ValueError("bug")])
def test_configuration_and_programming_errors_are_not_retriable(error: Exception) -> None:
    assert not is_retriable(error)


def test_a_bad_key_is_raised_not_hidden_behind_the_fallback() -> None:
    first, second = Failing(HttpError(401)), FakeLLMClient(["x"])
    with pytest.raises(HttpError):
        FallbackClient([first, second]).complete_json(system="s", user="u")
    assert second.calls == []


def test_when_every_provider_fails_the_last_error_is_raised() -> None:
    chain = FallbackClient([Failing(HttpError(429)), Failing(HttpError(503))])
    with pytest.raises(HttpError, match="503"):
        chain.complete_json(system="s", user="u")


def test_too_large_prompts_try_the_next_provider_then_surface_for_the_reviewer_to_shrink() -> None:
    chain = FallbackClient([Failing(PromptTooLargeError("a")), Failing(PromptTooLargeError("b"))])
    with pytest.raises(PromptTooLargeError):
        chain.complete_json(system="s", user="u")


def test_a_fallback_needs_at_least_one_client() -> None:
    with pytest.raises(ValueError):
        FallbackClient([])
