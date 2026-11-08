from pr_agent.llm import NOTHING_FOUND, FakeLLMClient, LLMCall, findings_reply
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
