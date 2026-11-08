import pytest

from pr_agent.providers import (
    PROVIDERS, LLMConfigError, ResolvedProvider, configured_providers, describe, resolve, select_providers,
)

GROQ = {"GROQ_API_KEY": "gsk_secret_groq"}
HF = {"HF_TOKEN": "hf_secret_token"}


def names(env: dict[str, str]) -> list[str]:
    return [p.name for p in select_providers(env)]


def test_auto_uses_whatever_is_configured_and_nothing_else() -> None:
    assert names(GROQ) == ["groq"]
    assert names(HF) == ["huggingface"]
    assert names({**GROQ, **HF}) == ["groq", "huggingface"]


def test_each_provider_has_its_own_separate_variables() -> None:
    # Setting one provider's fields never switches another on.
    assert [p.name for p in configured_providers(HF)] == ["huggingface"]
    assert resolve(PROVIDERS["groq"], HF) is None
    assert resolve(PROVIDERS["huggingface"], GROQ) is None


def test_auto_follows_the_priority_order_not_the_order_variables_were_set() -> None:
    env = {"OPENROUTER_API_KEY": "k1", "GEMINI_API_KEY": "k2", "CEREBRAS_API_KEY": "k3", **GROQ}
    assert names(env) == ["groq", "cerebras", "gemini", "openrouter"]


def test_each_provider_has_a_default_model_and_its_own_override() -> None:
    assert select_providers(GROQ)[0].model == "openai/gpt-oss-120b"
    assert select_providers({**GROQ, "GROQ_MODEL": "openai/gpt-oss-20b"})[0].model == "openai/gpt-oss-20b"
    both = select_providers({**GROQ, **HF, "HF_MODEL": "some/model"})
    assert (both[0].model, both[1].model) == ("openai/gpt-oss-120b", "some/model")  # overrides don't bleed


def test_an_explicit_provider_overrides_auto() -> None:
    assert names({**GROQ, **HF, "LLM_PROVIDER": "huggingface"}) == ["huggingface"]


def test_an_explicit_chain_keeps_the_given_order() -> None:
    assert names({**GROQ, **HF, "LLM_PROVIDER": "huggingface, groq"}) == ["huggingface", "groq"]


def test_nothing_configured_lists_every_variable_it_could_use() -> None:
    with pytest.raises(LLMConfigError) as error:
        select_providers({})
    for variable in ("GROQ_API_KEY", "HF_TOKEN", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY", "GEMINI_API_KEY",
                     "OLLAMA_MODEL", "CUSTOM_BASE_URL"):
        assert variable in str(error.value)


def test_naming_an_unconfigured_provider_says_which_variable_to_set() -> None:
    with pytest.raises(LLMConfigError, match="HF_TOKEN"):
        select_providers({**GROQ, "LLM_PROVIDER": "huggingface"})


def test_an_unknown_provider_lists_the_known_ones() -> None:
    with pytest.raises(LLMConfigError, match="groq"):
        select_providers({**GROQ, "LLM_PROVIDER": "nonesuch"})


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_a_blank_key_counts_as_not_configured(blank: str) -> None:
    assert resolve(PROVIDERS["groq"], {"GROQ_API_KEY": blank}) is None


def test_surrounding_whitespace_in_a_key_is_stripped() -> None:
    assert select_providers({"GROQ_API_KEY": "  gsk_x \n"})[0].api_key == "gsk_x"


def test_ollama_needs_no_key_but_must_be_switched_on_with_a_model() -> None:
    assert resolve(PROVIDERS["ollama"], {}) is None
    local = select_providers({"OLLAMA_MODEL": "qwen2.5-coder:7b"})[0]
    assert local.base_url == "http://localhost:11434/v1" and local.api_key
    other = select_providers({"OLLAMA_MODEL": "m", "OLLAMA_BASE_URL": "http://box:11434/v1"})[0]
    assert other.base_url == "http://box:11434/v1"


def test_custom_needs_url_key_and_model_together() -> None:
    full = {"CUSTOM_BASE_URL": "https://x.example/v1", "CUSTOM_API_KEY": "k", "CUSTOM_MODEL": "m"}
    assert names(full) == ["custom"]
    for missing in full:
        assert resolve(PROVIDERS["custom"], {k: v for k, v in full.items() if k != missing}) is None


def test_keys_never_appear_in_repr_or_describe_or_errors() -> None:
    resolved = select_providers({**GROQ, **HF})
    for text in (repr(resolved[0]), repr(resolved), describe(resolved)):
        assert "secret" not in text
    with pytest.raises(LLMConfigError) as error:
        select_providers({**GROQ, "LLM_PROVIDER": "huggingface"})
    assert "secret" not in str(error.value)


def test_only_groq_is_marked_verified() -> None:
    assert [s.name for s in PROVIDERS.values() if s.verified] == ["groq"]


def test_unverified_providers_default_to_the_safe_capability_flags() -> None:
    for spec in PROVIDERS.values():
        if not spec.verified:
            assert not spec.json_mode and not spec.reasoning_effort


def test_resolved_provider_is_a_plain_value() -> None:
    assert isinstance(select_providers(GROQ)[0], ResolvedProvider)


# ---- cost: auto must never spend money on its own ------------------------------------------------------

PAID = {"DEEPSEEK_API_KEY": "sk-paid", "OPENAI_API_KEY": "sk-paid2"}


def test_auto_skips_paid_providers_even_when_their_keys_are_set() -> None:
    assert names({**GROQ, **PAID}) == ["groq"]


def test_auto_with_only_paid_keys_is_an_error_not_a_silent_charge() -> None:
    with pytest.raises(LLMConfigError):
        select_providers(PAID)


def test_a_paid_provider_is_used_when_named_explicitly() -> None:
    assert names({**PAID, "LLM_PROVIDER": "deepseek"}) == ["deepseek"]
    assert names({**GROQ, **PAID, "LLM_PROVIDER": "groq,deepseek"}) == ["groq", "deepseek"]


def test_allow_paid_adds_paid_providers_after_the_free_ones() -> None:
    assert names({**GROQ, **PAID, "LLM_ALLOW_PAID": "1"}) == ["groq", "deepseek", "openai"]


@pytest.mark.parametrize("off", ["0", "", "no", "false"])
def test_allow_paid_is_off_unless_clearly_enabled(off: str) -> None:
    assert names({**GROQ, **PAID, "LLM_ALLOW_PAID": off}) == ["groq"]


def test_free_credit_providers_come_after_free_tiers_in_auto() -> None:
    env = {"FIREWORKS_API_KEY": "k", "NVIDIA_API_KEY": "k", "HF_TOKEN": "k", **GROQ}
    assert names(env) == ["groq", "huggingface", "nvidia", "fireworks"]


def test_describe_marks_anything_that_is_not_free() -> None:
    text = describe(select_providers({"FIREWORKS_API_KEY": "k", **GROQ}))
    assert "fireworks:" in text and "[CREDITS]" in text and "groq:openai/gpt-oss-120b ->" in text
    assert "[FREE]" not in text


def test_a_model_with_no_sensible_default_must_be_chosen_by_the_user() -> None:
    assert resolve(PROVIDERS["moonshot"], {"MOONSHOT_API_KEY": "k"}) is None
    assert resolve(PROVIDERS["moonshot"], {"MOONSHOT_API_KEY": "k", "MOONSHOT_MODEL": "kimi-x"}) is not None


def test_no_two_providers_share_an_environment_variable() -> None:
    """The 'separate fields' guarantee: setting one provider's variable can never affect another."""
    seen: dict[str, str] = {}
    for spec in PROVIDERS.values():
        for variable in (spec.key_env, spec.model_env, spec.base_url_env):
            if variable:
                assert variable not in seen, f"{variable} used by both {seen[variable]} and {spec.name}"
                seen[variable] = spec.name


def test_github_models_does_not_reuse_the_github_pull_request_token() -> None:
    assert PROVIDERS["github"].key_env == "GITHUB_MODELS_TOKEN"
    assert "GITHUB_TOKEN" not in {v for s in PROVIDERS.values() for v in (s.key_env, s.model_env)}


def test_every_provider_has_a_valid_cost_and_only_unverified_ones_beyond_groq() -> None:
    assert {s.cost for s in PROVIDERS.values()} <= {"free", "credits", "paid"}
    assert {n for n, s in PROVIDERS.items() if s.cost == "paid"} == {"deepseek", "moonshot", "openai", "anthropic"}


def test_every_variable_in_the_registry_is_documented_in_env_example() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    documented = {line.split("=")[0] for line in text.splitlines() if "=" in line and not line.startswith("#")}
    for spec in PROVIDERS.values():
        for variable in (spec.key_env, spec.model_env, spec.base_url_env):
            if variable:
                assert variable in documented, f"{variable} is missing from .env.example"
    assert {"LLM_PROVIDER", "LLM_ALLOW_PAID", "GITHUB_TOKEN"} <= documented
