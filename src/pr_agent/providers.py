"""Which LLM providers exist, and which of them are configured.

Each provider has its OWN clearly named environment variables (`GROQ_API_KEY`,
`HF_TOKEN`, ...), so nothing is shared or ambiguous. By default (`LLM_PROVIDER=auto`)
every provider whose variables are set is used, in the priority order below, as a
fallback chain: when the first runs out of free-tier quota the next takes over.

Almost every free service speaks the same OpenAI-style chat API, so a provider is only
data here: a base URL, the variable names, and a few capability flags.

`auto` never falls back onto a PAID provider: that would spend money without anyone choosing
to. Paid providers are used only when named in LLM_PROVIDER, or with LLM_ALLOW_PAID=1.

Only Groq has been verified against a live account. The other defaults come from
documentation and may have changed; run `python experiments/list_models.py <provider>`
to see what your key can actually use, and set `<PROVIDER>_MODEL` accordingly.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LLMConfigError(RuntimeError):
    """The LLM configuration is unusable: unknown provider, missing key, nothing configured."""


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    default_base_url: str | None
    key_env: str | None  # None: needs no key (local Ollama)
    model_env: str
    default_model: str | None  # None: the user must choose (there is no sensible default)
    base_url_env: str | None = None
    # Capability flags. A flag left False is the safe choice: the reviewer already copes
    # with fenced or chatty JSON, whereas an unsupported parameter is a hard 400 error.
    json_mode: bool = False  # response_format={"type": "json_object"}
    reasoning_effort: bool = False  # gpt-oss style `reasoning_effort`
    tokens_per_minute: int | None = None  # for client-side pacing; None = rely on 429 + retry
    verified: bool = False  # exercised against a live account
    # "free": a free tier with rate limits. "credits": free signup credit that, once spent,
    # may start billing if a card is on file. "paid": costs money from the first call.
    cost: str = "free"
    note: str = ""


# Priority order for `auto`: free tiers first, then free-credit services, then local, and
# paid providers last (and only when explicitly allowed).
PROVIDERS: dict[str, ProviderSpec] = {spec.name: spec for spec in [
    # ---- free tiers ---------------------------------------------------------------------
    ProviderSpec("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_MODEL",
                 "openai/gpt-oss-120b", json_mode=True, reasoning_effort=True,
                 tokens_per_minute=8_000, verified=True,
                 note="free tier: 1000 requests/day, 8000 tokens/minute (measured)"),
    ProviderSpec("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "CEREBRAS_MODEL",
                 "gpt-oss-120b", note="free tier; check current limits in its console"),
    ProviderSpec("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/",
                 "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.5-flash",
                 note="Google AI Studio free tier"),
    ProviderSpec("github", "https://models.github.ai/inference", "GITHUB_MODELS_TOKEN",
                 "GITHUB_MODELS_MODEL", "openai/gpt-4.1-mini",
                 note="GitHub Models: free, rate-limited; a GitHub token with the models:read permission"),
    ProviderSpec("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", "MISTRAL_MODEL",
                 "mistral-small-latest", note="La Plateforme 'Experiment' free tier"),
    ProviderSpec("sambanova", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY", "SAMBANOVA_MODEL",
                 "gpt-oss-120b", note="SambaNova Cloud free tier"),
    ProviderSpec("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                 "OPENROUTER_MODEL", "openai/gpt-oss-20b:free",
                 note="models ending in :free; small daily request limit"),
    ProviderSpec("huggingface", "https://router.huggingface.co/v1", "HF_TOKEN", "HF_MODEL",
                 "openai/gpt-oss-120b",
                 note="Inference Providers; the token needs that permission; small monthly free credit"),
    ProviderSpec("ollama", "http://localhost:11434/v1", None, "OLLAMA_MODEL", None,
                 base_url_env="OLLAMA_BASE_URL",
                 note="local model, no quota; set OLLAMA_MODEL to turn it on"),
    # ---- free signup credit (can bill once spent, if a card is on file) -------------------
    ProviderSpec("nvidia", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", "NVIDIA_MODEL",
                 "openai/gpt-oss-120b", cost="credits", note="NVIDIA NIM trial credits"),
    ProviderSpec("fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY",
                 "FIREWORKS_MODEL", "accounts/fireworks/models/gpt-oss-120b", cost="credits",
                 note="small signup credit, then paid"),
    ProviderSpec("together", "https://api.together.xyz/v1", "TOGETHER_API_KEY", "TOGETHER_MODEL",
                 "openai/gpt-oss-120b", cost="credits", note="signup credit may not be available; check"),
    # ---- paid: never used by `auto` -------------------------------------------------------
    ProviderSpec("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL",
                 "deepseek-chat", cost="paid", note="very cheap, but not free"),
    ProviderSpec("moonshot", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY", "MOONSHOT_MODEL",
                 None, cost="paid", note="Kimi; you must choose a model id from its console"),
    ProviderSpec("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "OPENAI_MODEL",
                 "gpt-4.1-mini", cost="paid", note="ChatGPT's API; no free tier"),
    ProviderSpec("anthropic", "https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
                 "claude-haiku-5-5", cost="paid",
                 note="Claude via Anthropic's OpenAI-compatibility endpoint; no free tier"),
    # ---- anything else -----------------------------------------------------------------------
    ProviderSpec("custom", None, "CUSTOM_API_KEY", "CUSTOM_MODEL", None,
                 base_url_env="CUSTOM_BASE_URL",
                 note="any other OpenAI-compatible service; needs CUSTOM_BASE_URL, CUSTOM_API_KEY, CUSTOM_MODEL"),
]}


@dataclass(frozen=True)
class ResolvedProvider:
    """A provider with everything needed to call it. `api_key` must never be logged."""

    spec: ProviderSpec
    api_key: str
    model: str
    base_url: str

    @property
    def name(self) -> str:
        return self.spec.name

    def __repr__(self) -> str:  # so a stray print() or traceback can't leak the key
        return f"ResolvedProvider(name={self.name!r}, model={self.model!r}, base_url={self.base_url!r})"


def _value(env: Mapping[str, str], name: str | None) -> str:
    return (env.get(name, "") if name else "").strip()


def resolve(spec: ProviderSpec, env: Mapping[str, str]) -> ResolvedProvider | None:
    """The provider if every variable it needs is set, else None."""
    key = _value(env, spec.key_env) if spec.key_env else "not-needed"
    model = _value(env, spec.model_env) or (spec.default_model or "")
    base_url = _value(env, spec.base_url_env) or (spec.default_base_url or "")
    if not (key and model and base_url):
        return None
    return ResolvedProvider(spec, key, model, base_url)


def _variables(spec: ProviderSpec) -> str:
    names = [n for n in (spec.key_env, spec.base_url_env) if n]
    if spec.default_model is None:
        names.append(spec.model_env)
    return ", ".join(names) or spec.model_env


def configured_providers(env: Mapping[str, str]) -> list[ResolvedProvider]:
    """Every provider whose variables are set, in priority order (paid ones included)."""
    return [r for spec in PROVIDERS.values() if (r := resolve(spec, env)) is not None]


def _allows_paid(env: Mapping[str, str]) -> bool:
    return _value(env, "LLM_ALLOW_PAID").lower() in ("1", "true", "yes")


def select_providers(env: Mapping[str, str]) -> list[ResolvedProvider]:
    """Apply LLM_PROVIDER: `auto` (default), one name, or a comma-separated chain."""
    choice = _value(env, "LLM_PROVIDER").lower() or "auto"
    if choice == "auto":
        found = [p for p in configured_providers(env) if p.spec.cost != "paid" or _allows_paid(env)]
        if not found:
            options = "; ".join(f"{s.name}: {_variables(s)}" for s in PROVIDERS.values())
            raise LLMConfigError(
                "No LLM provider is configured. Set the variables for at least one in .env "
                f"(see .env.example). Options: {options}")
        return found

    chosen: list[ResolvedProvider] = []
    for name in (part.strip() for part in choice.split(",") if part.strip()):
        spec = PROVIDERS.get(name)
        if spec is None:
            raise LLMConfigError(f"Unknown LLM provider {name!r}. Known: {', '.join(PROVIDERS)}, or 'auto'.")
        resolved = resolve(spec, env)
        if resolved is None:
            raise LLMConfigError(f"LLM_PROVIDER includes {name!r} but it is not configured. "
                                 f"Set: {_variables(spec)}")
        chosen.append(resolved)
    return chosen


def describe(providers: list[ResolvedProvider]) -> str:
    """A safe one-line summary for logs: names and models, never keys. Marks anything not free."""
    return " -> ".join(
        f"{p.name}:{p.model}" + ("" if p.spec.cost == "free" else f" [{p.spec.cost.upper()}]") for p in providers)


# Per-run spending caps. A run is one invocation of the agent. The defaults are far above a
# normal review and far below an accident, so a bug or a huge PR cannot silently burn a day's
# free quota or run up a paid bill. Each provider in a chain gets its own cap.
DEFAULT_MAX_CALLS_PER_RUN = 100
DEFAULT_MAX_TOKENS_PER_RUN = 200_000
_UNLIMITED = ("none", "off", "unlimited", "0")


def _cap(env: Mapping[str, str], name: str, default: int) -> int | None:
    raw = _value(env, name).lower()
    if not raw:
        return default
    if raw in _UNLIMITED:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise LLMConfigError(f"{name} must be a whole number, or 'none' for no cap (got {raw!r})") from None
    if value < 0:
        raise LLMConfigError(f"{name} cannot be negative (got {value})")
    return value


def run_caps(env: Mapping[str, str]) -> tuple[int | None, int | None]:
    """(max calls, max tokens) per provider per run, from LLM_MAX_CALLS_PER_RUN / LLM_MAX_TOKENS_PER_RUN."""
    return (_cap(env, "LLM_MAX_CALLS_PER_RUN", DEFAULT_MAX_CALLS_PER_RUN),
            _cap(env, "LLM_MAX_TOKENS_PER_RUN", DEFAULT_MAX_TOKENS_PER_RUN))
