"""List the model ids each configured provider's key can use.

    python experiments/list_models.py              # every configured provider
    python experiments/list_models.py openrouter   # one provider

Defaults for providers other than Groq come from documentation and may be stale; this is how
you find the real ids, then set <PROVIDER>_MODEL in .env. Makes one cheap listing request per
provider (it does not spend model tokens). Prints model ids only, never keys.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pr_agent.env import load_env  # noqa: E402
from pr_agent.providers import configured_providers  # noqa: E402


def main() -> None:
    load_env()
    from openai import OpenAI

    wanted = {name.lower() for name in sys.argv[1:]}
    found = [p for p in configured_providers(os.environ) if not wanted or p.name in wanted]
    if not found:
        print("No matching configured provider. Fill in its fields in .env (see .env.example).")
        return
    for provider in found:
        print(f"\n== {provider.name} (default model: {provider.model}) ==")
        try:
            models = sorted(m.id for m in OpenAI(base_url=provider.base_url, api_key=provider.api_key,
                                                 max_retries=0, timeout=30).models.list().data)
        except Exception as error:  # noqa: BLE001 - report and carry on with the next provider
            print(f"  could not list models: {type(error).__name__}: {str(error)[:150]}")
            continue
        for model_id in models[:60]:
            print("  ", model_id)
        if len(models) > 60:
            print(f"   ... and {len(models) - 60} more")


if __name__ == "__main__":
    main()
