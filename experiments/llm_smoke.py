"""One tiny live call through whatever provider(s) your .env configures.

    python experiments/llm_smoke.py

Sends a few dozen tokens and no repository code. Prints provider, model, latency and token
counts. It never prints a key.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pr_agent.chat_client import make_client  # noqa: E402
from pr_agent.env import load_env  # noqa: E402
from pr_agent.providers import describe, select_providers  # noqa: E402
from pr_agent.reviewer import parse_review  # noqa: E402


def main() -> None:
    load_env()
    import os

    print("providers:", describe(select_providers(os.environ)))
    client = make_client(max_calls=1)
    started = time.perf_counter()
    raw = client.complete_json(
        system='You are a test. Reply with exactly this JSON object and nothing else: {"findings": []}',
        user="Respond now.",
    )
    print(f"latency:  {time.perf_counter() - started:.2f}s")
    print(f"raw:      {raw!r}")
    print(f"parsed:   {parse_review(raw)!r}")


if __name__ == "__main__":
    main()
