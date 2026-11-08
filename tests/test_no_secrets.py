"""Fail if a secret-shaped string is committed.

GitHub push protection rejects a push containing anything that looks like a live credential,
fake or not, and it scans every commit in the push, so a string that was committed once and
removed later still blocks it. A key-shaped placeholder in a test fixture did exactly that.
Build such strings at run time from pieces instead (`"sk_" + "live_" + ...`).

Only files tracked by git are scanned, so a local, git-ignored .env is never read.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# (name, pattern). These patterns contain brackets where a real key has letters, so this file
# does not match its own rules.
SECRET_PATTERNS = {
    "Stripe key": r"\b[sr]k_(live|test)_[0-9a-zA-Z]{16,}",
    "AWS access key id": r"\bAKIA[0-9A-Z]{16}\b",
    "GitHub token": r"\b(ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{30,}",
    "GitHub fine-grained token": r"\bgithub_pat_[A-Za-z0-9_]{30,}",
    "Slack token": r"\bxox[baprs]-[A-Za-z0-9-]{10,}",
    "Google API key": r"\bAIza[0-9A-Za-z_-]{35}",
    "Groq key": r"\bgsk_[A-Za-z0-9]{30,}",
    "Hugging Face token": r"\bhf_[A-Za-z0-9]{30,}",
    "OpenAI-style key": r"\bsk-[A-Za-z0-9]{30,}",
    "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}


def find_secrets(text: str) -> list[str]:
    return [name for name, pattern in SECRET_PATTERNS.items() if re.search(pattern, text)]


def tracked_files() -> list[Path]:
    try:
        listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / name for name in listing.decode().split("\0") if name]


def test_no_tracked_file_contains_a_secret_shaped_string() -> None:
    offenders: list[str] = []
    for path in tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: not source text
        offenders += [f"{path.relative_to(ROOT)}: {name}" for name in find_secrets(text)]
    assert not offenders, "secret-shaped strings are committed (GitHub will refuse the push):\n" + "\n".join(offenders)


@pytest.mark.parametrize(("name", "sample"), [
    ("Stripe key", "sk_" + "live_" + "A" * 24),
    ("Stripe key", "rk_" + "test_" + "b1" * 12),
    ("AWS access key id", "AKIA" + "ABCDEFGHIJKLMNOP"),
    ("GitHub token", "ghp_" + "a" * 36),
    ("GitHub fine-grained token", "github_pat_" + "A1_" * 12),
    ("Groq key", "gsk_" + "x" * 40),
    ("Hugging Face token", "hf_" + "y" * 34),
    ("OpenAI-style key", "sk-" + "z" * 48),
    ("private key block", "-----BEGIN " + "RSA PRIVATE KEY-----"),
])
def test_the_scanner_recognises_each_kind_of_secret(name: str, sample: str) -> None:
    """If this fails the guard above has gone blind, which is worse than having no guard."""
    assert name in find_secrets(f'API_KEY = "{sample}"')


@pytest.mark.parametrize("harmless", [
    "API_KEY = os.environ['GROQ_API_KEY']", "GROQ_API_KEY=gsk_...", "sk_live_ is the prefix",
    "token = 'ghp_'", "OPENAI_API_KEY=", "hf_ prefix",
])
def test_placeholders_and_variable_names_are_not_flagged(harmless: str) -> None:
    assert find_secrets(harmless) == []
