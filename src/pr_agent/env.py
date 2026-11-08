"""Loading `.env`, in one place."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path | None = None) -> None:
    """Load `.env` from the project root into the process environment.

    The path is explicit because python-dotenv's own search starts from the calling file
    and fails when called from a REPL or stdin. `override=False` means a real environment
    variable (a CI secret, an `export`) always wins over the file, which is what you want
    when the same code later runs in GitHub Actions.
    """
    _ = load_dotenv(path or PROJECT_ROOT / ".env", override=False)
