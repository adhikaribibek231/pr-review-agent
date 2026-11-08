"""Synthetic pull requests with known bugs, for measuring a reviewer.

Each case is a repository before and after a change, plus where the bug is (found by text,
so edits to the sources can't make the line number stale). `clean` cases contain no bug: a
good reviewer must stay silent on them. All code here is invented; nothing from a real
codebase is ever sent to a model.
"""

import difflib
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pr_agent.diff_parser import DiffHunk, parse_hunks  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    before: dict[str, str]
    after: dict[str, str]
    bug_file: str | None = None  # None for a clean case
    bug_text: str | None = None  # a substring of the buggy line in the AFTER file
    description: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return self.bug_file is None

    def bug_line(self) -> int | None:
        if self.bug_file is None or self.bug_text is None:
            return None
        for number, line in enumerate(_lines(self.after[self.bug_file]), start=1):
            if self.bug_text in line:
                return number
        raise LookupError(f"{self.name}: {self.bug_text!r} not found in {self.bug_file}")


def _lines(source: str) -> list[str]:
    return textwrap.dedent(source).lstrip("\n").splitlines()


def write_repo(root: Path, files: dict[str, str]) -> None:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text("\n".join(_lines(source)) + "\n", encoding="utf-8")


def hunks_of(case: Case) -> list[DiffHunk]:
    hunks: list[DiffHunk] = []
    for name in sorted(set(case.before) | set(case.after)):
        old = _lines(case.before[name]) if name in case.before else []
        new = _lines(case.after[name]) if name in case.after else []
        if old == new:
            continue
        diff = list(difflib.unified_diff(old, new, lineterm="", n=3))
        hunks.extend(parse_hunks(name, "\n".join(x for x in diff if not x.startswith(("--- ", "+++ ")))))
    return hunks


USERS = """
def find_user(users, uid):
    for user in users:
        if user.id == uid:
            return user
"""

SHOP_PRICING_V1 = "def apply_discount(price, rate):\n    return price * (1 - rate)\n"
SHOP_PRICING_V2 = "def apply_discount(price, rate, floor):\n    return max(price * (1 - rate), floor)\n"
SHOP_CHECKOUT = """
from shop.pricing import apply_discount

def checkout_total(cart):
    return sum(apply_discount(item.price, item.rate) for item in cart)
"""

# Assembled at run time so this file never contains a complete secret-shaped string:
# GitHub push protection blocks anything that looks like a live Stripe key, fake or not.
_FAKE_KEY = "sk_" + "live_" + "EXAMPLEEXAMPLEEXAMPLE123456"

CASES: list[Case] = [
    Case("sql_injection",
         before={"app/db.py": "import sqlite3\n"},
         after={"app/db.py": '''
            import sqlite3

            def find_user(conn, name):
                query = f"SELECT * FROM users WHERE name = '{name}'"
                return conn.execute(query).fetchall()
         '''},
         bug_file="app/db.py", bug_text="query = f\"SELECT",
         description="user input interpolated into SQL"),
    Case("hardcoded_secret",
         before={"app/client.py": "import requests\n"},
         after={"app/client.py": '''
            import requests

            API_KEY = "__FAKE_KEY__"

            def fetch_orders():
                headers = {"Authorization": "Bearer " + API_KEY}
                return requests.get("https://api.example.com/orders", headers=headers)
         '''.replace("__FAKE_KEY__", _FAKE_KEY)},
         bug_file="app/client.py", bug_text="API_KEY = ",
         description="a live-looking API key committed in source"),
    Case("off_by_one",
         before={"stats.py": "def total(xs):\n    return sum(xs)\n"},
         after={"stats.py": '''
            def total(xs):
                return sum(xs)

            def last_three(items):
                result = []
                for i in range(len(items) - 3, len(items) + 1):
                    result.append(items[i])
                return result
         '''},
         bug_file="stats.py", bug_text="range(len(items) - 3, len(items) + 1)",
         description="loop runs one past the end of the list"),
    Case("missing_none_check",
         before={"users.py": USERS},
         after={"users.py": USERS, "greet.py": '''
            from users import find_user

            def greeting(users, uid):
                user = find_user(users, uid)
                return "Hello, " + user.name
         '''},
         bug_file="greet.py", bug_text='"Hello, " + user.name',
         description="find_user returns None when nobody matches; the result is dereferenced"),
    Case("divide_by_zero",
         before={"metrics.py": "def count(xs):\n    return len(xs)\n"},
         after={"metrics.py": '''
            def count(xs):
                return len(xs)

            def average(values):
                return sum(values) / len(values)
         '''},
         bug_file="metrics.py", bug_text="sum(values) / len(values)",
         description="empty input raises ZeroDivisionError"),
    Case("signature_change_forgotten_caller",
         before={"shop/pricing.py": SHOP_PRICING_V1, "shop/checkout.py": SHOP_CHECKOUT,
                 "shop/emails.py": "def receipt(order):\n    return f'Thanks {order.id}'\n"},
         after={"shop/pricing.py": SHOP_PRICING_V2, "shop/checkout.py": SHOP_CHECKOUT,
                "shop/emails.py": "def receipt(order):\n    return f'Thanks {order.id}'\n"},
         bug_file="shop/pricing.py", bug_text="def apply_discount(price, rate, floor)",
         description="a required parameter was added; checkout.py (untouched) still passes two"),
    Case("clean_refactor",
         before={"cart.py": '''
            def total(items):
                t = 0
                for i in items:
                    t = t + i.price
                return t
         '''},
         after={"cart.py": '''
            def total(items):
                return sum(item.price for item in items)
         '''},
         description="behaviour-preserving simplification: expect NO findings"),
    Case("clean_safe_query",
         before={"app/db.py": "import sqlite3\n"},
         after={"app/db.py": '''
            import sqlite3

            def find_user(conn, name):
                return conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchall()
         '''},
         description="a correctly parameterised query: expect NO findings"),
]
