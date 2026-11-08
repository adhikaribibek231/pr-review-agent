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

    # ---- harder bugs (subtler, closer to real review work) -----------------------------------
    Case("mutable_default_argument",
         before={"tags.py": "def normalize(tag):\n    return tag.strip().lower()\n"},
         after={"tags.py": '''
            def normalize(tag):
                return tag.strip().lower()

            def add_tag(tag, tags=[]):
                tags.append(normalize(tag))
                return tags
         '''},
         bug_file="tags.py", bug_text="def add_tag(tag, tags=[])",
         description="the default list is shared between calls"),
    Case("identity_comparison_on_string",
         before={"orders.py": "def total(o):\n    return o.amount\n"},
         after={"orders.py": '''
            def total(o):
                return o.amount

            def is_active(order):
                if order.status is "active":
                    return True
                return False
         '''},
         bug_file="orders.py", bug_text='order.status is "active"',
         description="`is` compares identity, not equality"),
    Case("swallowed_exception",
         before={"store.py": "def load(path):\n    return open(path).read()\n"},
         after={"store.py": '''
            def load(path):
                return open(path).read()

            def save(record, db):
                try:
                    db.insert(record)
                except Exception:
                    pass
         '''},
         bug_file="store.py", bug_text="except Exception:",
         description="a failed insert is silently discarded"),
    Case("shell_injection",
         before={"images.py": "import subprocess\n"},
         after={"images.py": '''
            import subprocess

            def convert(filename):
                subprocess.run(f"convert {filename} out.png", shell=True)
         '''},
         bug_file="images.py", bug_text="shell=True",
         description="a filename is interpolated into a shell command"),
    Case("weak_password_hash",
         before={"auth.py": "import hashlib\n"},
         after={"auth.py": '''
            import hashlib

            def hash_password(password):
                return hashlib.md5(password.encode()).hexdigest()
         '''},
         bug_file="auth.py", bug_text="hashlib.md5(password",
         description="MD5 is unsuitable for storing passwords"),
    Case("unsafe_deserialization",
         before={"api.py": "import json\n"},
         after={"api.py": '''
            import json
            import pickle

            def parse(request_body):
                return pickle.loads(request_body)
         '''},
         bug_file="api.py", bug_text="pickle.loads(request_body)",
         description="unpickling data from a request allows code execution"),
    Case("path_traversal",
         before={"files.py": "import os\n\nUPLOAD_DIR = '/srv/uploads'\n"},
         after={"files.py": '''
            import os

            UPLOAD_DIR = '/srv/uploads'

            def read_upload(filename):
                with open(os.path.join(UPLOAD_DIR, filename)) as handle:
                    return handle.read()
         '''},
         bug_file="files.py", bug_text="os.path.join(UPLOAD_DIR, filename)",
         description="a user-supplied filename can escape the upload directory with .."),
    Case("always_true_condition",
         before={"perm.py": "def is_staff(user):\n    return user.staff\n"},
         after={"perm.py": '''
            def is_staff(user):
                return user.staff

            def can_delete(user):
                if user.role == "admin" or "superuser":
                    return True
                return False
         '''},
         bug_file="perm.py", bug_text='user.role == "admin" or "superuser"',
         description="operator precedence makes the check always true: everyone may delete"),
    Case("missing_await",
         before={"users.py": "async def fetch_user(uid):\n    return await db.get(uid)\n"},
         after={"users.py": "async def fetch_user(uid):\n    return await db.get(uid)\n",
                "handler.py": '''
            from users import fetch_user

            async def handler(uid):
                result = fetch_user(uid)
                return result.name
         '''},
         bug_file="handler.py", bug_text="result = fetch_user(uid)",
         description="a coroutine is used without being awaited"),
    # ---- impact: the bug is in code the PR did not touch ------------------------------------------
    Case("forgotten_caller_after_rename",
         before={"tax.py": "def compute_tax(amount):\n    return amount * 0.13\n",
                 "invoice.py": "from tax import compute_tax\n\ndef invoice_total(amount):\n    return amount + compute_tax(amount)\n",
                 "report.py": "from tax import compute_tax\n\ndef report_tax(amounts):\n    return sum(compute_tax(a) for a in amounts)\n"},
         after={"tax.py": "def calculate_tax(amount):\n    return amount * 0.13\n",
                "invoice.py": "from tax import calculate_tax\n\ndef invoice_total(amount):\n    return amount + calculate_tax(amount)\n",
                "report.py": "from tax import compute_tax\n\ndef report_tax(amounts):\n    return sum(compute_tax(a) for a in amounts)\n"},
         bug_file="tax.py", bug_text="def calculate_tax(amount)",
         description="renamed; invoice.py was updated but report.py (untouched by the PR) still imports the old name"),
    Case("field_added_constructor_caller",
         before={"model.py": "from dataclasses import dataclass\n\n@dataclass\nclass Order:\n    id: int\n    total: float\n",
                 "build.py": "from model import Order\n\ndef make():\n    return Order(1, 9.5)\n"},
         after={"model.py": "from dataclasses import dataclass\n\n@dataclass\nclass Order:\n    id: int\n    total: float\n    currency: str\n",
                "build.py": "from model import Order\n\ndef make():\n    return Order(1, 9.5)\n"},
         bug_file="model.py", bug_text="currency: str",
         description="a required field was added; build.py (untouched) still constructs Order with two arguments"),
    Case("removed_function_still_used",
         before={"utils.py": "def keep(x):\n    return x\n\n\ndef legacy_total(items):\n    return sum(items)\n\n\ndef other(x):\n    return x\n",
                 "report.py": "from utils import legacy_total\n\ndef report(items):\n    return legacy_total(items)\n"},
         after={"utils.py": "def keep(x):\n    return x\n\n\ndef other(x):\n    return x\n",
                "report.py": "from utils import legacy_total\n\ndef report(items):\n    return legacy_total(items)\n"},
         bug_file="utils.py", bug_text="def other(x)",
         description="a function was deleted but report.py (untouched) still imports it; the diff has no added line to attach a comment to"),
    # ---- more clean changes: a good reviewer stays silent -------------------------------------------
    Case("clean_type_hints_only",
         before={"geo.py": "def area(w, h):\n    return w * h\n"},
         after={"geo.py": "def area(w: float, h: float) -> float:\n    return w * h\n"},
         description="annotations added, behaviour identical: expect NO findings"),
    Case("clean_safe_subprocess",
         before={"images.py": "import subprocess\n"},
         after={"images.py": '''
            import subprocess

            def convert(filename):
                subprocess.run(["convert", filename, "out.png"], check=True)
         '''},
         description="argument list, no shell: expect NO findings"),
    Case("clean_mutable_default_fixed",
         before={"tags.py": "def add_tag(tag, tags=[]):\n    tags.append(tag)\n    return tags\n"},
         after={"tags.py": "def add_tag(tag, tags=None):\n    if tags is None:\n        tags = []\n    tags.append(tag)\n    return tags\n"},
         description="the classic bug being FIXED: expect NO findings"),
    Case("clean_context_manager",
         before={"io_utils.py": "def read(path):\n    f = open(path)\n    data = f.read()\n    f.close()\n    return data\n"},
         after={"io_utils.py": "def read(path):\n    with open(path) as handle:\n        return handle.read()\n"},
         description="a correct simplification: expect NO findings"),
    Case("clean_new_tests",
         before={"calc.py": "def add(a, b):\n    return a + b\n"},
         after={"calc.py": "def add(a, b):\n    return a + b\n",
                "tests/test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n\ndef test_add_negative():\n    assert add(-1, 1) == 0\n"},
         description="only tests added: expect NO findings"),
    Case("clean_rename_local_variable",
         before={"stats.py": "def mean(xs):\n    t = sum(xs)\n    return t / len(xs) if xs else 0.0\n"},
         after={"stats.py": "def mean(xs):\n    total = sum(xs)\n    return total / len(xs) if xs else 0.0\n"},
         description="a local variable renamed, empty input still handled: expect NO findings"),
]
