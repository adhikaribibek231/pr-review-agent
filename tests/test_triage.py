import pytest

from pr_agent.diff_parser import parse_hunks
from pr_agent.triage import is_trivial_hunk, triage_hunks


def hunk(body: str, filename: str = "a.py"):
    return parse_hunks(filename, body)[0]


@pytest.mark.parametrize(
    "patch",
    [
        "@@ -1,1 +1,1 @@\n-x = 1\n+    x = 1",  # re-indented, same code
        "@@ -1,2 +1,2 @@\n-a = 1\n-b = 2\n+b = 2\n+a = 1",  # reordered
        "@@ -1,1 +1,2 @@\n x = 1\n+",  # added a blank line
        "@@ -1,0 +1,2 @@\n+# explain why\n+# and how",  # comments only
        "@@ -1,0 +1,4 @@\n+import os\n+from pathlib import Path\n+import numpy as np, sys\n+from . import sibling  # local",  # imports only
        "@@ -1,1 +1,1 @@\n-import a\n+import b",  # swapped import
    ],
)
def test_trivial_changes_are_recognised(patch: str) -> None:
    assert is_trivial_hunk(hunk(patch))


@pytest.mark.parametrize(
    "patch",
    [
        "@@ -1,1 +1,1 @@\n-x = 1\n+x = 2",  # a value changed
        "@@ -1,0 +1,2 @@\n+# comment\n+x = compute()",  # comment AND code
        "@@ -1,0 +1,1 @@\n+import os; os.system(cmd)",  # statement hiding after an import
        "@@ -1,0 +1,1 @@\n+from os import path; run(path)",
        "@@ -1,0 +1,1 @@\n+from x import (a, b)",  # parenthesised: stay conservative
        "@@ -1,2 +1,0 @@\n-if user.is_admin:\n-    allow()",  # a check was deleted
        "@@ -1,0 +1,1 @@\n+return None",
    ],
)
def test_real_changes_are_not_trivial(patch: str) -> None:
    assert not is_trivial_hunk(hunk(patch))


def test_triage_splits_hunks_and_preserves_order() -> None:
    boring = hunk("@@ -1,0 +1,1 @@\n+# note")
    real = hunk("@@ -5,1 +5,1 @@\n-a = 1\n+a = 2")
    other = hunk("@@ -9,0 +9,1 @@\n+run()")
    result = triage_hunks([real, boring, other])
    assert result.reviewable == (real, other)
    assert result.skipped == (boring,)


def test_triage_of_nothing_is_empty() -> None:
    result = triage_hunks([])
    assert result.reviewable == () and result.skipped == ()
