"""One test per kind of change. Each builds a real before/after repository.

Read these as the specification of what "impact" means: for every way a change can
ripple into code the PR did not touch, what must be found, what must NOT be found,
and how it is ranked. docs/retrieval_scenarios.md maps them to the full taxonomy.
"""

from pathlib import Path
from typing import NamedTuple

from helpers import make_pr

from pr_agent.impact import (
    BEHAVIOUR, CONTRACT, DANGLING, DATA, MAX_IMPACT_SITES, MAX_TEST_SITES, POSSIBLE, TEST, TEXTUAL,
    ImpactItem, analyze_impact, changed_symbols,
)
from pr_agent.retriever import discover_python_files
from pr_agent.symbols import SymbolIndex


class Outcome(NamedTuple):
    items: list[ImpactItem]
    notes: list[str]

    @property
    def files(self) -> set[str]:
        return {i.file for i in self.items}

    def item(self, file: str) -> ImpactItem:
        matches = [i for i in self.items if i.file == file]
        assert matches, f"{file} not in impact; got {sorted(self.files)}"
        return matches[0]


def analyze(tmp_path: Path, before: dict[str, str], after: dict[str, str]) -> Outcome:
    pr = make_pr(tmp_path, before, after)
    symbols = SymbolIndex.build(tmp_path, discover_python_files(tmp_path))
    items: list[ImpactItem] = []
    notes: list[str] = []
    for hunk in pr.hunks:
        report = analyze_impact(symbols, hunk, pr.hunks)
        items += report.items
        notes += report.notes
    return Outcome(items, notes)


PRICING_V1 = """
def apply_discount(price, rate):
    return price * (1 - rate)
"""
PRICING_V2 = """
def apply_discount(price, rate, floor):
    return max(price * (1 - rate), floor)
"""

CALLERS = {
    "shop/checkout.py": """
        from shop.pricing import apply_discount

        def checkout_total(cart):
            return sum(apply_discount(i.price, i.rate) for i in cart)
    """,
    "shop/invoice.py": """
        import shop.pricing as pricing

        def invoice_line(item):
            return pricing.apply_discount(item.price, item.rate)
    """,
    "shop/refunds.py": """
        from shop.pricing import apply_discount as discounted

        def refund(item):
            return discounted(item.price, 0.1)
    """,
    "shop/reports.py": """
        def monthly_report(orders):
            return sum(o.total for o in orders)
    """,
}


# ---- 1. a function's signature changes --------------------------------------------------


def test_signature_change_finds_untouched_callers_however_they_import_it(tmp_path: Path) -> None:
    out = analyze(
        tmp_path,
        {"shop/pricing.py": PRICING_V1, **CALLERS},
        {"shop/pricing.py": PRICING_V2, **CALLERS},
    )
    assert {"shop/checkout.py", "shop/invoice.py", "shop/refunds.py"} <= out.files  # from-import, module attr, alias
    assert "shop/reports.py" not in out.files  # unrelated
    for caller in ("shop/checkout.py", "shop/invoice.py", "shop/refunds.py"):
        assert out.item(caller).bucket == CONTRACT
        assert "signature changed" in out.item(caller).reason


def test_the_import_line_alone_is_not_reported_as_a_call_site(tmp_path: Path) -> None:
    out = analyze(tmp_path, {"shop/pricing.py": PRICING_V1, **CALLERS}, {"shop/pricing.py": PRICING_V2, **CALLERS})
    # checkout.py's import is on line 1; the call is on line 4
    assert out.item("shop/checkout.py").line == 4


def test_a_call_site_is_reported_with_the_reason_it_matters(tmp_path: Path) -> None:
    out = analyze(tmp_path, {"shop/pricing.py": PRICING_V1, **CALLERS}, {"shop/pricing.py": PRICING_V2, **CALLERS})
    assert out.item("shop/checkout.py").reason == "calls `apply_discount`, whose signature changed in this PR"


# ---- 2. only the body changes -----------------------------------------------------------


def test_body_change_finds_callers_but_ranks_them_below_contract_changes(tmp_path: Path) -> None:
    after = PRICING_V1.replace("price * (1 - rate)", "round(price * (1 - rate), 2)")
    out = analyze(tmp_path, {"shop/pricing.py": PRICING_V1, **CALLERS}, {"shop/pricing.py": after, **CALLERS})
    assert out.item("shop/checkout.py").bucket == BEHAVIOUR
    assert "behaviour changed" in out.item("shop/checkout.py").reason


# ---- 3 + 4. removal and rename ----------------------------------------------------------


def test_removed_function_leaves_dangling_references_including_imports(tmp_path: Path) -> None:
    out = analyze(tmp_path, {"shop/pricing.py": PRICING_V1, **CALLERS}, {"shop/pricing.py": "TAX = 0.2", **CALLERS})
    assert {"shop/checkout.py", "shop/invoice.py", "shop/refunds.py"} <= out.files
    assert out.item("shop/checkout.py").bucket == DANGLING
    assert "removed or renamed" in out.item("shop/checkout.py").reason


def test_rename_flags_callers_still_using_the_old_name(tmp_path: Path) -> None:
    renamed = PRICING_V1.replace("apply_discount", "discounted_price")
    updated = {"shop/checkout.py": """
        from shop.pricing import discounted_price

        def checkout_total(cart):
            return sum(discounted_price(i.price, i.rate) for i in cart)
    """}
    out = analyze(
        tmp_path,
        {"shop/pricing.py": PRICING_V1, **CALLERS},
        {"shop/pricing.py": renamed, **CALLERS, **updated},
    )
    assert "shop/invoice.py" in out.files and "shop/refunds.py" in out.files  # forgotten
    assert "shop/checkout.py" not in out.files  # updated in this PR
    assert out.item("shop/invoice.py").bucket == DANGLING


def test_deleting_a_whole_file_flags_importers_of_its_names(tmp_path: Path) -> None:
    after = {k: v for k, v in CALLERS.items()}
    out = analyze(tmp_path, {"shop/pricing.py": PRICING_V1, **CALLERS}, after)
    assert "shop/checkout.py" in out.files
    assert out.item("shop/checkout.py").bucket == DANGLING


# ---- 5. callers updated in the same PR -----------------------------------------------------


def test_callers_the_pr_already_updated_are_not_reported_as_untouched(tmp_path: Path) -> None:
    fixed = {
        "shop/checkout.py": """
            from shop.pricing import apply_discount

            def checkout_total(cart):
                return sum(apply_discount(i.price, i.rate, 0) for i in cart)
        """,
    }
    out = analyze(
        tmp_path,
        {"shop/pricing.py": PRICING_V1, **CALLERS},
        {"shop/pricing.py": PRICING_V2, **CALLERS, **fixed},
    )
    assert "shop/checkout.py" not in out.files
    assert {"shop/invoice.py", "shop/refunds.py"} <= out.files  # still stale
    assert any("also modified in this PR" in note for note in out.notes)


def test_when_every_caller_was_updated_the_note_says_so(tmp_path: Path) -> None:
    fixed = {name: src.replace("item.rate)", "item.rate, 0)").replace("i.rate)", "i.rate, 0)").replace("0.1)", "0.1, 0)")
             for name, src in CALLERS.items()}
    out = analyze(tmp_path, {"shop/pricing.py": PRICING_V1, **CALLERS}, {"shop/pricing.py": PRICING_V2, **fixed})
    assert not out.files
    assert any("also modified in this PR" in note for note in out.notes)


# ---- 6. methods -------------------------------------------------------------------------


def test_self_calls_resolve_exactly_and_instance_calls_are_marked_as_guesses(tmp_path: Path) -> None:
    before = {
        "cart.py": """
            class Cart:
                def total(self):
                    return sum(self.prices())

                def prices(self):
                    return [1, 2]
        """,
        "views.py": """
            def show(cart):
                return cart.prices()
        """,
    }
    after = {**before, "cart.py": before["cart.py"].replace("return [1, 2]", "return [1.0, 2.0]")}
    out = analyze(tmp_path, before, after)
    assert out.item("views.py").bucket == POSSIBLE
    assert "receiver type unknown" in out.item("views.py").reason
    assert out.item("cart.py").bucket == BEHAVIOUR  # self.prices() inside the class is exact


def test_methods_named_like_builtin_methods_are_not_guessed_by_name(tmp_path: Path) -> None:
    before = {
        "store.py": """
            class Store:
                def update(self, key):
                    return key
        """,
        "noise.py": """
            def merge(a, b):
                a.update(b)
                return a
        """,
    }
    after = {**before, "store.py": before["store.py"].replace("return key", "return key.lower()")}
    out = analyze(tmp_path, before, after)
    assert "noise.py" not in out.files  # dict.update, not Store.update


def test_a_subclass_override_of_a_changed_method_is_reported(tmp_path: Path) -> None:
    before = {
        "base.py": """
            class Handler:
                def handle(self, event):
                    return event
        """,
        "special.py": """
            from base import Handler

            class SpecialHandler(Handler):
                def handle(self, event):
                    return event.upper()
        """,
    }
    after = {**before, "base.py": before["base.py"].replace("def handle(self, event)", "def handle(self, event, ctx)")}
    out = analyze(tmp_path, before, after)
    assert "overrides" in out.item("special.py").reason


# ---- 7 + 8. classes ---------------------------------------------------------------------


MODEL = """
from dataclasses import dataclass

@dataclass
class Order:
    id: int
    total: float
"""


def test_adding_a_field_flags_constructors_and_subclasses(tmp_path: Path) -> None:
    users = {
        "build.py": """
            from model import Order

            def make():
                return Order(1, 9.5)
        """,
        "special.py": """
            from model import Order

            class RushOrder(Order):
                pass
        """,
    }
    out = analyze(
        tmp_path,
        {"model.py": MODEL, **users},
        {"model.py": MODEL + "    currency: str\n", **users},
    )
    assert out.item("build.py").bucket == DATA
    assert out.item("build.py").reason == "instantiates `Order`, which gained or changed the field `currency` in this PR"
    assert "special.py" in out.files


def test_changing_class_bases_flags_subclasses_and_instantiations(tmp_path: Path) -> None:
    before = {
        "model.py": "class Order:\n    pass\n",
        "build.py": "from model import Order\n\ndef make():\n    return Order()\n",
    }
    after = {**before, "model.py": "class Order(object):\n    pass\n"}
    out = analyze(tmp_path, before, after)
    assert out.item("build.py").bucket == CONTRACT


# ---- 9 + 10. constants and enums -------------------------------------------------------------


def test_changing_a_constant_flags_every_reader(tmp_path: Path) -> None:
    before = {
        "config.py": "TIMEOUT = 5\n",
        "client.py": "from config import TIMEOUT\n\ndef wait():\n    return TIMEOUT * 2\n",
        "other.py": "import config\n\ndef limit():\n    return config.TIMEOUT\n",
    }
    after = {**before, "config.py": "TIMEOUT = 30\n"}
    out = analyze(tmp_path, before, after)
    assert {"client.py", "other.py"} <= out.files
    assert out.item("client.py").bucket == DATA


COLORS = """
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
"""


def test_removing_an_enum_member_flags_code_that_still_uses_it(tmp_path: Path) -> None:
    before = {
        "color.py": COLORS,
        "paint.py": "from color import Color\n\ndef pick():\n    return Color.BLUE\n",
        "other.py": "from color import Color\n\ndef ok():\n    return Color.RED\n",
    }
    after = {**before, "color.py": COLORS.replace("    BLUE = 3\n", "")}
    out = analyze(tmp_path, before, after)
    assert out.item("paint.py").bucket == DANGLING
    assert "other.py" not in out.files


# ---- 11 - 13. how a name can reach the caller ------------------------------------------------


def test_reexport_through_a_package_init_is_followed(tmp_path: Path) -> None:
    before = {
        "pkg/__init__.py": "from pkg.core import work\n",
        "pkg/core.py": "def work(x):\n    return x\n",
        "app.py": "from pkg import work\n\ndef run():\n    return work(1)\n",
    }
    after = {**before, "pkg/core.py": "def work(x, y):\n    return x + y\n"}
    assert "app.py" in analyze(tmp_path, before, after).files


def test_relative_imports_are_resolved(tmp_path: Path) -> None:
    before = {
        "pkg/core.py": "def work(x):\n    return x\n",
        "pkg/a.py": "from .core import work\n\ndef run():\n    return work(1)\n",
        "pkg/b.py": "from . import core\n\ndef run():\n    return core.work(1)\n",
    }
    after = {**before, "pkg/core.py": "def work(x, y):\n    return x + y\n"}
    assert {"pkg/a.py", "pkg/b.py"} <= analyze(tmp_path, before, after).files


def test_a_function_passed_as_a_value_is_still_a_dependent(tmp_path: Path) -> None:
    before = {
        "tasks.py": "def job(x):\n    return x\n",
        "runner.py": "from tasks import job\n\ndef go(items):\n    return list(map(job, items))\n",
    }
    after = {**before, "tasks.py": "def job(x, y):\n    return x + y\n"}
    assert "runner.py" in analyze(tmp_path, before, after).files


def test_star_imports_are_followed(tmp_path: Path) -> None:
    before = {
        "tools.py": "def helper(x):\n    return x\n",
        "app.py": "from tools import *\n\ndef run():\n    return helper(1)\n",
    }
    after = {**before, "tools.py": "def helper(x, y):\n    return x + y\n"}
    assert "app.py" in analyze(tmp_path, before, after).files


def test_async_callers_and_nested_scopes_are_found(tmp_path: Path) -> None:
    before = {
        "svc.py": "async def fetch(url):\n    return url\n",
        "app.py": "from svc import fetch\n\nasync def run(urls):\n    def inner():\n        return [fetch(u) for u in urls]\n    return await fetch('x')\n",
    }
    after = {**before, "svc.py": "async def fetch(url, timeout):\n    return url\n"}
    assert "app.py" in analyze(tmp_path, before, after).files


def test_decorator_changes_count_as_signature_changes(tmp_path: Path) -> None:
    before = {
        "views.py": "def route(f):\n    return f\n\n@route\ndef home():\n    return 1\n",
        "app.py": "from views import home\n\ndef run():\n    return home()\n",
    }
    after = {**before, "views.py": before["views.py"].replace("@route\n", "@route\n@cache\n")}
    out = analyze(tmp_path, before, after)
    assert out.item("app.py").bucket == CONTRACT


# ---- 14. dynamic use --------------------------------------------------------------------------


def test_dispatch_by_string_is_caught_by_the_textual_fallback(tmp_path: Path) -> None:
    before = {
        "plugins.py": "def apply_discount(x):\n    return x\n",
        "registry.py": "import plugins\n\ndef run(name, x):\n    return getattr(plugins, 'apply_discount')(x)\n",
    }
    after = {**before, "plugins.py": "def apply_discount(x, y):\n    return x + y\n"}
    out = analyze(tmp_path, before, after)
    assert out.item("registry.py").bucket == TEXTUAL
    assert "string" in out.item("registry.py").reason


# ---- 15. big fan-out ----------------------------------------------------------------------------


def test_many_callers_are_capped_with_one_site_per_file_first(tmp_path: Path) -> None:
    callers = {f"mod{i}/use.py": f"from lib import work\n\ndef a{i}():\n    return work({i})\n\ndef b{i}():\n    return work(0)\n"
               for i in range(30)}
    before = {"lib.py": "def work(x):\n    return x\n", **callers}
    after = {**before, "lib.py": "def work(x, y):\n    return x + y\n"}
    out = analyze(tmp_path, before, after)
    shown = [i for i in out.items if not i.is_test]
    assert len(shown) == MAX_IMPACT_SITES
    assert len({i.file for i in shown}) == MAX_IMPACT_SITES  # diversity: no file twice
    assert any("60 reference site(s) in 30 file(s)" in note for note in out.notes)


def test_tests_are_shown_but_capped_and_ranked_after_production_code(tmp_path: Path) -> None:
    tests = {f"tests/test_{i}.py": f"from lib import work\n\ndef test_{i}():\n    assert work({i}) == {i}\n" for i in range(6)}
    before = {"lib.py": "def work(x):\n    return x\n",
              "app.py": "from lib import work\n\ndef run():\n    return work(1)\n", **tests}
    after = {**before, "lib.py": "def work(x, y):\n    return x + y\n"}
    out = analyze(tmp_path, before, after)
    assert sum(i.is_test for i in out.items) == MAX_TEST_SITES
    assert out.items[0].file == "app.py" and out.items[0].bucket < TEST


# ---- 16 - 19. precision --------------------------------------------------------------------------


def test_two_modules_defining_the_same_name_are_not_confused(tmp_path: Path) -> None:
    before = {
        "a.py": "def process(x):\n    return x\n",
        "b.py": "def process(x):\n    return x * 2\n",
        "uses_a.py": "from a import process\n\ndef run():\n    return process(1)\n",
        "uses_b.py": "from b import process\n\ndef run():\n    return process(1)\n",
    }
    after = {**before, "a.py": "def process(x, y):\n    return x + y\n"}
    out = analyze(tmp_path, before, after)
    assert "uses_a.py" in out.files and "uses_b.py" not in out.files


def test_a_local_variable_named_like_the_function_is_not_a_caller(tmp_path: Path) -> None:
    before = {
        "lib.py": "def total(xs):\n    return sum(xs)\n",
        "other.py": "def run(total):\n    return total(3)\n",
    }
    after = {**before, "lib.py": "def total(xs, start):\n    return sum(xs) + start\n"}
    assert "other.py" not in analyze(tmp_path, before, after).files


def test_recursion_inside_the_changed_function_is_not_a_dependent(tmp_path: Path) -> None:
    before = {"walk.py": "def walk(n):\n    return walk(n - 1) if n else 0\n"}
    after = {"walk.py": "def walk(n, depth):\n    return walk(n - 1, depth) if n else 0\n"}
    assert not analyze(tmp_path, before, after).files


def test_recursion_is_ignored_even_when_only_the_signature_changed(tmp_path: Path) -> None:
    # The recursive call is on an UNCHANGED line, so only the "inside itself" rule excludes it.
    body = "    a = 1\n    b = 2\n    c = 3\n    return walk(n - 1) if n else a + b + c\n"
    before = {"walk.py": f"def walk(n):\n{body}"}
    after = {"walk.py": f"def walk(n, depth=0):\n{body}"}
    assert not analyze(tmp_path, before, after).files


def test_an_unparseable_file_does_not_break_analysis(tmp_path: Path) -> None:
    before = {"lib.py": "def work(x):\n    return x\n", "app.py": "from lib import work\n\ndef run():\n    return work(1)\n"}
    after = {**before, "lib.py": "def work(x, y):\n    return x + y\n", "broken.py": "def oops(:\n"}
    assert "app.py" in analyze(tmp_path, before, after).files


# ---- 20. nothing depends on it ---------------------------------------------------------------------


def test_no_dependents_produces_an_honest_note_not_silence(tmp_path: Path) -> None:
    before = {"lib.py": "def lonely(x):\n    return x\n"}
    after = {"lib.py": "def lonely(x, y):\n    return x + y\n"}
    out = analyze(tmp_path, before, after)
    assert not out.items
    assert any("No references were found" in n and "reflection" in n for n in out.notes)


def test_top_level_script_code_has_no_symbol_to_chase(tmp_path: Path) -> None:
    pr = make_pr(tmp_path, {"run.py": "print('a')\n"}, {"run.py": "print('b')\n"})
    symbols = SymbolIndex.build(tmp_path, discover_python_files(tmp_path))
    assert changed_symbols(symbols, pr.hunks[0]) == []


# ---- how changes are classified -----------------------------------------------------------------------


def test_signature_vs_body_is_decided_by_which_lines_changed(tmp_path: Path) -> None:
    src = "def f(a, b):\n    x = a\n    return x + b\n"
    sig = make_pr(tmp_path / "s", {"m.py": src}, {"m.py": src.replace("(a, b)", "(a, b, c)")})
    body = make_pr(tmp_path / "b", {"m.py": src}, {"m.py": src.replace("x + b", "x - b")})
    for root, pr, expected in ((tmp_path / "s", sig, "signature"), (tmp_path / "b", body, "body")):
        symbols = SymbolIndex.build(root, discover_python_files(root))
        assert [s.change for s in changed_symbols(symbols, pr.hunks[0])] == [expected]


# ---- deletions: attributing them without blaming the neighbours -------------------------------------


def _changes(tmp_path: Path, before: str, after: str, name: str = "m.py") -> list[tuple[str, str]]:
    pr = make_pr(tmp_path, {name: before}, {name: after})
    symbols = SymbolIndex.build(tmp_path, discover_python_files(tmp_path))
    return sorted((s.qualname, s.change) for h in pr.hunks for s in changed_symbols(symbols, h))


TWO_FUNCS = "def keep(x):\n    y = x + 1\n    return y\n\n\ndef drop(x):\n    return x\n"


def test_removing_a_function_does_not_blame_the_function_above_it(tmp_path: Path) -> None:
    after = "def keep(x):\n    y = x + 1\n    return y\n"
    assert _changes(tmp_path, TWO_FUNCS, after) == [("drop", "removed")]


def test_removing_a_method_does_not_blame_the_method_above_it(tmp_path: Path) -> None:
    before = "class A:\n    def keep(self):\n        return 1\n\n    def drop(self):\n        return 2\n"
    after = "class A:\n    def keep(self):\n        return 1\n"
    assert _changes(tmp_path, before, after) == [("drop", "removed")]


def test_deleting_a_statement_from_a_body_is_a_behaviour_change(tmp_path: Path) -> None:
    after = TWO_FUNCS.replace("    y = x + 1\n    return y", "    return x")
    assert _changes(tmp_path, TWO_FUNCS, after) == [("keep", "body")]


def test_removing_a_decorator_is_a_contract_change_of_the_function_below(tmp_path: Path) -> None:
    before = "def cache(f):\n    return f\n\n\n@cache\ndef slow(x):\n    return x\n"
    after = before.replace("@cache\n", "")
    assert _changes(tmp_path, before, after) == [("slow", "signature")]


def test_removing_a_parameter_line_from_a_multiline_signature_is_a_contract_change(tmp_path: Path) -> None:
    before = "def f(\n    a,\n    b,\n    c,\n):\n    return a\n"
    after = "def f(\n    a,\n    b,\n):\n    return a\n"
    assert _changes(tmp_path, before, after) == [("f", "signature")]


def test_removing_a_parameter_flags_callers_that_still_pass_it(tmp_path: Path) -> None:
    before = {"lib.py": "def f(a, b):\n    return a\n", "app.py": "from lib import f\n\ndef run():\n    return f(1, 2)\n"}
    after = {**before, "lib.py": "def f(a):\n    return a\n"}
    assert analyze(tmp_path, before, after).item("app.py").bucket == CONTRACT


def test_imports_under_type_checking_and_try_blocks_are_followed(tmp_path: Path) -> None:
    before = {
        "lib.py": "def work(x):\n    return x\n",
        "typed.py": "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from lib import work\n\ndef run(f: 'work'):\n    return f\n",
        "guarded.py": "try:\n    from lib import work\nexcept ImportError:\n    work = None\n\ndef run():\n    return work(1)\n",
    }
    after = {**before, "lib.py": "def work(x, y):\n    return x + y\n"}
    out = analyze(tmp_path, before, after)
    assert "guarded.py" in out.files  # `work = None` is a fallback, not a different `work`


def test_a_body_change_to_main_does_not_report_the_scripts_own_entry_line(tmp_path: Path) -> None:
    before = {"run.py": "def main():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n    print(a)\n\n\nif __name__ == '__main__':\n    main()\n"}
    after = {"run.py": before["run.py"].replace("print(a)", "print(a + d)")}
    assert not analyze(tmp_path, before, after).files


def test_a_module_level_caller_in_another_file_is_still_reported(tmp_path: Path) -> None:
    before = {"lib.py": "def work():\n    return 1\n", "script.py": "from lib import work\n\nwork()\n"}
    after = {**before, "lib.py": "def work():\n    return 2\n"}
    assert "script.py" in analyze(tmp_path, before, after).files
