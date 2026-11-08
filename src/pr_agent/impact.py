"""Change impact: which code outside the diff could this change break?

A reviewer looking at one hunk of `def apply_discount(price, rate, floor)` cannot
see that `checkout.py`, which this PR never touched, still calls it with two
arguments. This module finds those places.

For each hunk it works out:

  1. WHICH SYMBOLS CHANGED and HOW. A line inside a function's signature is a
     contract change; a line in its body is a behaviour change; a field added to a
     class changes every constructor call; a deleted `def` leaves dangling
     references. Different changes endanger different code.
  2. WHO DEPENDS ON THEM: callers, instantiations, subclasses, importers,
     overrides, readers of a constant.
  3. WHICH DEPENDENTS ARE UNTOUCHED. A caller this PR also edited is probably
     already updated. One it did not edit is where the bug is. They are separated.
  4. WHAT TO SHOW, within a budget: worst-first, one site per file before a second,
     tests capped, and a note saying exactly how many exist and how many are shown.

Static analysis cannot see everything (reflection, dispatch through strings,
other languages). The notes say what was and was not searched so the reviewer
does not mistake "none found" for "none exist".
"""

import re
from dataclasses import dataclass

from pr_agent.diff_parser import DiffHunk, changed_new_lines, deletions
from pr_agent.symbols import COMMON_ATTRIBUTES, Definition, Reference, SymbolIndex

MAX_IMPACT_SITES = 6  # non-test dependents shown per hunk
MAX_TEST_SITES = 2
MAX_TEXTUAL_SITES = 1
MAX_POSSIBLE_PER_SYMBOL = 4  # unresolved `obj.method()` candidates; they are guesses
_MIN_TEXTUAL_NAME = 5  # a shorter name in a string is too likely a coincidence

# Lower bucket = shown first.
DANGLING, CONTRACT, DATA, BEHAVIOUR, TEST, POSSIBLE, TEXTUAL = range(7)

_PRECEDENCE = {"signature": 0, "header": 1, "removed": 2, "field": 3, "value": 4, "body": 5}


@dataclass(frozen=True)
class ChangedSymbol:
    file: str
    name: str
    qualname: str
    kind: str  # function | method | class | variable | field | removed
    change: str  # signature | body | header | field | value | removed
    owner: str | None = None  # the class, for methods and fields


@dataclass(frozen=True)
class ImpactItem:
    file: str
    line: int  # a line inside the dependent code, used to find its chunk
    bucket: int
    reason: str
    is_test: bool


@dataclass(frozen=True)
class ImpactReport:
    items: tuple[ImpactItem, ...]
    notes: tuple[str, ...]


# --------------------------------------------------------------------------
# 1. what changed
# --------------------------------------------------------------------------

_DELETED_DEFINITION = re.compile(r"^(\s*)(?:async\s+def|def|class)\s+(\w+)")
_DELETED_ASSIGNMENT = re.compile(r"^(\s*)(\w+)\s*(?::\s*[^=]+?)?\s*(?:=(?!=)|$)")
_CLASS_LEVEL_INDENT = ("    ", "\t")


def _deleted_lines(hunk: DiffHunk) -> list[str]:
    return [line[1:] for line in hunk.patch.splitlines()[1:] if line.startswith("-")]


def _removed_names(symbols: SymbolIndex, hunk: DiffHunk) -> set[str]:
    """Names whose definition a deleted line removed and the new file no longer has.

    Covers deleted `def`/`class`, deleted module variables, and deleted class-level
    fields/enum members. A name still defined in the new file is a modification,
    not a removal; a deleted *file* has no definitions left, so all its names go.
    """
    heading = hunk.patch.splitlines()[0].split("@@")[-1].strip()
    in_class = heading.startswith("class ")
    names: set[str] = set()
    for line in _deleted_lines(hunk):
        match = _DELETED_DEFINITION.match(line)
        if match:
            names.add(match.group(2))
            continue
        match = _DELETED_ASSIGNMENT.match(line)
        if not match or not match.group(2):
            continue
        indent, name = match.groups()
        annotated = ":" in line.split("=")[0]
        if indent == "" or (indent in _CLASS_LEVEL_INDENT and (annotated or name.isupper() or in_class)):
            names.add(name)
    names -= symbols.defined_names(hunk.filename)
    return {name for name in names if not name.startswith("__")}


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip())


def changed_symbols(symbols: SymbolIndex, hunk: DiffHunk) -> list[ChangedSymbol]:
    """The symbols this hunk changes, each classified by *how* it changed."""
    found: dict[str, ChangedSymbol] = {}

    def keep(symbol: ChangedSymbol) -> None:
        known = found.get(symbol.qualname)
        if known is None or _PRECEDENCE[symbol.change] < _PRECEDENCE[known.change]:
            found[symbol.qualname] = symbol

    def of_function(function: Definition, change: str) -> ChangedSymbol:
        return ChangedSymbol(function.file, function.name, function.qualname, function.kind,
                             change, function.owner)

    # Added lines have real positions in the new file, so they classify precisely.
    for line in sorted(hunk.added_lines):
        chain = symbols.containing(hunk.filename, line)
        function = next((d for d in reversed(chain) if d.kind in ("function", "method")), None)
        if function is not None:
            keep(of_function(function, "signature" if line <= function.header_end else "body"))
            continue
        owner = next((d for d in reversed(chain) if d.kind == "class"), None)
        if owner is not None:
            field = next((d for d in reversed(chain) if d.kind == "field"), None)
            if field is not None:
                keep(ChangedSymbol(field.file, field.name, field.qualname, "field", "field", owner.qualname))
            elif line <= owner.header_end:
                keep(ChangedSymbol(owner.file, owner.name, owner.qualname, "class", "header", owner.owner))
            continue
        variable = next((d for d in reversed(chain) if d.kind == "variable"), None)
        if variable is not None:
            keep(ChangedSymbol(variable.file, variable.name, variable.qualname, "variable", "value"))

    # Deleted lines have no new-file position, so they are attributed carefully. Only
    # function changes are inferred from them: a deleted def/class/field is the job of
    # `_removed_names`, and blaming the neighbouring symbol would be a false alarm.
    source = symbols.lines.get(hunk.filename, [])
    blocks: dict[int, list[str]] = {}  # deleted lines sharing a position are one contiguous block
    for before, text in deletions(hunk):
        blocks.setdefault(before, []).append(text)

    for before, texts in blocks.items():
        chain = symbols.containing(hunk.filename, before)
        function = next((d for d in reversed(chain) if d.kind in ("function", "method")), None)
        def_indent = _indent(source[function.start - 1]) if function and function.start - 1 < len(source) else 0
        # A block that contains a def/class at or above this function's level is a removed
        # sibling. Its inner lines are indented deeper than the neighbour's def, but they are
        # not part of the neighbour.
        if any((m := _DELETED_DEFINITION.match(t)) and _indent(t) <= def_indent for t in texts):
            continue
        for text in texts:
            stripped = text.strip()
            if not stripped:
                continue
            if stripped.startswith("@"):  # a removed decorator changes the def right below it
                below = next((d for d in symbols.containing(hunk.filename, before + 1)
                              if d.start == before + 1 and d.kind in ("function", "method")), None)
                if below is not None:
                    keep(of_function(below, "signature"))
                continue
            if function is None or _indent(text) <= def_indent:
                continue  # shallower than the def: not part of this function's body
            # Above the closing line of the header it is a parameter; at or below it, the body.
            keep(of_function(function, "signature" if function.start <= before < function.header_end else "body"))

    for name in sorted(_removed_names(symbols, hunk)):
        keep(ChangedSymbol(hunk.filename, name, name, "removed", "removed"))
    return list(found.values())


# --------------------------------------------------------------------------
# 2. who depends on them
# --------------------------------------------------------------------------


def is_test_file(path: str) -> bool:
    parts = path.lower().split("/")
    name = parts[-1]
    return (
        name.startswith("test_") or name.endswith(("_test.py", "_tests.py")) or name == "conftest.py"
        or any(part in ("test", "tests", "testing") for part in parts[:-1])
    )


def _distinctive(symbols: SymbolIndex, name: str) -> bool:
    """Is `obj.name` specific enough that matching it by name is not just noise?"""
    return (
        not name.startswith("__")
        and name not in COMMON_ATTRIBUTES
        and len(symbols.definitions_named(name)) <= 3
    )


@dataclass(frozen=True)
class _Dependent:
    file: str
    line: int
    enclosing: str | None
    kind: str  # call | use | base | import | override
    exact: bool
    via_owner: bool = False  # for a field change: the reference is to the owning class


def _dependents(symbols: SymbolIndex, symbol: ChangedSymbol) -> list[_Dependent]:
    def from_refs(refs: list[Reference], kinds: tuple[str, ...], exact: bool = True,
                  via_owner: bool = False) -> list[_Dependent]:
        return [_Dependent(r.file, r.line, r.enclosing, r.kind, exact, via_owner)
                for r in refs if r.kind in kinds]

    here = (symbol.file, symbol.qualname)
    # Imports are deliberately absent: `from m import f` says little about a changed `f`;
    # its uses say more. (They are only reported for a *removed* name: a dangling import.)
    everything = ("call", "use", "base")
    found: list[_Dependent] = []

    if symbol.change == "removed":
        # A removed name is dangling wherever it is still imported or used, including a
        # reference whose module no longer resolves because the whole file was deleted.
        for ref in symbols.refs_named(symbol.name):
            if ref.target is not None and ref.target[0] == symbol.file and ref.target[1].split(".")[-1] == symbol.name:
                found.append(_Dependent(ref.file, ref.line, ref.enclosing, ref.kind, True))
            elif ref.target is None and ref.kind == "import":
                found.append(_Dependent(ref.file, ref.line, ref.enclosing, ref.kind, True))
            elif ref.attribute and _distinctive(symbols, symbol.name):
                found.append(_Dependent(ref.file, ref.line, ref.enclosing, ref.kind, False))
        return found

    if symbol.kind == "field" and symbol.owner:
        owner = (symbol.file, symbol.owner)
        found += from_refs(symbols.refs_to(owner), ("call", "base"), via_owner=True)  # constructors, subclasses
        found += from_refs(symbols.refs_to(here), everything)  # `Color.RED`
        if _distinctive(symbols, symbol.name):  # `order.total`: type unknown, so a guess
            found += from_refs(symbols.attribute_refs(symbol.name), everything, exact=False)
        return found

    found += from_refs(symbols.refs_to(here), everything)
    if symbol.kind == "method":
        if _distinctive(symbols, symbol.name):
            found += from_refs(symbols.attribute_refs(symbol.name), everything, exact=False)
        if symbol.owner:  # a subclass that overrides this method must stay compatible with it
            for sub_file, sub_qual in symbols.subclasses((symbol.file, symbol.owner)):
                override = symbols.definition(sub_file, f"{sub_qual}.{symbol.name}")
                if override is not None:
                    found.append(_Dependent(sub_file, override.start, override.qualname, "override", True))
    return found


# --------------------------------------------------------------------------
# 3 + 4. which are untouched, and what to show
# --------------------------------------------------------------------------

_CHANGE_PHRASE = {
    "signature": "whose signature changed in this PR",
    "body": "whose behaviour changed in this PR",
    "header": "whose definition (bases or decorators) changed in this PR",
    "field": "which gained or changed a field in this PR",
    "value": "whose value changed in this PR",
    "removed": "which this PR removed or renamed",
}
_KIND_VERB = {"call": "calls", "use": "references", "base": "subclasses", "import": "imports",
              "override": "overrides"}
_NOTE_CHANGE = {
    "signature": "signature changed", "body": "behaviour changed", "header": "definition changed",
    "field": "field changed", "value": "value changed", "removed": "removed or renamed",
}


def _reason(symbol: ChangedSymbol, dependent: _Dependent) -> str:
    """Why this dependent is shown, worded about the right subject."""
    guess = " (receiver type unknown: a possible caller)" if not dependent.exact else ""
    if symbol.kind == "field" and dependent.via_owner and symbol.owner:
        owner = symbol.owner.split(".")[-1]
        verb = {"call": "instantiates", "base": "subclasses"}.get(dependent.kind, "references")
        return f"{verb} `{owner}`, which gained or changed the field `{symbol.name}` in this PR"
    return f"{_KIND_VERB[dependent.kind]} `{symbol.name}`, {_CHANGE_PHRASE[symbol.change]}{guess}"


def _bucket(symbol: ChangedSymbol, dependent: _Dependent, is_test: bool) -> int:
    if is_test:
        return TEST
    if not dependent.exact:
        return POSSIBLE
    if symbol.change == "removed":
        return DANGLING
    if symbol.change in ("signature", "header"):
        return CONTRACT
    if symbol.change in ("field", "value"):
        return DATA
    return BEHAVIOUR


def _proximity(a: str, b: str) -> int:
    """0 = same directory, 1 = same top-level directory, 2 = elsewhere."""
    pa, pb = a.split("/")[:-1], b.split("/")[:-1]
    if pa == pb:
        return 0
    return 1 if pa[:1] == pb[:1] else 2


def _pr_changed_lines(pr_hunks: list[DiffHunk]) -> dict[str, frozenset[int]]:
    lines: dict[str, set[int]] = {}
    for hunk in pr_hunks:
        lines.setdefault(hunk.filename, set()).update(changed_new_lines(hunk))
    return {file: frozenset(numbers) for file, numbers in lines.items()}


def _inside_symbol(symbols: SymbolIndex, symbol: ChangedSymbol, dependent: _Dependent) -> bool:
    """A recursive call, or a use inside the changed function itself, is not a dependent."""
    if dependent.file != symbol.file or symbol.change == "removed":
        return False
    definition: Definition | None = symbols.definition(symbol.file, symbol.qualname)
    return definition is not None and definition.start <= dependent.line <= definition.end


def _select(items: list[ImpactItem]) -> list[ImpactItem]:
    """Worst first; one site per file before any second; tests and guesses capped."""
    ordered = sorted(items, key=lambda i: (i.bucket, i.file, i.line))
    chosen: list[ImpactItem] = []
    quotas = {"main": MAX_IMPACT_SITES, "test": MAX_TEST_SITES, "textual": MAX_TEXTUAL_SITES}

    def group(item: ImpactItem) -> str:
        return "test" if item.is_test else "textual" if item.bucket == TEXTUAL else "main"

    for allow_repeat_file in (False, True):
        seen_files = {item.file for item in chosen}
        for item in ordered:
            if item in chosen or quotas[group(item)] <= 0:
                continue
            if not allow_repeat_file and item.file in seen_files:
                continue
            chosen.append(item)
            seen_files.add(item.file)
            quotas[group(item)] -= 1
    return sorted(chosen, key=lambda i: (i.bucket, i.file, i.line))


def analyze_impact(symbols: SymbolIndex, hunk: DiffHunk, pr_hunks: list[DiffHunk]) -> ImpactReport:
    """Everything outside this hunk that its change could break, ranked and explained."""
    pr_lines = _pr_changed_lines(pr_hunks or [hunk])
    items: list[ImpactItem] = []
    notes: list[str] = []

    for symbol in changed_symbols(symbols, hunk):
        untouched: dict[tuple[str, str | None], ImpactItem] = {}
        touched_sites: set[tuple[str, str | None]] = set()
        possible: dict[tuple[str, str | None], ImpactItem] = {}

        for dependent in _dependents(symbols, symbol):
            if _inside_symbol(symbols, symbol, dependent):
                continue
            site = (dependent.file, dependent.enclosing)
            if dependent.line in pr_lines.get(dependent.file, frozenset()):
                touched_sites.add(site)  # this PR already edited it: probably updated
                continue
            is_test = is_test_file(dependent.file)
            bucket = _bucket(symbol, dependent, is_test)
            item = ImpactItem(dependent.file, dependent.line, bucket, _reason(symbol, dependent), is_test)
            target = untouched if dependent.exact else possible
            known = target.get(site)
            if known is None or dependent.line < known.line:
                target[site] = item

        exact_items = list(untouched.values())
        guess_items = sorted(possible.values(), key=lambda i: (_proximity(symbol.file, i.file), i.file))
        guess_items = [g for g in guess_items if (g.file, None) not in {(x.file, None) for x in exact_items}]
        items += exact_items + guess_items[:MAX_POSSIBLE_PER_SYMBOL]

        if symbol.name.isidentifier() and len(symbol.name) >= _MIN_TEXTUAL_NAME and symbol.name not in COMMON_ATTRIBUTES:
            seen = {item.file for item in exact_items} | {symbol.file}
            pattern = re.compile(rf"""["']{re.escape(symbol.name)}["']""")
            for file, lines in sorted(symbols.lines.items()):
                if file in seen or file in pr_lines:
                    continue
                for number, text in enumerate(lines, start=1):
                    if pattern.search(text):
                        items.append(ImpactItem(
                            file, number, TEXTUAL,
                            f"mentions `{symbol.name}` as a string (possible dynamic use such as getattr or a registry)",
                            is_test_file(file)))
                        break

        notes.append(_note(symbol, exact_items, touched_sites, len(guess_items)))

    return ImpactReport(items=tuple(_select(items)), notes=tuple(notes))


def _note(symbol: ChangedSymbol, untouched: list[ImpactItem], touched: set[tuple[str, str | None]],
          guesses: int) -> str:
    qualified = f"`{symbol.qualname}` ({symbol.file}): {_NOTE_CHANGE[symbol.change]}."
    total = len(untouched) + len(touched)
    if total == 0 and guesses == 0:
        return (f"{qualified} No references were found elsewhere in the repository. "
                "Static analysis cannot see reflection, string dispatch, or non-Python code.")
    parts = [qualified]
    if untouched:
        files = sorted({item.file for item in untouched})
        shown = min(len(untouched), MAX_IMPACT_SITES + MAX_TEST_SITES)
        parts.append(f"{len(untouched)} reference site(s) in {len(files)} file(s) are in code this PR does NOT "
                     f"modify ({', '.join(files[:6])}{', ...' if len(files) > 6 else ''}); showing up to {shown}.")
    if touched:
        parts.append(f"{len(touched)} other site(s) were also modified in this PR "
                     "and are likely already updated.")
    if guesses:
        parts.append(f"{guesses} more are guesses by method name (receiver type unknown).")
    return " ".join(parts)
