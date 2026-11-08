# Retrieval scenarios: how a change ripples, and what the agent can see

A pull request changes code in one place. Whether it is *correct* often depends on code
the PR never touched. Retrieval exists to put that code in front of the reviewer.

Two different tools answer two different questions:

| Question | Tool | Why |
| --- | --- | --- |
| "What else **depends on** what changed?" (callers, subclasses, importers, readers) | **Static analysis** (`symbols.py`, `impact.py`) | It is a question about structure. Deterministic, explainable, and independent of repository size. |
| "What else **looks like** this?" (siblings, duplicates, the convention being broken) | **Embeddings** (`search.py`) | It is a question about meaning. Fuzzy, but it finds what no import connects. |

Measured on this repository (538 chunks, `experiments/retrieval_eval.py`): when a function's
signature changes, cross-file callers reached the reviewer **0 of 5** times with similarity
search alone and **5 of 5** with impact analysis. Similarity works on a 7-file toy repo because
everything is near everything; it does not scale.

Status key: **covered** = a test builds a real before/after repository and asserts it ·
**partial** = found as a *guess* or only the existence is flagged · **not covered**.

## A. What changed in a symbol

| # | Scenario | Status | Test (`tests/test_impact.py`) | What the reviewer is shown |
| - | --- | --- | --- | --- |
| 1 | Function signature changed (param added/removed/renamed, default, annotation) | covered | `test_signature_change_finds_untouched_callers...`, `test_removing_a_parameter_flags_callers...` | untouched callers, "whose signature changed" |
| 2 | Decorator added or removed | covered | `test_decorator_changes_count...`, `test_removing_a_decorator...` | callers of the decorated function |
| 3 | Parameter line deleted from a multi-line signature | covered | `test_removing_a_parameter_line...` | classified as a signature change |
| 4 | Function body changed (return value, exceptions, side effects) | covered, ranked lower | `test_body_change_finds_callers...` | callers, "whose behaviour changed" |
| 5 | Function removed | covered | `test_removed_function_leaves_dangling_references...` | every remaining import and call, ranked first |
| 6 | Function renamed | covered | `test_rename_flags_callers_still_using_the_old_name` | callers of the old name; those the PR updated are excluded |
| 7 | Whole file deleted | covered | `test_deleting_a_whole_file_flags_importers...` | importers of its names |
| 8 | Method changed | covered | `test_self_calls_resolve_exactly...` | `self.m()` exactly; `obj.m()` as flagged guesses |
| 9 | Method changed, subclass overrides it | covered | `test_a_subclass_override_of_a_changed_method...` | the overriding method |
| 10 | Class field added or changed (dataclass, Pydantic, ORM) | covered | `test_adding_a_field_flags_constructors...` | constructor sites and subclasses |
| 11 | Class bases or decorators changed | covered | `test_changing_class_bases_flags...` | subclasses and instantiations |
| 12 | Module constant or config value changed | covered | `test_changing_a_constant_flags_every_reader` | readers via import and `module.NAME` |
| 13 | Enum member removed | covered | `test_removing_an_enum_member...` | uses of that member only, not of its siblings |
| 14 | Removed a sibling definition (must not blame its neighbour) | covered | `test_removing_a_function_does_not_blame...`, `..._method_...` | nothing spurious |
| 15 | Statement deleted from a body | covered | `test_deleting_a_statement_from_a_body...` | classified as a behaviour change |

## B. How a name reaches its users

| # | Scenario | Status | Test |
| - | --- | --- | --- |
| 16 | `from m import f` | covered | scenario 1 |
| 17 | `import pkg.mod as x` then `x.f()` | covered | scenario 1 |
| 18 | `from m import f as g` | covered | scenario 1 |
| 19 | Re-export through `__init__.py` | covered | `test_reexport_through_a_package_init...` |
| 20 | Relative imports (`from .m import f`, `from . import m`) | covered | `test_relative_imports_are_resolved` |
| 21 | `from m import *` | covered | `test_star_imports_are_followed` |
| 22 | Function passed as a value (`map(f, xs)`, `target=f`) | covered | `test_a_function_passed_as_a_value...` |
| 23 | `async`/`await` callers, calls inside nested functions and comprehensions | covered | `test_async_callers_and_nested_scopes...` |
| 24 | Imports inside `if TYPE_CHECKING:` / `try:` | covered | definitions and imports are collected through these blocks |
| 25 | String-based dispatch: `getattr(mod, "f")`, registries, config names | partial | `test_dispatch_by_string...` finds quoted occurrences of a name; it cannot prove a dispatch |

## C. Precision: not reporting what is not affected

| # | Scenario | Status | Test |
| - | --- | --- | --- |
| 26 | Two modules define the same name; only one is imported | covered | `test_two_modules_defining_the_same_name...` |
| 27 | A local variable or parameter named like the function | covered | `test_a_local_variable_named_like...` |
| 28 | Recursion inside the changed function | covered | `test_recursion_is_ignored...` |
| 29 | `d.update(...)` on a dict vs. a changed `Store.update` | covered | `test_methods_named_like_builtin_methods...` |
| 30 | The import line alone (not a call site) | covered | `test_the_import_line_alone...` |
| 31 | A syntactically broken file elsewhere in the repo | covered | `test_an_unparseable_file...` |

## D. Making it usable at scale

| # | Scenario | Status | How it is handled |
| - | --- | --- | --- |
| 32 | Callers the PR **already updated** vs. ones it left **untouched** | covered | Untouched ones are shown; updated ones are excluded and counted in the note ("N other sites were also modified"). The dangerous caller is the untouched one. |
| 33 | 200 callers of a utility function | covered | Capped (6 production + 2 test sites), one site per file before a second, and the note says "60 reference site(s) in 30 file(s); showing up to 8". |
| 34 | Tests that exercise the changed code | covered | Capped at 2, ranked after production code. |
| 35 | Nothing depends on it | covered | An explicit note: "No references were found... static analysis cannot see reflection, string dispatch, or non-Python code." Silence would read as "safe". |
| 36 | The reviewer must know *why* each chunk is there | covered | Each chunk carries a `reason`; the prompt explains how to use them and says to report a broken caller **on the changed line**, naming the affected file, because findings are only accepted on added lines. |

## E. Known limits (not covered)

| Scenario | Why not | Idea if it matters |
| --- | --- | --- |
| Transitive impact (a caller of a caller; a return type change rippling two levels) | Depth 1 only. Going deeper multiplies noise. | Add depth 2 for return-type and exception changes, behind a budget, and measure. |
| Instance-method calls on typed values (`order.total()` where `order: Order`) | Needs type inference. Today these are *guesses by name*, ranked low and labelled. | Use annotations on parameters/variables to resolve receivers; or a type checker's output. |
| Field *readers* (`order.currency`) | Same: receiver type unknown, so guesses only. | As above. |
| pytest fixtures (resolved by parameter *name*) | Not an import or a call. | Treat a fixture definition's name as referenced by same-named test parameters. |
| Framework wiring: Flask/FastAPI routes, Django signals, Celery tasks, plugin entry points, decorators that register | Registered by side effect. | Per-framework rules; textual fallback catches some. |
| Enum member *added* (non-exhaustive `match`/`if` chains) | Needs exhaustiveness reasoning. | Flag `match`/`if` chains over the enum; let the LLM judge. |
| What a body change actually *did* (new `None` return, new exception) | Static analysis finds who calls it, not which assumption broke. | That is the reviewer model's job, given the callers. |
| Symmetric pairs and copy-paste siblings (`encode`/`decode`, schema vs. serializer) | No structural link. | Semantic search provides candidates; their quality is not separately measured yet. |
| Non-Python references: YAML/JSON config, SQL, protobuf, templates, docs | Python AST only. | Extend the textual fallback to non-Python files. |
| Other languages | Python only, by scope. | Tree-sitter. |
| Two files in different packages with the same dotted suffix | Resolution treats an ambiguous suffix as unresolvable rather than guessing. | Configure source roots. |
| Index freshness and size | The index is built from the PR's head checkout and re-built each run; every file is parsed. | Incremental indexing, cached by commit. |

## Adding a scenario

Write a test in `tests/test_impact.py`: a `before` dict and an `after` dict of files, then
`analyze(tmp_path, before, after)`. `tests/helpers.py::make_pr` writes the `after` repository
and computes real diffs with `difflib`, so the hunks have real line numbers. Assert what must
be found *and* what must not. Then add a row above.
