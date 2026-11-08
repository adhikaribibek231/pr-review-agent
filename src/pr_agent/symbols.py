"""Static reference analysis: who defines what, and who uses it.

Embeddings answer "which code is *similar*?". A pull request raises a different
question: "if this function changed, which *other* code depends on it?". That is
a question about structure, so it is answered from the AST, deterministically:

  * every definition (function, method, class, module variable, class field)
    with its line span, and where its signature ends;
  * every reference (call, plain use, base class, import), resolved to the
    definition it points at by following imports: `from m import f as g`,
    `import pkg.mod as x` then `x.f()`, re-exports through `__init__.py`,
    and `self.method()` inside a class.

Resolution is by *import path*, so two modules that both define `process` do not
get confused. What a static pass cannot see is listed in docs/retrieval_scenarios.md
(dynamic dispatch, reflection, other languages); the impact module compensates
where it cheaply can and says so where it cannot.
"""

import ast
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)

# Attribute names that exist on built-in types. `d.update(...)` on a dict says
# nothing about a repo method that happens to be called `update`.
COMMON_ATTRIBUTES: frozenset[str] = frozenset(
    name for kind in (dict, list, set, str, bytes, tuple, int, float, object, type) for name in dir(kind)
)


@dataclass(frozen=True)
class Definition:
    file: str
    name: str
    qualname: str  # "func", "Class", "Class.method", "outer.inner"
    kind: str  # function | method | class | variable | field
    start: int  # first line, including decorators
    end: int
    header_end: int  # last line of the signature / class header
    owner: str | None = None  # qualname of the enclosing class, for methods and fields


@dataclass(frozen=True)
class Reference:
    file: str
    line: int
    name: str  # the simple name that was referenced
    kind: str  # call | use | base | import
    enclosing: str | None  # qualname of the innermost def/class containing it; None at module level
    target: tuple[str, str] | None = None  # (file, qualname) when resolved through imports
    attribute: bool = False  # `obj.name` where obj's type is unknown (so target is None)


@dataclass(frozen=True)
class _Binding:
    kind: str  # "module" | "symbol"
    file: str | None  # module: its file; symbol: the file it is imported from (None if external)
    name: str | None = None  # symbol: original name in that file
    dotted: str | None = None  # module: dotted path, so `a.b.c` can be extended one part at a time


# --------------------------------------------------------------------------
# module resolution
# --------------------------------------------------------------------------


class _Modules:
    """Maps dotted module names to repository files."""

    def __init__(self, files: list[str]) -> None:
        self._files = set(files)
        self._exact: dict[str, str] = {}
        self._suffix: dict[str, str | None] = {}
        for file in files:
            parts = list(PurePosixPath(file).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if not parts:
                continue
            self._exact[".".join(parts)] = file
            # `src/pkg/mod.py` is importable as `pkg.mod` when src/ is the source root,
            # so every dotted suffix is a candidate. A suffix two files share is ambiguous.
            for i in range(len(parts)):
                dotted = ".".join(parts[i:])
                if dotted in self._suffix and self._suffix[dotted] != file:
                    self._suffix[dotted] = None
                else:
                    self._suffix[dotted] = file

    def resolve(self, dotted: str) -> str | None:
        return self._exact.get(dotted) or self._suffix.get(dotted)

    def resolve_relative(self, from_file: str, level: int, module: str | None) -> str | None:
        base = PurePosixPath(from_file).parent
        for _ in range(level - 1):
            base = base.parent
        target = base / module.replace(".", "/") if module else base
        for candidate in (f"{target}.py", f"{target}/__init__.py"):
            candidate = candidate.removeprefix("./")
            if candidate in self._files:
                return candidate
        return None


# --------------------------------------------------------------------------
# pass 1: definitions
# --------------------------------------------------------------------------


def _header_end(node: ast.stmt) -> int:
    body = getattr(node, "body", None)
    if not body:
        return node.lineno
    return max(node.lineno, body[0].lineno - 1)


def _start(node: ast.stmt) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])


def _collect_definitions(file: str, tree: ast.Module) -> list[Definition]:
    found: list[Definition] = []

    def add_assignments(node: ast.stmt, prefix: str, owner: str | None) -> None:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                found.append(Definition(
                    file, target.id, f"{prefix}{target.id}", "field" if owner else "variable",
                    node.lineno, node.end_lineno or node.lineno, node.end_lineno or node.lineno, owner,
                ))

    def walk(body: list[ast.stmt], prefix: str, owner: str | None, in_function: bool) -> None:
        for node in body:
            if isinstance(node, _FUNCTIONS):
                qual = f"{prefix}{node.name}"
                found.append(Definition(
                    file, node.name, qual, "method" if owner else "function",
                    _start(node), node.end_lineno or node.lineno, _header_end(node), owner,
                ))
                walk(node.body, f"{qual}.", None, True)
            elif isinstance(node, ast.ClassDef):
                qual = f"{prefix}{node.name}"
                found.append(Definition(
                    file, node.name, qual, "class",
                    _start(node), node.end_lineno or node.lineno, _header_end(node), owner,
                ))
                walk(node.body, f"{qual}.", qual, in_function)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                if not in_function:
                    add_assignments(node, prefix, owner)
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                # `if TYPE_CHECKING:`, `try: import x except ImportError:` ...
                for block in (getattr(node, "body", []), getattr(node, "orelse", []),
                              getattr(node, "finalbody", [])):
                    walk(block, prefix, owner, in_function)
                for handler in getattr(node, "handlers", []):
                    walk(handler.body, prefix, owner, in_function)

    walk(tree.body, "", None, False)
    return found


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------


class SymbolIndex:
    """Definitions and resolved references for every Python file under a root."""

    def __init__(self) -> None:
        self.definitions: dict[str, list[Definition]] = {}
        self.lines: dict[str, list[str]] = {}
        self._by_qual: dict[tuple[str, str], Definition] = {}
        self._top: dict[str, dict[str, Definition]] = {}
        self._bindings: dict[str, dict[str, _Binding]] = {}
        self._stars: dict[str, list[str]] = {}
        self._modules = _Modules([])
        self._refs_to: dict[tuple[str, str], list[Reference]] = defaultdict(list)
        self._attribute_refs: dict[str, list[Reference]] = defaultdict(list)
        self._refs_named: dict[str, list[Reference]] = defaultdict(list)
        self._defs_named: dict[str, list[Definition]] = defaultdict(list)

    # ---- building ----------------------------------------------------------

    @classmethod
    def build(cls, root: Path, files: list[Path]) -> "SymbolIndex":
        index = cls()
        trees: dict[str, ast.Module] = {}
        for path in files:
            relative = path.relative_to(root).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
                trees[relative] = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError, OSError, ValueError) as error:
                logger.warning("Symbol analysis skipped %s: %s", relative, error)
                continue
            index.lines[relative] = source.split("\n")

        index._modules = _Modules(list(trees))
        for file, tree in trees.items():
            defs = _collect_definitions(file, tree)
            index.definitions[file] = defs
            index._top[file] = {d.qualname: d for d in defs if "." not in d.qualname}
            for definition in defs:
                index._by_qual[(file, definition.qualname)] = definition
                index._defs_named[definition.name].append(definition)
        for file, tree in trees.items():
            index._collect_bindings(file, tree)
        for file, tree in trees.items():
            _Collector(index, file, tree).run()
        return index

    def _collect_bindings(self, file: str, tree: ast.Module) -> None:
        bindings: dict[str, _Binding] = {}
        stars: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        bindings[alias.asname] = _Binding(
                            "module", self._modules.resolve(alias.name), dotted=alias.name)
                    else:
                        first = alias.name.split(".")[0]
                        bindings[first] = _Binding("module", self._modules.resolve(first), dotted=first)
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                source = (self._modules.resolve_relative(file, node.level, module) if node.level
                          else (self._modules.resolve(module) if module else None))
                for alias in node.names:
                    bound = alias.asname or alias.name
                    if alias.name == "*":
                        if source:
                            stars.append(source)
                        continue
                    submodule = (
                        self._modules.resolve_relative(
                            file, node.level, f"{module}.{alias.name}" if module else alias.name)
                        if node.level
                        else (self._modules.resolve(f"{module}.{alias.name}") if module else None)
                    )
                    if submodule:
                        dotted = None if node.level else f"{module}.{alias.name}"
                        bindings[bound] = _Binding("module", submodule, dotted=dotted)
                    else:
                        bindings[bound] = _Binding("symbol", source, name=alias.name)
        self._bindings[file] = bindings
        self._stars[file] = stars

    # ---- resolution --------------------------------------------------------

    def follow(self, file: str, name: str, depth: int = 0) -> tuple[str, str]:
        """Where does `name`, as seen from `file`, actually live? Follows re-exports.

        If nothing defines it, returns (file, name) anyway: a reference to a
        removed symbol must still point somewhere so it can be found.
        """
        if self._defines_locally(file, name):
            return (file, name)
        if depth < 5:
            binding = self._bindings.get(file, {}).get(name)
            if binding and binding.kind == "symbol" and binding.file and binding.name:
                return self.follow(binding.file, binding.name, depth + 1)
            for star in self._stars.get(file, []):
                if name in self._top.get(star, {}):
                    return self.follow(star, name, depth + 1)
        return (file, name)

    def _defines_locally(self, file: str, name: str) -> bool:
        """Does `file` define `name` itself, rather than just importing it?

        A plain module variable that shares a name with an import is the fallback in
        `try: from lib import work / except ImportError: work = None`. That is the
        imported thing, not a different one, so the import wins.
        """
        definition = self._top.get(file, {}).get(name)
        if definition is None:
            return False
        return not (definition.kind == "variable" and name in self._bindings.get(file, {}))

    def lookup_name(self, file: str, name: str) -> tuple[str, tuple[str, str] | _Binding] | None:
        if self._defines_locally(file, name):
            return ("symbol", (file, name))
        binding = self._bindings.get(file, {}).get(name)
        if binding:
            if binding.kind == "module":
                return ("module", binding)
            if binding.file and binding.name:
                return ("symbol", self.follow(binding.file, binding.name))
            return None
        for star in self._stars.get(file, []):
            if name in self._top.get(star, {}):
                return ("symbol", self.follow(star, name))
        return None

    def resolve_module_chain(
        self, binding: _Binding, parts: list[str]
    ) -> tuple[tuple[str, str], list[str]] | None:
        """`pkg.mod.func` / `mod.Class.method`: walk submodules, then land on a symbol.

        Returns the symbol and whatever attribute parts follow it.
        """
        current_file, dotted = binding.file, binding.dotted
        for position, part in enumerate(parts):
            if dotted:
                candidate = f"{dotted}.{part}"
                submodule = self._modules.resolve(candidate)
                if submodule:
                    current_file, dotted = submodule, candidate
                    continue
            if current_file is None:
                return None  # a third-party module such as os or numpy
            return self.follow(current_file, part), parts[position + 1:]
        return None

    # ---- queries -----------------------------------------------------------

    def definition(self, file: str, qualname: str) -> Definition | None:
        return self._by_qual.get((file, qualname))

    def defined_names(self, file: str) -> set[str]:
        return {d.name for d in self.definitions.get(file, [])}

    def definitions_named(self, name: str) -> list[Definition]:
        return self._defs_named.get(name, [])

    def containing(self, file: str, line: int) -> list[Definition]:
        """Definitions whose span holds `line`, outermost first."""
        inside = [d for d in self.definitions.get(file, []) if d.start <= line <= d.end]
        return sorted(inside, key=lambda d: (d.start, -d.end))

    def refs_to(self, target: tuple[str, str]) -> list[Reference]:
        return self._refs_to.get(target, [])

    def attribute_refs(self, name: str) -> list[Reference]:
        return self._attribute_refs.get(name, [])

    def refs_named(self, name: str) -> list[Reference]:
        return self._refs_named.get(name, [])

    def subclasses(self, target: tuple[str, str]) -> list[tuple[str, str]]:
        """(file, qualname) of every class that lists `target` as a base."""
        return [(r.file, r.enclosing) for r in self.refs_to(target) if r.kind == "base" and r.enclosing]

    def record(self, reference: Reference) -> None:
        if reference.target:
            self._refs_to[reference.target].append(reference)
        if reference.attribute:
            self._attribute_refs[reference.name].append(reference)
        self._refs_named[reference.name].append(reference)


# --------------------------------------------------------------------------
# pass 2: references
# --------------------------------------------------------------------------


class _Collector(ast.NodeVisitor):
    def __init__(self, index: SymbolIndex, file: str, tree: ast.Module) -> None:
        self.index = index
        self.file = file
        self.tree = tree
        self.quals: list[str] = []  # qualnames of enclosing defs/classes, outermost first
        self.classes: list[str] = []  # qualnames of enclosing classes
        self.locals: list[set[str]] = []  # local names of each enclosing function

    def run(self) -> None:
        self.visit(self.tree)

    # ---- scope helpers -----------------------------------------------------

    @property
    def enclosing(self) -> str | None:
        return self.quals[-1] if self.quals else None

    def is_local(self, name: str) -> bool:
        return any(name in names for names in self.locals)

    def _qual(self, name: str) -> str:
        return f"{self.quals[-1]}.{name}" if self.quals else name

    @staticmethod
    def _locals_of(node: ast.AST) -> set[str]:
        names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.arg):
                names.add(child.arg)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                names.add(child.id)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child is not node:
                names.add(child.name)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
        return names

    # ---- recording ---------------------------------------------------------

    def _emit(self, node: ast.AST, name: str, kind: str, target: tuple[str, str] | None,
              attribute: bool = False, enclosing: str | None = None) -> None:
        self.index.record(Reference(
            file=self.file, line=getattr(node, "lineno", 0), name=name, kind=kind,
            enclosing=enclosing if enclosing is not None else self.enclosing,
            target=target, attribute=attribute,
        ))

    def _reference(self, expr: ast.expr, kind: str, enclosing: str | None = None) -> None:
        if isinstance(expr, ast.Name):
            self._name(expr, kind, enclosing)
        elif isinstance(expr, ast.Attribute):
            self._attribute(expr, kind, enclosing)
        else:
            self.visit(expr)

    def _name(self, node: ast.Name, kind: str, enclosing: str | None = None) -> None:
        if self.is_local(node.id):
            return
        found = self.index.lookup_name(self.file, node.id)
        if found and found[0] == "symbol":
            target = found[1]
            assert isinstance(target, tuple)
            self._emit(node, target[1].split(".")[-1], kind, target, enclosing=enclosing)

    def _attribute(self, node: ast.Attribute, kind: str, enclosing: str | None = None) -> None:
        parts: list[str] = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        parts.reverse()

        if not isinstance(current, ast.Name):
            self.visit(current)  # e.g. `f().x` or `items[0].x`: the base may hold references
            self._emit(node, parts[-1], kind, None, attribute=True, enclosing=enclosing)
            return

        base = current.id
        if base in ("self", "cls") and self.classes:
            qual = f"{self.classes[-1]}.{parts[0]}"
            if (self.file, qual) in self.index._by_qual:
                self._emit(node, parts[0], kind, (self.file, qual), enclosing=enclosing)
            else:  # inherited or dynamic: the defining class is unknown
                self._emit(node, parts[-1], kind, None, attribute=True, enclosing=enclosing)
            return

        if self.is_local(base):  # `item.price`, `order.total()`: a value of unknown type
            self._emit(node, parts[-1], kind, None, attribute=True, enclosing=enclosing)
            return

        found = self.index.lookup_name(self.file, base)
        if found is None:
            return  # builtin or an unresolvable global
        if found[0] == "module":
            binding = found[1]
            assert isinstance(binding, _Binding)
            resolved = self.index.resolve_module_chain(binding, parts)
            if resolved is None:
                return
            target, rest = resolved
        else:
            symbol = found[1]
            assert isinstance(symbol, tuple)
            target, rest = symbol, parts

        definition = self.index.definition(*target)
        if rest and definition is not None and definition.kind == "class":
            # `Color.RED` / `Finding.model_validate(...)`: a use of the class AND of its
            # member. The member reference points at (file, "Color.RED") even if RED no
            # longer exists, so a reference to a *removed* member can still be found.
            self._emit(node, target[1].split(".")[-1], "use", target, enclosing=enclosing)
            self._emit(node, rest[0], kind, (target[0], f"{target[1]}.{rest[0]}"), enclosing=enclosing)
        else:
            self._emit(node, target[1].split(".")[-1], kind, target, enclosing=enclosing)

    # ---- visitors ----------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._name(node, "use")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            self._attribute(node, "use")
        else:
            self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        self._reference(node.func, "call")
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Import(self, node: ast.Import) -> None:
        return

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        source = None
        if node.level:
            source = self.index._modules.resolve_relative(self.file, node.level, node.module)
        elif node.module:
            source = self.index._modules.resolve(node.module)
        for alias in node.names:
            if alias.name == "*":
                continue
            target = self.index.follow(source, alias.name) if source else None
            self._emit(node, alias.name, "import", target)

    def _visit_annotation(self, annotation: ast.expr | None) -> None:
        if annotation is None:
            return
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:  # forward reference: "Finding"
                parsed = ast.parse(annotation.value, mode="eval").body
            except SyntaxError:
                return
            ast.copy_location(parsed, annotation)
            for child in ast.walk(parsed):
                ast.copy_location(child, annotation)
            self.visit(parsed)
        else:
            self.visit(annotation)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        arguments = node.args
        for default in [*arguments.defaults, *(d for d in arguments.kw_defaults if d is not None)]:
            self.visit(default)
        for argument in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                         *(a for a in (arguments.vararg, arguments.kwarg) if a)]:
            self._visit_annotation(argument.annotation)
        self._visit_annotation(node.returns)

        qual = self._qual(node.name)
        self.quals.append(qual)
        self.locals.append(self._locals_of(node))
        for statement in node.body:
            self.visit(statement)
        self.locals.pop()
        self.quals.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        qual = self._qual(node.name)
        for base in node.bases:
            self._reference(base, "base", enclosing=qual)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self.quals.append(qual)
        self.classes.append(qual)
        for statement in node.body:
            self.visit(statement)
        self.classes.pop()
        self.quals.pop()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._visit_annotation(node.annotation)
        if node.value is not None:
            self.visit(node.value)
        if not isinstance(node.target, ast.Name):
            self.visit(node.target)
