"""Independent raw-AST completeness oracle for configuration/path routes.

This scanner deliberately does not import the audit visitor, inventory, or
generated ledger. It follows only annotations, strict mapping constructors,
``Path`` constructors, explicit resolvers, and dataflow from those verified
values. The audit compares these source occurrence identities with its frozen
inventory so the generated ledger cannot prove its own completeness.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class OracleCategory(StrEnum):
    """Migration categories independently observable from raw source."""

    CONFIGURATION_REFERENCE = "configuration-reference"
    PYTHON_RUNTIME_DEFAULT = "python-runtime-default"
    CONFIGURATION_FALLBACK = "configuration-fallback"
    PATH_RESOLUTION = "path-resolution"


@dataclass(frozen=True, slots=True, order=True)
class SourceOccurrence:
    """One raw-source operation with a ledger-independent stable identity."""

    module: str
    qualified_name: str
    line: int
    column: int
    category: OracleCategory

    @property
    def occurrence_id(self) -> str:
        """Return a stable identity independent of route rendering."""
        payload = json.dumps(
            [
                self.module,
                self.qualified_name,
                self.line,
                self.column,
                self.category.value,
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]


def occurrence_id(
    module: str,
    qualified_name: str,
    line: int,
    column: int,
    category: str,
) -> str:
    """Build the oracle identity for an inventory record."""
    return SourceOccurrence(
        module,
        qualified_name,
        line,
        column,
        OracleCategory(category),
    ).occurrence_id


def _annotation(annotation: ast.expr | None) -> str:
    return "" if annotation is None else ast.unparse(annotation).lower()


def _is_mapping_annotation(annotation: ast.expr | None) -> bool:
    rendered = _annotation(annotation)
    return ("config" in rendered and "schema" not in rendered) or bool(
        re.search(r"(^|[^a-z])(dict|mapping)([^a-z]|$)", rendered)
    )


def _is_path_annotation(annotation: ast.expr | None) -> bool:
    return bool(re.search(r"(^|[^a-z])path([^a-z]|$)", _annotation(annotation)))


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _rendered_in(node: ast.AST, names: set[str]) -> bool:
    return ast.unparse(node) in names


def _mapping_value(node: ast.AST, names: set[str]) -> bool:
    if _rendered_in(node, names):
        return True
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _mapping_value(node.value, names)
    if not isinstance(node, ast.Call):
        return False
    name = _call_name(node).lower()
    if name in {
        "as_config_mapping",
        "as_mapping",
        "exact_config_mapping",
        "exact_mapping",
        "require_config_mapping",
        "_exact",
        "_mapping",
        "_model_mapping",
        "_plain",
    }:
        return True
    configuration_arguments = (*node.args, *(item.value for item in node.keywords))
    if name.startswith("validate_"):
        return any(
            _contains_mapping_value(argument, names)
            for argument in configuration_arguments
        )
    if name == "validate" and isinstance(node.func, ast.Attribute):
        receiver = ast.unparse(node.func.value).lower()
        return "schema" in receiver or any(
            _contains_mapping_value(argument, names)
            for argument in configuration_arguments
        )
    if name == "cast" and len(node.args) >= 2:
        return _is_mapping_annotation(node.args[0])
    return False


def _contains_mapping_value(node: ast.AST, names: set[str]) -> bool:
    return any(
        isinstance(child, (ast.Name, ast.Attribute, ast.Subscript, ast.Call))
        and _mapping_value(child, names)
        for child in ast.walk(node)
    )


def _is_explicit_boolean_value(node: ast.AST) -> bool:
    """Return whether one ``or`` operand is intrinsically a truth predicate."""
    return (
        isinstance(node, (ast.Compare, ast.BoolOp))
        or isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        or isinstance(node, ast.Constant)
        and type(node.value) is bool
        or isinstance(node, ast.Call)
        and _call_name(node) in {"all", "any", "bool", "isinstance", "issubclass"}
    )


def _is_predicate_expression(
    node: ast.AST,
    parents: dict[int, ast.AST],
) -> bool:
    """Return whether an expression contributes to control-flow truth testing."""
    current = node
    parent = parents.get(id(current))
    while isinstance(parent, (ast.BoolOp, ast.UnaryOp)):
        current = parent
        parent = parents.get(id(current))
    return (
        isinstance(parent, (ast.If, ast.IfExp, ast.While, ast.Assert))
        and parent.test is current
        or isinstance(parent, ast.comprehension)
        and current in parent.ifs
    )


def _path_call(node: ast.Call, names: set[str]) -> bool:
    if isinstance(node.func, ast.Name):
        if node.func.id == "Path":
            return True
        return node.func.id == "cast" and bool(node.args) and _is_path_annotation(
            node.args[0]
        )
    if not isinstance(node.func, ast.Attribute):
        return False
    receiver = node.func.value
    rendered = ast.unparse(receiver)
    if node.func.attr in {"resolve", "validate", "root"} and (
        rendered.rsplit(".", maxsplit=1)[-1].endswith("resolver")
        or rendered.rsplit(".", maxsplit=1)[-1] == "roots"
    ):
        return True
    if node.func.attr in {
        "data",
        "project",
        "checkpoint",
        "output",
        "cache",
        "artifact",
        "external_asset",
    } and (
        rendered.rsplit(".", maxsplit=1)[-1].endswith("paths")
        or "RuntimePaths" in rendered
    ):
        return True
    return node.func.attr in {"absolute", "expanduser", "joinpath", "resolve"} and _path_value(
        receiver, names
    )


def _resolved_boundary_path(node: ast.AST, names: set[str]) -> bool:
    if not isinstance(node, ast.Attribute) or node.attr != "path":
        return False
    if _rendered_in(node.value, names):
        return True
    return (
        isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "declared"
    )


def _runtime_root_path(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr.endswith("_root")
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "roots"
    )


def _path_value(node: ast.AST, names: set[str]) -> bool:
    if _rendered_in(node, names):
        return True
    if isinstance(node, ast.Call):
        return _path_call(node, names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _path_value(node.left, names)
    if isinstance(node, ast.Attribute):
        return (
            _resolved_boundary_path(node, names)
            or _runtime_root_path(node)
            or node.attr in {"parent", "parents"}
            and _path_value(node.value, names)
        )
    if isinstance(node, ast.Subscript):
        return _path_value(node.value, names)
    if isinstance(node, ast.IfExp):
        return _path_value(node.body, names) and _path_value(node.orelse, names)
    return False


def _targets(statement: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
    return tuple(
        ast.unparse(target)
        for target in targets
        if isinstance(target, (ast.Name, ast.Attribute))
    )


def _scope_nodes(body: Sequence[ast.stmt]) -> tuple[ast.AST, ...]:
    nodes: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        nodes.append(node)
        if isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in body:
        visit(statement)
    return tuple(nodes)


def _iterates_path(node: ast.AST, paths: set[str]) -> bool:
    if _path_value(node, paths):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_path_value(item, paths) for item in node.elts)
    if isinstance(node, ast.Call) and _call_name(node) in {
        "sorted",
        "list",
        "tuple",
        "iter",
    }:
        return bool(node.args) and _iterates_path(node.args[0], paths)
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "declared_many"
    )


def _contains_none_comparison(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Compare)
        and any(
            isinstance(comparator, ast.Constant) and comparator.value is None
            for comparator in child.comparators
        )
        for child in ast.walk(node)
    )


def _none_compared_names(node: ast.AST) -> set[str]:
    return {
        name.id
        for comparison in ast.walk(node)
        if isinstance(comparison, ast.Compare)
        and (
            isinstance(comparison.left, ast.Constant)
            and comparison.left.value is None
            or any(
                isinstance(comparator, ast.Constant) and comparator.value is None
                for comparator in comparison.comparators
            )
        )
        for name in ast.walk(comparison)
        if isinstance(name, ast.Name)
    }


def _declared_global_names(body: Sequence[ast.stmt]) -> set[str]:
    return {
        name
        for statement in body
        if isinstance(statement, ast.Global)
        for name in statement.names
    }


def _preserves_explicit_none(node: ast.IfExp) -> bool:
    """Return whether a conditional conversion keeps absence as ``None``."""
    return any(
        isinstance(branch, ast.Constant) and branch.value is None
        for branch in (node.body, node.orelse)
    )


def _if_assigns_tested_name(node: ast.If) -> bool:
    tested_names = {
        child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)
    }
    return any(
        isinstance(child, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id in tested_names
            for target in (
                child.targets if isinstance(child, ast.Assign) else (child.target,)
            )
        )
        for statement in node.body
        for child in ast.walk(statement)
    )


def _carriers(
    body: Sequence[ast.stmt],
    arguments: Sequence[ast.arg],
    *,
    callable_name: str,
    initial_mappings: set[str] | None = None,
    initial_paths: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    mappings = set() if initial_mappings is None else set(initial_mappings)
    mappings.update({
        argument.arg
        for argument in arguments
        if argument.arg.lower() in {"cfg", "config"}
        or argument.arg.lower().endswith(("_cfg", "_config"))
        or (
            "config" in _annotation(argument.annotation)
            and "schema" not in _annotation(argument.annotation)
        )
        or (
            _is_mapping_annotation(argument.annotation)
            and any(
                token in callable_name
                for token in ("config", "mapping", "schema", "validate", "boundary")
            )
        )
    })
    paths = set() if initial_paths is None else set(initial_paths)
    paths.update({
        argument.arg for argument in arguments if _is_path_annotation(argument.annotation)
    })
    statements = _scope_nodes(body)
    for statement in statements:
        if isinstance(statement, ast.AnnAssign):
            if (
                "config" in _annotation(statement.annotation)
                and "schema" not in _annotation(statement.annotation)
            ) or (
                _is_mapping_annotation(statement.annotation)
                and any(
                    token in callable_name
                    for token in (
                        "config",
                        "mapping",
                        "schema",
                        "validate",
                        "boundary",
                    )
                )
            ):
                mappings.update(_targets(statement))
            if _is_path_annotation(statement.annotation):
                paths.update(_targets(statement))
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            targets = tuple(
                ast.unparse(target)
                for target in ast.walk(statement.target)
                if isinstance(target, (ast.Name, ast.Attribute))
            )
            if _mapping_value(statement.iter, mappings):
                mappings.update(targets)
            if _iterates_path(statement.iter, paths):
                paths.update(targets)
    changed = True
    while changed:
        changed = False
        for statement in statements:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            value = statement.value
            if value is None:
                continue
            targets = _targets(statement)
            before = len(mappings)
            if _mapping_value(value, mappings):
                mappings.update(targets)
            changed |= len(mappings) != before
            before = len(paths)
            if _path_value(value, paths):
                paths.update(targets)
            changed |= len(paths) != before
    return mappings, paths


class _OracleVisitor(ast.NodeVisitor):
    def __init__(self, module: str, tree: ast.Module) -> None:
        self.module = module
        self.scope: list[str] = []
        mappings, paths = _carriers(tree.body, (), callable_name="<module>")
        self.mapping_names = [mappings]
        self.path_names = [paths]
        self.global_names: list[set[str]] = [set()]
        self.occurrences: set[SourceOccurrence] = set()
        self._parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }

    @property
    def qualified_name(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def _record(
        self,
        node: ast.expr | ast.stmt,
        category: OracleCategory,
    ) -> None:
        self.occurrences.add(
            SourceOccurrence(
                self.module,
                self.qualified_name,
                node.lineno,
                node.col_offset,
                category,
            )
        )

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        mappings, paths = _carriers(
            node.body,
            arguments,
            callable_name=node.name,
            initial_mappings=self.mapping_names[-1],
            initial_paths=self.path_names[-1],
        )
        self.mapping_names.append(mappings)
        self.path_names.append(paths)
        self.global_names.append(_declared_global_names(node.body))
        for default in node.args.defaults:
            self._record(default, OracleCategory.PYTHON_RUNTIME_DEFAULT)
        for keyword_default in node.args.kw_defaults:
            if keyword_default is not None:
                self._record(
                    keyword_default,
                    OracleCategory.PYTHON_RUNTIME_DEFAULT,
                )
        self.generic_visit(node)
        self.path_names.pop()
        self.mapping_names.pop()
        self.global_names.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        class_mappings: set[str] = set()
        class_paths: set[str] = set()
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and statement.value is not None:
                self._record(statement.value, OracleCategory.PYTHON_RUNTIME_DEFAULT)
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                if (
                    "config" in _annotation(statement.annotation)
                    and "schema" not in _annotation(statement.annotation)
                ):
                    class_mappings.add(f"self.{statement.target.id}")
                if _is_path_annotation(statement.annotation):
                    class_paths.add(f"self.{statement.target.id}")
        constructor = next(
            (
                statement
                for statement in node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == "__init__"
            ),
            None,
        )
        if constructor is not None:
            arguments = (
                *constructor.args.posonlyargs,
                *constructor.args.args,
                *constructor.args.kwonlyargs,
            )
            constructor_mappings, constructor_paths = _carriers(
                constructor.body,
                arguments,
                callable_name=constructor.name,
            )
            class_mappings.update(
                name for name in constructor_mappings if name.startswith("self.")
            )
            class_paths.update(
                name for name in constructor_paths if name.startswith("self.")
            )
        self.mapping_names.append(self.mapping_names[-1] | class_mappings)
        self.path_names.append(self.path_names[-1] | class_paths)
        self.generic_visit(node)
        self.path_names.pop()
        self.mapping_names.pop()
        self.scope.pop()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr != "get" and _mapping_value(node.value, self.mapping_names[-1]):
            self._record(node, OracleCategory.CONFIGURATION_REFERENCE)
        if _path_value(node, self.path_names[-1]):
            self._record(node, OracleCategory.PATH_RESOLUTION)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if _mapping_value(node.value, self.mapping_names[-1]):
            self._record(node, OracleCategory.CONFIGURATION_REFERENCE)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "setdefault"}
            and _mapping_value(node.func.value, self.mapping_names[-1])
        ):
            self._record(node, OracleCategory.CONFIGURATION_FALLBACK)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 3
            and _mapping_value(node.args[0], self.mapping_names[-1])
        ):
            self._record(node, OracleCategory.CONFIGURATION_FALLBACK)
        if _path_call(node, self.path_names[-1]):
            self._record(node, OracleCategory.PATH_RESOLUTION)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Div) and _path_value(node.left, self.path_names[-1]):
            self._record(node, OracleCategory.PATH_RESOLUTION)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if (
            isinstance(node.op, ast.Or)
            and not _is_predicate_expression(node, self._parents)
            and not any(_is_explicit_boolean_value(value) for value in node.values)
            and _contains_mapping_value(
                node,
                self.mapping_names[-1],
            )
        ):
            self._record(node, OracleCategory.PYTHON_RUNTIME_DEFAULT)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        if (
            _contains_none_comparison(node.test)
            and not _preserves_explicit_none(node)
            and _contains_mapping_value(
                node,
                self.mapping_names[-1],
            )
        ):
            self._record(node, OracleCategory.PYTHON_RUNTIME_DEFAULT)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if (
            _contains_none_comparison(node.test)
            and _if_assigns_tested_name(node)
            and not (_none_compared_names(node.test) & self.global_names[-1])
            and _contains_mapping_value(node.test, self.mapping_names[-1])
        ):
            self._record(node, OracleCategory.PYTHON_RUNTIME_DEFAULT)
        self.generic_visit(node)


def inspect_raw_source(source_root: Path) -> tuple[SourceOccurrence, ...]:
    """Return the independent occurrence set for all Python source files."""
    occurrences: set[SourceOccurrence] = set()
    for path in sorted(source_root.rglob("*.py")):
        module = ".".join(path.relative_to(source_root.parent).with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _OracleVisitor(module, tree)
        visitor.visit(tree)
        occurrences.update(visitor.occurrences)
    return tuple(sorted(occurrences))


__all__ = [
    "OracleCategory",
    "SourceOccurrence",
    "inspect_raw_source",
    "occurrence_id",
]
