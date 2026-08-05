"""Executable AST audit for prohibited configuration and path resolution routes."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
import zlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.configuration.inventory import (
    DEFAULT_AUDIT_INVENTORY,
    EXPECTED_AUDIT_EXEMPTIONS,
    EXPECTED_AUDIT_RULES,
    EXPECTED_MIGRATION_RECORD_COUNT,
    EXPECTED_MIGRATION_RECORD_IDS,
    EXPECTED_RUNTIME_BOUNDARIES,
    AuditExemption,
    AuditInventory,
    AuditRule,
    BoundaryKind,
    MigrationAuthorityKind,
    MigrationCategory,
    MigrationRecord,
    MigrationStatus,
    audit_exemption_reason_code,
    migration_entrypoint_coverage,
    migration_manifest_digest,
    migration_route_audit_rule,
)
from src.utils.configuration.migration_data import MIGRATION_LEDGER_SHA256
from src.utils.configuration.source_oracle import inspect_raw_source, occurrence_id

__all__ = [
    "AuditFinding",
    "AuditReport",
    "BoundaryAuditIssue",
    "DiscoveredRuntimeBoundary",
    "MigrationAuditIssue",
    "audit_source",
    "discover_runtime_boundaries",
    "inspect_source",
    "main",
    "regenerate_exemption_rows",
    "regenerate_migration_rows",
    "write_generated_inventory_data",
]


@dataclass(frozen=True, slots=True, order=True)
class AuditFinding:
    """One statically detected site relevant to the strict-config migration."""

    module: str
    qualified_name: str
    line: int
    column: int
    rule: AuditRule


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Unclassified sites, stale exemptions, and migration-ledger violations."""

    unclassified: tuple[AuditFinding, ...]
    stale_exemptions: tuple[AuditExemption, ...]
    migration_issues: tuple[MigrationAuditIssue, ...]
    boundary_issues: tuple[BoundaryAuditIssue, ...] = field(default_factory=tuple)
    discovered_boundaries: tuple[DiscoveredRuntimeBoundary, ...] = field(
        default_factory=tuple
    )

    @property
    def passed(self) -> bool:
        """Whether source and its checked-in classifications agree exactly."""
        return (
            not self.unclassified
            and not self.stale_exemptions
            and not self.migration_issues
            and not self.boundary_issues
        )


@dataclass(frozen=True, slots=True, order=True)
class MigrationAuditIssue:
    """One exactness, completeness, or staleness failure in the ledger."""

    record_id: str
    reason: str


@dataclass(frozen=True, slots=True, order=True)
class BoundaryAuditIssue:
    """One omitted, stale, unbound, or silently non-executable boundary."""

    boundary_id: str
    reason: str


@dataclass(frozen=True, slots=True, order=True)
class DiscoveredRuntimeBoundary:
    """Source-only discovery result for one actual invocation boundary."""

    module: str
    callable_name: str
    kind: BoundaryKind
    executable_module: bool
    validator_key: str | None
    validator_callable: str | None
    subprocess_invokers: tuple[str, ...] = ()


class _Visitor(ast.NodeVisitor):
    def __init__(self, module: str, tree: ast.Module) -> None:
        self.module = module
        self.scope: list[str] = []
        self.scope_kinds: list[str] = []
        module_mappings, module_paths = _semantic_names(tree.body, (), callable_name="")
        self.mapping_names: list[set[str]] = [module_mappings]
        self.path_names: list[set[str]] = [module_paths]
        self.global_names: list[set[str]] = [set()]
        self.findings: list[AuditFinding] = []
        self.migration_routes: Counter[tuple[str, str, MigrationCategory, str]] = (
            Counter()
        )
        self.route_sites: list[tuple[str, str, int, int, MigrationCategory, str]] = []
        self.non_path_divisions: set[tuple[str, str, str]] = set()
        self.source_expressions: set[tuple[str, str, str]] = set()
        self.symbols: set[str] = {module}
        self._parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        self._docstrings = {
            id(body[0].value)
            for parent in ast.walk(tree)
            if isinstance(
                parent,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
            and (body := parent.body)
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        }

    @property
    def qualified_name(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def visit(self, node: ast.AST) -> None:
        """Visit one node while retaining its lexical source-expression identity."""
        if isinstance(node, ast.expr):
            self.source_expressions.add(
                (self.module, self.qualified_name, ast.unparse(node))
            )
        super().visit(node)

    def _record(self, node: ast.expr | ast.stmt, rule: AuditRule) -> None:
        self.findings.append(
            AuditFinding(
                module=self.module,
                qualified_name=self.qualified_name,
                line=node.lineno,
                column=node.col_offset,
                rule=rule,
            )
        )

    def _record_route(
        self,
        node: ast.expr | ast.stmt,
        category: MigrationCategory,
        route: str,
    ) -> None:
        self.migration_routes[(self.module, self.qualified_name, category, route)] += 1
        self.route_sites.append(
            (
                self.module,
                self.qualified_name,
                node.lineno,
                node.col_offset,
                category,
                route,
            )
        )

    def _record_rule_route(self, node: ast.expr | ast.stmt, rule: AuditRule) -> None:
        category = (
            MigrationCategory.PATH_RESOLUTION
            if rule
            in {
                AuditRule.HYDRA_ABSOLUTE_PATH,
                AuditRule.FILE_PARENT_INDEX,
                AuditRule.RUNTIME_PATH_LITERAL,
                AuditRule.PATH_JOIN,
                AuditRule.PROCESS_CWD,
                AuditRule.HYDRA_RUN_DIRECTORY,
            }
            else MigrationCategory.PYTHON_RUNTIME_DEFAULT
            if rule is AuditRule.NULL_COALESCING
            else MigrationCategory.CONFIGURATION_FALLBACK
        )
        self._record_route(
            node,
            category,
            f"{rule.value}: {ast.unparse(node)}",
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.scope_kinds.append("class")
        class_mappings, class_paths = _class_semantic_attributes(node)
        self.mapping_names.append(self.mapping_names[-1] | class_mappings)
        self.path_names.append(self.path_names[-1] | class_paths)
        self.symbols.add(f"{self.module}.{self.qualified_name}")
        self.generic_visit(node)
        self.path_names.pop()
        self.mapping_names.pop()
        self.scope_kinds.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.scope_kinds.append("function")
        mappings, paths = _semantic_names(
            node.body,
            _function_arguments(node),
            callable_name=node.name,
            initial_mappings=self.mapping_names[-1],
            initial_paths=self.path_names[-1],
        )
        self.mapping_names.append(mappings)
        self.path_names.append(paths)
        self.global_names.append(_declared_global_names(node.body))
        self.symbols.add(f"{self.module}.{self.qualified_name}")
        self._record_function_defaults(node)
        self.generic_visit(node)
        self.path_names.pop()
        self.mapping_names.pop()
        self.global_names.pop()
        self.scope_kinds.pop()
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.scope_kinds.append("function")
        mappings, paths = _semantic_names(
            node.body,
            _function_arguments(node),
            callable_name=node.name,
            initial_mappings=self.mapping_names[-1],
            initial_paths=self.path_names[-1],
        )
        self.mapping_names.append(mappings)
        self.path_names.append(paths)
        self.global_names.append(_declared_global_names(node.body))
        self.symbols.add(f"{self.module}.{self.qualified_name}")
        self._record_function_defaults(node)
        self.generic_visit(node)
        self.path_names.pop()
        self.mapping_names.pop()
        self.global_names.pop()
        self.scope_kinds.pop()
        self.scope.pop()

    def _record_function_defaults(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        positional = (*node.args.posonlyargs, *node.args.args)
        defaults: tuple[ast.expr | None, ...] = (
            *((None,) * (len(positional) - len(node.args.defaults))),
            *node.args.defaults,
        )
        for argument, default in zip(positional, defaults, strict=True):
            if default is not None:
                self._record_route(
                    default,
                    MigrationCategory.PYTHON_RUNTIME_DEFAULT,
                    f"argument {argument.arg}={ast.unparse(default)}",
                )
        for argument, default in zip(
            node.args.kwonlyargs, node.args.kw_defaults, strict=True
        ):
            if default is not None:
                self._record_route(
                    default,
                    MigrationCategory.PYTHON_RUNTIME_DEFAULT,
                    f"argument {argument.arg}={ast.unparse(default)}",
                )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self.scope and isinstance(node.target, ast.Name):
            self.symbols.add(f"{self.module}.{node.target.id}")
        if (
            self.scope_kinds
            and self.scope_kinds[-1] == "class"
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            self._record_route(
                node.value,
                MigrationCategory.PYTHON_RUNTIME_DEFAULT,
                f"field {node.target.id}={ast.unparse(node.value)}",
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self.scope:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.symbols.add(f"{self.module}.{target.id}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            configured_receiver = _is_configuration_route(
                node.func.value,
                self.mapping_names[-1],
            )
            if (
                configured_receiver
                and node.func.attr == "get"
                and len(node.args) == 1
                and not node.keywords
            ):
                self._record(node, AuditRule.GET_WITHOUT_FALLBACK)
                self._record_rule_route(node, AuditRule.GET_WITHOUT_FALLBACK)
            if configured_receiver and node.func.attr == "get" and len(node.args) >= 2:
                self._record(node, AuditRule.GET_WITH_FALLBACK)
                self._record_rule_route(node, AuditRule.GET_WITH_FALLBACK)
            if (
                node.func.attr == "get"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr == "get"
                and _is_configuration_route(
                    node.func.value.func.value,
                    self.mapping_names[-1],
                )
            ):
                self._record(node, AuditRule.CHAINED_GET)
                self._record_rule_route(node, AuditRule.CHAINED_GET)
            if configured_receiver and node.func.attr == "setdefault":
                self._record(node, AuditRule.SETDEFAULT)
                self._record_rule_route(node, AuditRule.SETDEFAULT)
            if node.func.attr == "to_absolute_path":
                self._record(node, AuditRule.HYDRA_ABSOLUTE_PATH)
                self._record_rule_route(node, AuditRule.HYDRA_ABSOLUTE_PATH)
            if node.func.attr in {"cwd", "getcwd"} and (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"Path", "os"}
                or isinstance(node.func.value, ast.Attribute)
                and ast.unparse(node.func.value) == "os.path"
            ):
                self._record(node, AuditRule.PROCESS_CWD)
                self._record_rule_route(node, AuditRule.PROCESS_CWD)
        if isinstance(node.func, ast.Name) and node.func.id == "to_absolute_path":
            self._record(node, AuditRule.HYDRA_ABSOLUTE_PATH)
            self._record_rule_route(node, AuditRule.HYDRA_ABSOLUTE_PATH)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 3
            and _is_configuration_route(node.args[0], self.mapping_names[-1])
        ):
            self._record(node, AuditRule.GETATTR_WITH_FALLBACK)
            self._record_rule_route(node, AuditRule.GETATTR_WITH_FALLBACK)
        if _is_path_construction_call(
            node,
            self.mapping_names[-1],
            self.path_names[-1],
        ):
            self._record_route(
                node,
                MigrationCategory.PATH_RESOLUTION,
                f"path-construction: {ast.unparse(node)}",
            )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        configured_path = _contains_configuration_route(
            node, self.mapping_names[-1]
        )
        if isinstance(node.op, ast.Div):
            if _looks_like_path_join(node, self.path_names[-1]):
                self._record(node, AuditRule.PATH_JOIN)
                self._record_route(
                    node,
                    MigrationCategory.PATH_RESOLUTION,
                    ("configured-path-join: " if configured_path else "path-join: ")
                    + ast.unparse(node),
                )
            else:
                self.non_path_divisions.add(
                    (self.module, self.qualified_name, ast.unparse(node))
                )
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if (
            isinstance(node.op, ast.Or)
            and not _is_predicate_expression(node, self._parents)
            and not any(_is_explicit_boolean_value(value) for value in node.values)
            and _contains_configuration_route(
                node,
                self.mapping_names[-1],
            )
        ):
            self._record(node, AuditRule.NULL_COALESCING)
            self._record_rule_route(node, AuditRule.NULL_COALESCING)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        if (
            _contains_none_comparison(node.test)
            and not _preserves_explicit_none(node)
            and _contains_configuration_route(
                node,
                self.mapping_names[-1],
            )
        ):
            self._record(node, AuditRule.NULL_COALESCING)
            self._record_rule_route(node, AuditRule.NULL_COALESCING)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if (
            _contains_none_comparison(node.test)
            and _if_assigns_tested_name(node)
            and not (_none_compared_names(node.test) & self.global_names[-1])
            and _contains_configuration_route(node.test, self.mapping_names[-1])
        ):
            self._record(node, AuditRule.NULL_COALESCING)
            self._record_rule_route(node, AuditRule.NULL_COALESCING)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if (
            id(node) not in self._docstrings
            and isinstance(node.value, str)
            and _looks_like_runtime_path_literal(node.value)
        ):
            self._record(node, AuditRule.RUNTIME_PATH_LITERAL)
            self._record_rule_route(node, AuditRule.RUNTIME_PATH_LITERAL)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr != "get" and _is_configuration_route(
            node.value, self.mapping_names[-1]
        ):
            self._record_route(
                node,
                MigrationCategory.CONFIGURATION_REFERENCE,
                f"direct: {ast.unparse(node)}",
            )
        if _is_verified_path_value(node, self.path_names[-1]):
            self._record_route(
                node,
                MigrationCategory.PATH_RESOLUTION,
                f"resolved-path-access: {ast.unparse(node)}",
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if _is_configuration_route(node.value, self.mapping_names[-1]):
            self._record_route(
                node,
                MigrationCategory.CONFIGURATION_REFERENCE,
                f"direct: {ast.unparse(node)}",
            )
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and _contains_dunder_file(node.value.value)
        ):
            self._record(node, AuditRule.FILE_PARENT_INDEX)
            self._record_rule_route(node, AuditRule.FILE_PARENT_INDEX)
        self.generic_visit(node)


def _contains_dunder_file(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id == "__file__"
        for child in ast.walk(node)
    )


_RUNTIME_PATH_SEGMENTS = (
    "data/",
    "outputs/",
    "output/",
    "checkpoints/",
    "checkpoint/",
    "ckpt/",
    "cache/",
    ".cache/",
    "third_party/",
    "external/",
    "build/",
)
def _looks_like_runtime_path_literal(value: str) -> bool:
    normalized = value.strip().replace("\\", "/").lower()
    if (
        not normalized
        or "\n" in normalized
        or normalized.startswith(("http://", "https://"))
    ):
        return False
    return any(segment in normalized for segment in _RUNTIME_PATH_SEGMENTS)


_MAPPING_CALL_NAMES = {
    "as_config_mapping",
    "as_mapping",
    "exact_config_mapping",
    "exact_mapping",
    "require_config_mapping",
    "_exact",
    "_mapping",
    "_model_mapping",
    "_plain",
}
_PATH_CALL_NAMES = {"Path", "resolve", "validate", "root"}
_PATH_NAME_SUFFIXES = ("_path", "_root", "_dir", "_directory", "_file")


def _function_arguments(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.arg, ...]:
    return (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    )


def _annotation_name(annotation: ast.expr | None) -> str:
    return "" if annotation is None else ast.unparse(annotation).lower()


def _mapping_annotation(annotation: ast.expr | None) -> bool:
    rendered = _annotation_name(annotation)
    return ("config" in rendered and "schema" not in rendered) or bool(
        re.search(r"(^|[^a-z])(dict|mapping)([^a-z]|$)", rendered)
    )


def _path_annotation(annotation: ast.expr | None) -> bool:
    rendered = _annotation_name(annotation)
    return bool(re.search(r"(^|[^a-z])path([^a-z]|$)", rendered))


def _assigned_names(statement: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
    return tuple(
        ast.unparse(target)
        for target in targets
        if isinstance(target, (ast.Name, ast.Attribute))
    )


def _bound_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, (ast.Name, ast.Attribute)):
        return (ast.unparse(target),)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(name for child in target.elts for name in _bound_names(child))
    return ()


def _scope_nodes(body: Sequence[ast.stmt]) -> tuple[ast.AST, ...]:
    """Return nodes in one lexical scope without leaking nested definitions."""
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


def _call_returns_mapping(node: ast.AST, mapping_names: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = _call_name(node).lower()
    if name in _MAPPING_CALL_NAMES:
        return True
    configuration_arguments = (*node.args, *(item.value for item in node.keywords))
    if name.startswith("validate_"):
        return any(
            _contains_configuration_route(argument, mapping_names)
            for argument in configuration_arguments
        )
    if name == "validate" and isinstance(node.func, ast.Attribute):
        receiver = ast.unparse(node.func.value).lower()
        return "schema" in receiver or any(
            _contains_configuration_route(argument, mapping_names)
            for argument in configuration_arguments
        )
    if name == "cast" and len(node.args) >= 2:
        return "mapping" in ast.unparse(node.args[0]).lower() or "dict" in ast.unparse(
            node.args[0]
        ).lower()
    return False


def _call_returns_path(node: ast.AST, path_names: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name) and node.func.id == "Path":
        return True
    if isinstance(node.func, ast.Name) and node.func.id == "cast" and node.args:
        return _path_annotation(node.args[0])
    if not isinstance(node.func, ast.Attribute):
        return False
    receiver = node.func.value
    receiver_text = ast.unparse(receiver)
    if node.func.attr in {"resolve", "validate", "root"} and (
        receiver_text == "resolver"
        or receiver_text.endswith(".resolver")
        or receiver_text == "PathResolver"
        or receiver_text.rsplit(".", maxsplit=1)[-1].endswith("resolver")
        or receiver_text.rsplit(".", maxsplit=1)[-1] == "roots"
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
        receiver_text.rsplit(".", maxsplit=1)[-1].endswith("paths")
        or "RuntimePaths" in receiver_text
    ):
        return True
    return node.func.attr in {"absolute", "expanduser", "joinpath", "resolve"} and (
        _is_verified_path_value(receiver, path_names)
        or isinstance(receiver, ast.Name)
        and receiver.id == "Path"
    )


def _is_verified_path_value(node: ast.AST, path_names: set[str]) -> bool:
    rendered = ast.unparse(node)
    if rendered in path_names:
        return True
    if isinstance(node, ast.Call):
        return _call_returns_path(node, path_names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _is_verified_path_value(node.left, path_names)
    if isinstance(node, ast.IfExp):
        return _is_verified_path_value(
            node.body, path_names
        ) and _is_verified_path_value(node.orelse, path_names)
    if isinstance(node, ast.Attribute):
        return (
            node.attr == "path"
            and (
                ast.unparse(node.value) in path_names
                or isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "declared"
            )
            or node.attr.endswith("_root")
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "roots"
            or node.attr in {"parent", "parents"}
            and _is_verified_path_value(node.value, path_names)
        )
    if isinstance(node, ast.Subscript):
        return _is_verified_path_value(node.value, path_names)
    return False


def _iterates_verified_path(node: ast.AST, path_names: set[str]) -> bool:
    if _is_verified_path_value(node, path_names):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_is_verified_path_value(item, path_names) for item in node.elts)
    if isinstance(node, ast.Call) and _call_name(node) in {
        "sorted",
        "list",
        "tuple",
        "iter",
    }:
        return bool(node.args) and _iterates_verified_path(node.args[0], path_names)
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "declared_many"
    )


def _contains_path_route(node: ast.AST, path_names: set[str]) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id in path_names
        or isinstance(child, ast.Attribute) and ast.unparse(child) in path_names
        or isinstance(child, ast.Call) and _call_returns_path(child, path_names)
        for child in ast.walk(node)
    )


def _semantic_names(
    body: Sequence[ast.stmt],
    arguments: Sequence[ast.arg],
    *,
    callable_name: str,
    initial_mappings: set[str] | None = None,
    initial_paths: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Infer validated mappings and typed paths without relying on local names."""
    mapping_names = set() if initial_mappings is None else set(initial_mappings)
    mapping_names.update({
        argument.arg
        for argument in arguments
        if (
            "config" in _annotation_name(argument.annotation)
            and "schema" not in _annotation_name(argument.annotation)
        )
        or argument.arg.lower() in {"cfg", "config"}
        or argument.arg.lower().endswith(("_cfg", "_config"))
        or (
            _mapping_annotation(argument.annotation)
            and any(
                token in callable_name
                for token in (
                    "config",
                    "mapping",
                    "boundary",
                    "schema",
                    "validate",
                )
            )
        )
    })
    path_names = set() if initial_paths is None else set(initial_paths)
    path_names.update({
        argument.arg for argument in arguments if _path_annotation(argument.annotation)
    })
    scoped_nodes = _scope_nodes(body)
    for call in (
        child
        for child in scoped_nodes
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "__init__"
    ):
        if any(_is_configuration_route(argument, mapping_names) for argument in call.args):
            mapping_names.add("self.config")
    for statement in scoped_nodes:
        if isinstance(statement, ast.AnnAssign):
            assigned = _assigned_names(statement)
            if (
                "config" in _annotation_name(statement.annotation)
                and "schema" not in _annotation_name(statement.annotation)
            ) or (
                _mapping_annotation(statement.annotation)
                and any(
                    token in callable_name
                    for token in ("config", "mapping", "schema", "validate", "boundary")
                )
            ):
                mapping_names.update(assigned)
            if _path_annotation(statement.annotation):
                path_names.update(assigned)
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            bound_names = _bound_names(statement.target)
            if _contains_configuration_route(statement.iter, mapping_names):
                mapping_names.update(bound_names)
            if _iterates_verified_path(statement.iter, path_names):
                path_names.update(bound_names)
    changed = True
    while changed:
        changed = False
        for statement in scoped_nodes:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            value = statement.value
            if value is None:
                continue
            names = _assigned_names(statement)
            if (
                _contains_configuration_route(value, mapping_names)
                or _call_returns_mapping(value, mapping_names)
            ):
                before = len(mapping_names)
                mapping_names.update(names)
                changed |= len(mapping_names) != before
            if _is_verified_path_value(value, path_names):
                before = len(path_names)
                path_names.update(names)
                changed |= len(path_names) != before
    return mapping_names, path_names


def _class_semantic_attributes(node: ast.ClassDef) -> tuple[set[str], set[str]]:
    """Infer typed instance carriers initialized by a class constructor."""
    declared_mappings = {
        f"self.{statement.target.id}"
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and _mapping_annotation(statement.annotation)
    }
    declared_paths = {
        f"self.{statement.target.id}"
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and _path_annotation(statement.annotation)
    }
    constructor = next(
        (
            statement
            for statement in node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == "__init__"
        ),
        None,
    )
    if constructor is None:
        return declared_mappings, declared_paths
    mappings, paths = _semantic_names(
        constructor.body,
        _function_arguments(constructor),
        callable_name=constructor.name,
    )
    return (
        declared_mappings | {name for name in mappings if name.startswith("self.")},
        declared_paths | {name for name in paths if name.startswith("self.")},
    )


def _looks_like_path_join(node: ast.BinOp, path_names: set[str]) -> bool:
    return _is_verified_path_value(node.left, path_names)


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


def _is_configuration_route(node: ast.AST, mapping_names: set[str]) -> bool:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript, ast.Call)):
        if ast.unparse(current) in mapping_names:
            return True
        if isinstance(current, ast.Call):
            return _call_returns_mapping(current, mapping_names)
        current = current.value
    return ast.unparse(current) in mapping_names


def _contains_configuration_route(node: ast.AST, mapping_names: set[str]) -> bool:
    return any(
        (
            isinstance(child, (ast.Attribute, ast.Subscript))
            and _is_configuration_route(child, mapping_names)
        )
        or (isinstance(child, ast.Name) and child.id in mapping_names)
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
    parents: Mapping[int, ast.AST],
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


def _is_path_construction_call(
    node: ast.Call,
    mapping_names: set[str],
    path_names: set[str],
) -> bool:
    if _call_returns_path(node, path_names):
        return True
    if not isinstance(node.func, ast.Attribute):
        return False
    return node.func.attr in {
        "data",
        "project",
        "checkpoint",
        "output",
        "cache",
        "artifact",
        "external_asset",
    } and _contains_configuration_route(node, mapping_names)


def _module_name(source_root: Path, path: Path) -> str:
    return ".".join(path.relative_to(source_root.parent).with_suffix("").parts)


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets: Sequence[ast.expr] = statement.targets
            assigned_value: ast.expr | None = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = (statement.target,)
            assigned_value = statement.value
        else:
            continue
        if isinstance(assigned_value, ast.Constant) and isinstance(
            assigned_value.value, str
        ):
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = assigned_value.value
    return constants


def _resolved_string(node: ast.AST, constants: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _contains_named_call(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Call) and _call_name(child) == name
        for child in ast.walk(node)
    )


def _has_executable_edge(tree: ast.Module) -> bool:
    for statement in tree.body:
        if (
            isinstance(statement, ast.If)
            and "__main__" in ast.unparse(statement.test)
            and any(isinstance(child, ast.Call) for child in ast.walk(statement))
        ):
            return True
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and _call_name(statement.value) in {"main", "_main"}
        ):
            return True
        if isinstance(statement, ast.Raise) and _contains_named_call(statement, "main"):
            return True
    return False


def _hydra_boundary_key(
    call: ast.Call,
    constants: Mapping[str, str],
) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == "validation_boundary":
            return _resolved_string(keyword.value, constants)
    return None


def _hydra_call(node: ast.AST) -> ast.Call | None:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _call_name(child) == "hydra_main":
            return child
    return None


def _validator_callable_symbol(
    module: str,
    node: ast.AST,
    line: int,
    imports: Mapping[str, str],
) -> str:
    if isinstance(node, ast.Name):
        return imports.get(node.id, f"{module}.{node.id}")
    if isinstance(node, ast.Lambda):
        return f"{module}.<lambda>@{line}"
    return f"{module}.{ast.unparse(node)}"


def _validator_registrations(
    module: str,
    tree: ast.Module,
    constants: Mapping[str, str],
) -> dict[str, str]:
    registrations: dict[str, str] = {}
    imports = {
        alias.asname or alias.name: f"{statement.module}.{alias.name}"
        for statement in tree.body
        if isinstance(statement, ast.ImportFrom) and statement.module is not None
        for alias in statement.names
    }
    string_sets: dict[str, tuple[str, ...]] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        target_nodes = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else (statement.target,)
        )
        values: tuple[str, ...] = ()
        if isinstance(value, ast.Dict):
            values = tuple(
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
        elif isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            values = tuple(
                item.value
                for item in value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
        for target in target_nodes:
            if isinstance(target, ast.Name) and values:
                string_sets[target.id] = values

    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or _call_name(node) != "register_boundary_validator"
        ):
            continue
        if len(node.args) != 2:
            continue
        key = _resolved_string(node.args[0], constants)
        if key is not None:
            registrations[key] = _validator_callable_symbol(
                module, node.args[1], node.lineno, imports
            )

    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        if not isinstance(node.iter, ast.Name):
            continue
        for call in (
            child
            for statement in node.body
            for child in ast.walk(statement)
            if isinstance(child, ast.Call)
            and _call_name(child) == "register_boundary_validator"
            and len(child.args) == 2
            and isinstance(child.args[0], ast.Name)
            and child.args[0].id == node.target.id
        ):
            for key in string_sets.get(node.iter.id, ()):
                validator_expression = call.args[1]
                if (
                    isinstance(validator_expression, ast.Call)
                    and len(validator_expression.args) == 1
                    and isinstance(validator_expression.args[0], ast.Name)
                    and validator_expression.args[0].id == node.target.id
                ):
                    rendered = f"{ast.unparse(validator_expression.func)}({key!r})"
                else:
                    rendered = ast.unparse(validator_expression)
                registrations[key] = f"{module}.{rendered}"
    return registrations


def _non_hydra_validator_declarations(
    module: str,
    tree: ast.Module,
) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if (
            not isinstance(value, ast.Call)
            or _call_name(value) != "NonHydraPathBoundary"
        ):
            continue
        name_node: ast.AST | None = value.args[0] if value.args else None
        if name_node is None:
            name_node = next(
                (keyword.value for keyword in value.keywords if keyword.arg == "name"),
                None,
            )
        if name_node is None:
            continue
        boundary_name = _resolved_string(name_node, {})
        if boundary_name is None:
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else (statement.target,)
        )
        for target in targets:
            if isinstance(target, ast.Name):
                declarations[target.id] = boundary_name
    invoked = {
        child.func.value.id
        for child in ast.walk(tree)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "validate"
        and isinstance(child.func.value, ast.Name)
    }
    imported_boundary = next(
        (
            f"{statement.module}.NonHydraPathBoundary.validate"
            for statement in tree.body
            if isinstance(statement, ast.ImportFrom)
            and statement.module is not None
            and any(alias.name == "NonHydraPathBoundary" for alias in statement.names)
        ),
        None,
    )
    validator_callable = imported_boundary or f"{module}.NonHydraPathBoundary.validate"
    if validator_callable == "src.utils.configuration.NonHydraPathBoundary.validate":
        validator_callable = (
            "src.utils.configuration.paths.NonHydraPathBoundary.validate"
        )
    return {
        name: validator_callable
        for variable, name in declarations.items()
        if variable in invoked
    }


_SELF_VALIDATING_COMMANDS = frozenset(
    {
        "src.configuration_validation",
        "src.synthetic_data_generation.config_validation",
        "src.tasks.ball_detection.validation",
        "src.tasks.blcs.validation",
        "src.tasks.plcs.validation_matrix",
        "src.utils.configuration.audit",
        "src.utils.configuration.validation",
    }
)


def _subprocess_module_targets(tree: ast.Module) -> tuple[str, ...]:
    targets: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        for index, item in enumerate(node.elts[:-1]):
            next_item = node.elts[index + 1]
            if (
                isinstance(item, ast.Constant)
                and item.value == "-m"
                and isinstance(next_item, ast.Constant)
                and isinstance(next_item.value, str)
            ):
                targets.add(next_item.value)
    return tuple(sorted(targets))


def discover_runtime_boundaries(
    source_root: Path,
) -> tuple[DiscoveredRuntimeBoundary, ...]:
    """Discover runtime callables and their actual validator bindings from source."""
    trees: dict[str, ast.Module] = {}
    constants_by_module: dict[str, dict[str, str]] = {}
    registrations: dict[str, str] = {}
    subprocess_invokers: dict[str, set[str]] = {}
    for path in sorted(source_root.rglob("*.py")):
        module = _module_name(source_root, path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        trees[module] = tree
        constants = _module_string_constants(tree)
        constants_by_module[module] = constants
        for key, callable_symbol in _validator_registrations(
            module, tree, constants
        ).items():
            if key in registrations and registrations[key] != callable_symbol:
                registrations[key] = "<multiple-bindings>"
            else:
                registrations[key] = callable_symbol
        for target in _subprocess_module_targets(tree):
            subprocess_invokers.setdefault(target, set()).add(module)

    discovered: dict[tuple[str, str], DiscoveredRuntimeBoundary] = {}
    for module, tree in trees.items():
        constants = constants_by_module[module]
        executable = _has_executable_edge(tree)
        non_hydra_validators = _non_hydra_validator_declarations(module, tree)
        functions = {
            function.name: function
            for function in tree.body
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        parser_functions = tuple(
            function
            for function in functions.values()
            if _contains_named_call(function, "ArgumentParser")
        )
        path_validation_functions = tuple(
            function
            for function in functions.values()
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "validate"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "PATH_BOUNDARY"
                for node in ast.walk(function)
            )
        )
        raw_boundaries = (
            path_validation_functions
            if non_hydra_validators and path_validation_functions
            else parser_functions
        )
        for function in raw_boundaries:
            validator_key = (
                next(iter(non_hydra_validators))
                if function in path_validation_functions
                else None
            )
            validator_callable = (
                non_hydra_validators[validator_key]
                if validator_key is not None
                else None
            )
            discovered[(module, function.name)] = DiscoveredRuntimeBoundary(
                module=module,
                callable_name=function.name,
                kind=(
                    BoundaryKind.VALIDATION_COMMAND
                    if module in _SELF_VALIDATING_COMMANDS
                    else BoundaryKind.ARGPARSE
                    if parser_functions
                    else BoundaryKind.CALLABLE
                ),
                executable_module=executable,
                validator_key=validator_key,
                validator_callable=validator_callable,
                subprocess_invokers=tuple(sorted(subprocess_invokers.get(module, ()))),
            )

        for function in functions.values():
            for decorator in function.decorator_list:
                hydra_call = _hydra_call(decorator)
                if hydra_call is None:
                    continue
                validator_key = _hydra_boundary_key(hydra_call, constants)
                discovered[(module, function.name)] = DiscoveredRuntimeBoundary(
                    module=module,
                    callable_name=function.name,
                    kind=BoundaryKind.HYDRA,
                    executable_module=executable,
                    validator_key=validator_key,
                    validator_callable=(
                        registrations.get(validator_key)
                        if validator_key is not None
                        else None
                    ),
                    subprocess_invokers=tuple(
                        sorted(subprocess_invokers.get(module, ()))
                    ),
                )

            for nested in ast.walk(function):
                if nested is function or not isinstance(
                    nested, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                for decorator in nested.decorator_list:
                    hydra_call = _hydra_call(decorator)
                    if hydra_call is None:
                        continue
                    validator_key = _hydra_boundary_key(hydra_call, constants)
                    discovered[(module, function.name)] = DiscoveredRuntimeBoundary(
                        module=module,
                        callable_name=function.name,
                        kind=BoundaryKind.HYDRA,
                        executable_module=executable,
                        validator_key=validator_key,
                        validator_callable=(
                            registrations.get(validator_key)
                            if validator_key is not None
                            else None
                        ),
                        subprocess_invokers=tuple(
                            sorted(subprocess_invokers.get(module, ()))
                        ),
                    )

        if module == "src.synthetic_data_generation.scripts.alignment.geometry_bridge":
            provider = functions.get("provider_main")
            if provider is not None:
                discovered[(module, "provider_main")] = DiscoveredRuntimeBoundary(
                    module=module,
                    callable_name="provider_main",
                    kind=BoundaryKind.CALLABLE,
                    executable_module=executable,
                    validator_key="synthetic.geometry_bridge",
                    validator_callable=(
                        "src.utils.configuration.paths.NonHydraPathBoundary.validate"
                    ),
                    subprocess_invokers=tuple(
                        sorted(subprocess_invokers.get(module, ()))
                    ),
                )

        for statement in tree.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else (statement.target,)
            )
            if not any(
                isinstance(target, ast.Name) and target.id == "main"
                for target in targets
            ):
                continue
            if statement.value is None:
                continue
            hydra_call = _hydra_call(statement.value)
            if hydra_call is None:
                continue
            validator_key = _hydra_boundary_key(hydra_call, constants)
            discovered[(module, "main")] = DiscoveredRuntimeBoundary(
                module=module,
                callable_name="main",
                kind=BoundaryKind.HYDRA,
                executable_module=executable,
                validator_key=validator_key,
                validator_callable=(
                    registrations.get(validator_key)
                    if validator_key is not None
                    else None
                ),
                subprocess_invokers=tuple(sorted(subprocess_invokers.get(module, ()))),
            )

        if not parser_functions and not any(key[0] == module for key in discovered):
            main_function = functions.get("main")
            if main_function is not None:
                kind = (
                    BoundaryKind.VALIDATION_COMMAND
                    if module in _SELF_VALIDATING_COMMANDS
                    else BoundaryKind.SUBPROCESS_MODULE
                    if module in subprocess_invokers
                    or module.endswith(".geometry_bridge")
                    else BoundaryKind.CALLABLE
                )
                discovered[(module, "main")] = DiscoveredRuntimeBoundary(
                    module=module,
                    callable_name="main",
                    kind=kind,
                    executable_module=executable,
                    validator_key=None,
                    validator_callable=None,
                    subprocess_invokers=tuple(
                        sorted(subprocess_invokers.get(module, ()))
                    ),
                )
    return tuple(sorted(discovered.values()))


def regenerate_migration_rows(
    source_root: Path,
    *,
    exemptions: Sequence[AuditExemption] | None = None,
) -> tuple[tuple[object, ...], ...]:
    """Regenerate a truthful current-plus-history ledger from source.

    Current source sites are LIVE or exactly EXEMPTED. Historical rows whose
    complete route disappeared are retained as MIGRATED with zero occurrences.
    The command emits data and never mutates its checked-in source authority.
    """
    sites: list[tuple[str, str, int, int, MigrationCategory, str]] = []
    route_counts: Counter[tuple[str, str, MigrationCategory, str]] = Counter()
    non_path_divisions: set[tuple[str, str, str]] = set()
    source_expressions: set[tuple[str, str, str]] = set()
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _Visitor(_module_name(source_root, path), tree)
        visitor.visit(tree)
        sites.extend(visitor.route_sites)
        route_counts.update(visitor.migration_routes)
        non_path_divisions.update(visitor.non_path_divisions)
        source_expressions.update(visitor.source_expressions)
    for finding in _inspect_yaml(source_root):
        route = f"{finding.rule.value}: yaml-source-route"
        category = MigrationCategory.PATH_RESOLUTION
        sites.append(
            (
                finding.module,
                finding.qualified_name,
                finding.line,
                finding.column,
                category,
                route,
            )
        )
        route_counts[(finding.module, finding.qualified_name, category, route)] += 1

    rows: list[tuple[object, ...]] = []
    seen_sites: set[tuple[str, str, int, int, MigrationCategory, str]] = set()
    current_route_keys = set(route_counts)
    active_exemptions = (
        DEFAULT_AUDIT_INVENTORY.exemptions if exemptions is None else exemptions
    )
    exemption_keys = {_exemption_key(exemption) for exemption in active_exemptions}
    for module, name, line, column, category, route in sorted(sites):
        site = (module, name, line, column, category, route)
        if site in seen_sites:
            continue
        seen_sites.add(site)
        identifier_payload = json.dumps(
            [module, name, line, column, route, category.value],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        record_id = hashlib.sha256(identifier_payload).hexdigest()[:20]
        route_rule = migration_route_audit_rule(route)
        is_exempted = any(
            key[:3] == (module, name, line) and key[3] is route_rule
            for key in exemption_keys
        )
        status = MigrationStatus.EXEMPTED if is_exempted else MigrationStatus.LIVE
        authority_kind, canonical_symbol, authority_field = _live_route_authority(
            module,
            name,
            category,
            route,
        )
        rows.append(
            (
                record_id,
                module,
                name,
                line,
                column,
                route,
                route_counts[(module, name, category, route)],
                category.value,
                status.value,
                authority_kind.value,
                canonical_symbol,
                authority_field,
            )
        )
    current_sites = {
        (str(row[1]), str(row[2]), MigrationCategory(str(row[7])), str(row[5]))
        for row in rows
    }
    current_normalized_routes = {
        (module, name, category, _normalized_route_identity(route))
        for module, name, category, route in current_sites
    }
    for record in DEFAULT_AUDIT_INVENTORY.migrations:
        route_key = (
            record.former_module,
            record.former_qualified_name,
            record.category,
            record.former_route,
        )
        normalized_route_key = (
            record.former_module,
            record.former_qualified_name,
            record.category,
            _normalized_route_identity(record.former_route),
        )
        route_prefix, separator, expression = record.former_route.partition(": ")
        if (
            separator
            and route_prefix in {"path-join", "configured-path-join"}
            and (
                record.former_module,
                record.former_qualified_name,
                expression,
            )
            in non_path_divisions
        ):
            continue
        source_expression = _route_source_expression(record.former_route)
        if source_expression is not None and (
            record.former_module,
            record.former_qualified_name,
            source_expression,
        ) in source_expressions:
            continue
        if (
            route_key in current_route_keys
            or route_key in current_sites
            or normalized_route_key in current_normalized_routes
        ):
            continue
        authority_kind = MigrationAuthorityKind.EXECUTION_BOUNDARY
        canonical_symbol = _replacement_boundary_symbol(record.domain)
        rows.append(
            (
                record.record_id,
                record.former_module,
                record.former_qualified_name,
                record.former_line,
                record.former_column,
                record.former_route,
                0,
                record.category.value,
                MigrationStatus.MIGRATED.value,
                authority_kind.value,
                canonical_symbol,
                None,
            )
        )

    def row_sort_key(row: tuple[object, ...]) -> tuple[str, int, int, str]:
        line = row[3]
        column = row[4]
        if type(line) is not int or type(column) is not int:
            raise AssertionError("Generated migration sites require integer locations.")
        return str(row[1]), line, column, str(row[0])

    return tuple(sorted(rows, key=row_sort_key))


def regenerate_exemption_rows(
    source_root: Path,
    *,
    approved_exemptions: Sequence[AuditExemption] = (),
) -> tuple[
    tuple[tuple[object, ...], ...],
    tuple[AuditExemption, ...],
    tuple[AuditFinding, ...],
]:
    """Re-anchor exact exemptions by unchanged source-route identity.

    New suspicious constructs are deliberately returned as unresolved instead
    of being auto-exempted. Reviewed additions must identify an exact current
    site and stable reason code; stale or duplicate approvals fail closed.
    """
    old_by_key = {_exemption_key(item): item for item in DEFAULT_AUDIT_INVENTORY.exemptions}
    approved_by_key = {_exemption_key(item): item for item in approved_exemptions}
    if len(approved_by_key) != len(approved_exemptions):
        raise ValueError("Approved audit exemptions must have unique exact identities.")
    duplicate_approvals = sorted(set(approved_by_key) & set(old_by_key))
    if duplicate_approvals:
        raise ValueError(
            "Approved audit exemptions already exist in generated data: "
            f"{duplicate_approvals!r}."
        )
    for exemption in approved_exemptions:
        audit_exemption_reason_code(exemption)
    old_reason_by_route: dict[tuple[str, str, str, str], AuditExemption] = {}
    for record in DEFAULT_AUDIT_INVENTORY.migrations:
        matched = next(
            (
                exemption
                for key, exemption in old_by_key.items()
                if key[:3]
                == (
                    record.former_module,
                    record.former_qualified_name,
                    record.former_line,
                )
                and migration_route_audit_rule(record.former_route) is key[3]
            ),
            None,
        )
        if matched is not None:
            route_rule = migration_route_audit_rule(record.former_route)
            old_reason_by_route[
                (
                    record.former_module,
                    record.former_qualified_name,
                    (
                        route_rule.value
                        if route_rule is not None
                        else record.former_route.partition(":")[0]
                    ),
                    _normalized_route_identity(record.former_route),
                )
            ] = matched

    findings: list[AuditFinding] = []
    sites: list[tuple[str, str, int, MigrationCategory, str]] = []
    for path in sorted(source_root.rglob("*.py")):
        module = _module_name(source_root, path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _Visitor(module, tree)
        visitor.visit(tree)
        findings.extend(visitor.findings)
        sites.extend(
            (site_module, name, line, category, route)
            for site_module, name, line, _, category, route in visitor.route_sites
        )
    yaml_findings = _inspect_yaml(source_root)
    findings.extend(yaml_findings)
    sites.extend(
        (
            finding.module,
            finding.qualified_name,
            finding.line,
            MigrationCategory.PATH_RESOLUTION,
            f"{finding.rule.value}: yaml-source-route",
        )
        for finding in yaml_findings
    )

    routes_by_finding: dict[tuple[str, str, int, str], set[str]] = {}
    for module, name, line, _, route in sites:
        route_rule = migration_route_audit_rule(route)
        rule = route_rule.value if route_rule is not None else route.partition(":")[0]
        routes_by_finding.setdefault((module, name, line, rule), set()).add(route)

    generated: dict[tuple[str, str, int, AuditRule], AuditExemption] = {}
    used_approvals: set[tuple[str, str, int, AuditRule]] = set()
    unresolved: list[AuditFinding] = []
    rows: list[tuple[object, ...]] = []
    for finding in sorted(findings):
        current_key = _finding_key(finding)
        previous = old_by_key.get(current_key)
        if previous is None and current_key in approved_by_key:
            previous = approved_by_key[current_key]
            used_approvals.add(current_key)
        routes = tuple(
            sorted(
                routes_by_finding.get(
                    (
                        finding.module,
                        finding.qualified_name,
                        finding.line,
                        finding.rule.value,
                    ),
                    (),
                )
            )
        )
        if previous is None:
            for route in routes:
                previous = old_reason_by_route.get(
                    (
                        finding.module,
                        finding.qualified_name,
                        finding.rule.value,
                        _normalized_route_identity(route),
                    )
                )
                if previous is not None:
                    break
        if previous is None:
            unresolved.append(finding)
            continue
        exemption = AuditExemption(
            module=finding.module,
            qualified_name=finding.qualified_name,
            line=finding.line,
            rule=finding.rule,
            reason=previous.reason,
        )
        generated[_exemption_key(exemption)] = exemption
        rows.append(
            (
                finding.module,
                finding.qualified_name,
                finding.line,
                finding.rule.value,
                audit_exemption_reason_code(previous),
            )
        )
    unique_rows = tuple(sorted(set(rows)))
    unused_approvals = sorted(set(approved_by_key) - used_approvals)
    if unused_approvals:
        raise ValueError(
            "Approved audit exemptions do not match current findings: "
            f"{unused_approvals!r}."
        )
    return unique_rows, tuple(generated[key] for key in sorted(generated)), tuple(unresolved)


def _encoded_payload(rows: Sequence[Sequence[object]]) -> tuple[str, str]:
    serialized = json.dumps(
        rows,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        base64.b85encode(zlib.compress(serialized, level=9)).decode("ascii"),
        hashlib.sha256(serialized).hexdigest(),
    )


def _payload_literal(payload: str) -> str:
    chunks = tuple(payload[index : index + 100] for index in range(0, len(payload), 100))
    return "(\n" + "\n".join(f"    {chunk!r}" for chunk in chunks) + "\n)"


def write_generated_inventory_data(
    source_root: Path,
    *,
    source_revision: str,
    approved_exemptions: Sequence[AuditExemption] = (),
) -> tuple[int, int]:
    """Atomically freeze rebased exemptions and the truthful migration ledger."""
    if not source_revision.strip():
        raise ValueError("source_revision must be a non-empty immutable label.")
    exemption_rows, exemptions, unresolved = regenerate_exemption_rows(
        source_root,
        approved_exemptions=approved_exemptions,
    )
    if unresolved:
        rendered = _render(unresolved)
        raise RuntimeError(
            "Inventory finalization found new unclassified source constructs; "
            "migrate or explicitly classify them before freezing:\n" + rendered
        )
    migration_rows = regenerate_migration_rows(source_root, exemptions=exemptions)

    migration_payload, migration_digest = _encoded_payload(migration_rows)
    migration_ids = sorted(str(row[0]) for row in migration_rows)
    migration_id_digest = hashlib.sha256(
        json.dumps(migration_ids, ensure_ascii=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    migration_counts = dict(Counter(str(row[7]) for row in migration_rows))
    migration_status_counts = dict(Counter(str(row[8]) for row in migration_rows))
    migration_source = (
        '"""Generated immutable strict configuration/path route inventory.\n\n'
        "Each zlib+base85 row contains exact source/former identity, occurrence count,\n"
        "category, truthful LIVE/MIGRATED/EXEMPTED state, and a verifiable authority.\n"
        '"""\n\n'
        f"MIGRATION_SOURCE_REVISION = {source_revision!r}\n"
        f"MIGRATION_LEDGER_RECORD_COUNT = {len(migration_rows)}\n"
        f"MIGRATION_LEDGER_RECORD_IDS_SHA256 = {migration_id_digest!r}\n"
        f"MIGRATION_LEDGER_SHA256 = {migration_digest!r}\n"
        f"MIGRATION_LEDGER_COUNTS = {migration_counts!r}\n"
        f"MIGRATION_LEDGER_STATUS_COUNTS = {migration_status_counts!r}\n"
        f"MIGRATION_LEDGER_PAYLOAD = {_payload_literal(migration_payload)}\n"
    )

    exemption_payload, exemption_digest = _encoded_payload(exemption_rows)
    exemption_ids = sorted(
        f"{row[0]}|{row[1]}|{row[2]}|{row[3]}" for row in exemption_rows
    )
    exemption_id_digest = hashlib.sha256(
        json.dumps(exemption_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    exemption_counts = dict(Counter(str(row[4]) for row in exemption_rows))
    exemption_source = (
        '"""Generated immutable exact exemptions for non-configuration constructs."""\n\n'
        f"AUDIT_EXEMPTION_RECORD_COUNT = {len(exemption_rows)}\n"
        f"AUDIT_EXEMPTION_RECORD_IDS_SHA256 = {exemption_id_digest!r}\n"
        f"AUDIT_EXEMPTION_SHA256 = {exemption_digest!r}\n"
        f"AUDIT_EXEMPTION_COUNTS = {exemption_counts!r}\n"
        f"AUDIT_EXEMPTION_PAYLOAD = {_payload_literal(exemption_payload)}\n"
    )

    target = source_root.joinpath("utils", "configuration")
    temporary_migration = target.joinpath(".migration_data.py.tmp")
    temporary_exemption = target.joinpath(".exemption_data.py.tmp")
    temporary_migration.write_text(migration_source, encoding="utf-8")
    temporary_exemption.write_text(exemption_source, encoding="utf-8")
    temporary_migration.replace(target.joinpath("migration_data.py"))
    temporary_exemption.replace(target.joinpath("exemption_data.py"))
    return len(migration_rows), len(exemption_rows)


def _normalized_route_identity(route: str) -> str:
    """Treat path-join classification upgrades as one surviving source route."""
    prefix, separator, expression = route.partition(": ")
    if separator and prefix in {"path-join", "configured-path-join"}:
        try:
            candidate = ast.parse(expression, mode="eval").body
        except SyntaxError:
            candidate = None
        if (
            isinstance(candidate, ast.BinOp)
            and isinstance(candidate.op, ast.Div)
            and isinstance(candidate.right, (ast.Constant, ast.JoinedStr))
        ):
            return f"path-join-child: {ast.unparse(candidate.right)}"
        return f"path-join: {expression}"
    if separator and prefix in {"configured-path-call", "path-construction"}:
        return f"path-construction: {expression}"
    return route


def _route_source_expression(route: str) -> str | None:
    """Extract the AST expression represented by one historical route."""
    prefix, separator, expression = route.partition(": ")
    if separator:
        if expression == "yaml-source-route":
            return None
        return expression
    if prefix.startswith(("argument ", "field ")):
        _, equals, value = prefix.partition("=")
        return value if equals else None
    return None


def _route_uses_resolver(route: str) -> bool:
    """Return whether the recorded source text invokes the shared resolver."""
    if route == f"{AuditRule.HYDRA_RUN_DIRECTORY.value}: yaml-source-route":
        return True
    return bool(
        re.search(
            r"(?:PathResolver|\bresolver|\.resolver)\.resolve\(",
            route,
        )
    )


def _live_route_authority(
    module: str,
    qualified_name: str,
    category: MigrationCategory,
    route: str,
) -> tuple[MigrationAuthorityKind, str, str | None]:
    """Return the concrete authority directly evidenced by one current route."""
    containing_symbol = (
        module if qualified_name == "<module>" else f"{module}.{qualified_name}"
    )
    if category is not MigrationCategory.PATH_RESOLUTION:
        return MigrationAuthorityKind.EXECUTION_INPUT, containing_symbol, None
    if _route_uses_resolver(route):
        return (
            MigrationAuthorityKind.PATH_RESOLVER,
            "src.utils.configuration.paths.PathResolver.resolve",
            None,
        )
    declared = re.search(r"\.declared\(['\"]([^'\"]+)['\"]\)\.path", route)
    if declared is not None:
        return (
            MigrationAuthorityKind.SCHEMA_FIELD,
            f"{module}.PATH_BOUNDARY",
            declared.group(1),
        )
    runtime_root = re.search(r"\.roots\.([a-z_]+_root)\b", route)
    if runtime_root is not None:
        return (
            MigrationAuthorityKind.SCHEMA_FIELD,
            "src.utils.configuration.paths.RuntimePathRoots",
            runtime_root.group(1),
        )
    return MigrationAuthorityKind.EXECUTION_INPUT, containing_symbol, None


def _replacement_boundary_symbol(domain: str) -> str:
    """Return the executable validation authority replacing a former route."""
    replacements = {
        "ball_detection": "src.tasks.ball_detection.configuration.validate_training",
        "base": "src.tasks.base.configuration.TrainingRuntimeConfig",
        "blcs": "src.tasks.blcs.configuration.validate_training_boundary",
        "court_detection": "src.tasks.court_detection.configuration.validate_train_boundary",
        "plcs": "src.tasks.plcs.configuration.PLCSModelConfig",
        "slcs": "src.tasks.slcs.configuration.SLCSTrainingRuntimeConfig",
        "submodules": "src.submodules.configuration.GvhmrDemoConfig",
        "synthetic_data_generation": "src.synthetic_data_generation.configuration.validate_config",
        "tennis_scene": "src.tennis_scene.configuration.PipelineRuntimeConfig",
        "utils": "src.utils.configuration.schema.StrictConfigSchema",
    }
    return replacements[domain]


def _exemption_key(exemption: AuditExemption) -> tuple[str, str, int, AuditRule]:
    return exemption.module, exemption.qualified_name, exemption.line, exemption.rule


def _finding_key(finding: AuditFinding) -> tuple[str, str, int, AuditRule]:
    return finding.module, finding.qualified_name, finding.line, finding.rule


def audit_source(
    source_root: Path,
    *,
    inventory: AuditInventory = DEFAULT_AUDIT_INVENTORY,
) -> tuple[AuditFinding, ...]:
    """Return unclassified findings under ``source_root`` in stable order."""
    return inspect_source(source_root, inventory=inventory).unclassified


def inspect_source(
    source_root: Path,
    *,
    inventory: AuditInventory = DEFAULT_AUDIT_INVENTORY,
) -> AuditReport:
    """Compare all detected sites with exact, source-owned classifications."""
    exemptions = {_exemption_key(exemption) for exemption in inventory.exemptions}
    findings: list[AuditFinding] = []
    routes: Counter[tuple[str, str, MigrationCategory, str]] = Counter()
    symbols: set[str] = set()
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _Visitor(_module_name(source_root, path), tree)
        visitor.visit(tree)
        findings.extend(visitor.findings)
        routes.update(visitor.migration_routes)
        symbols.update(visitor.symbols)
    yaml_findings = _inspect_yaml(source_root)
    findings.extend(yaml_findings)
    symbols.update(finding.module for finding in yaml_findings)
    symbols.update(f"{finding.module}.<yaml>" for finding in yaml_findings)
    routes.update(
        (
            finding.module,
            finding.qualified_name,
            MigrationCategory.PATH_RESOLUTION,
            f"{finding.rule.value}: yaml-source-route",
        )
        for finding in yaml_findings
    )
    finding_keys = {_finding_key(finding) for finding in findings}
    unclassified = tuple(
        sorted(
            finding for finding in findings if _finding_key(finding) not in exemptions
        )
    )
    stale = tuple(
        exemption
        for exemption in inventory.exemptions
        if _exemption_key(exemption) not in finding_keys
    )
    migration_issues = tuple(
        sorted(
            (
                *_inspect_migrations(
                    inventory,
                    current_routes=routes,
                    current_symbols=symbols,
                ),
                *_inspect_source_oracle(inventory, source_root=source_root),
            )
        )
    )
    discovered_boundaries = discover_runtime_boundaries(source_root)
    boundary_issues = _inspect_boundaries(
        inventory,
        discovered=discovered_boundaries,
    )
    return AuditReport(
        unclassified=unclassified,
        stale_exemptions=stale,
        migration_issues=migration_issues,
        boundary_issues=boundary_issues,
        discovered_boundaries=discovered_boundaries,
    )


def _inspect_yaml(source_root: Path) -> tuple[AuditFinding, ...]:
    findings: list[AuditFinding] = []
    for path in sorted((*source_root.rglob("*.yaml"), *source_root.rglob("*.yml"))):
        module = _module_name(source_root, path)
        hydra_indent: int | None = None
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if stripped == "hydra:":
                hydra_indent = indent
                continue
            if hydra_indent is not None and indent <= hydra_indent:
                hydra_indent = None
            value = stripped.split("#", maxsplit=1)[0].strip()
            if hydra_indent is not None and re.match(r"^(run|sweep)\s*:", value):
                continue
            if hydra_indent is not None and re.match(r"^dir\s*:", value):
                findings.append(
                    AuditFinding(
                        module=module,
                        qualified_name="<yaml>",
                        line=line_number,
                        column=indent,
                        rule=AuditRule.HYDRA_RUN_DIRECTORY,
                    )
                )
            raw_value = (
                value.split(":", maxsplit=1)[1].strip() if ":" in value else value
            )
            if _looks_like_runtime_path_literal(raw_value.strip("'\"[]")):
                findings.append(
                    AuditFinding(
                        module=module,
                        qualified_name="<yaml>",
                        line=line_number,
                        column=indent,
                        rule=AuditRule.RUNTIME_PATH_LITERAL,
                    )
                )
    return tuple(findings)


def _inspect_source_oracle(
    inventory: AuditInventory,
    *,
    source_root: Path,
) -> tuple[MigrationAuditIssue, ...]:
    """Compare the frozen inventory with an independent raw-source oracle."""
    oracle = inspect_raw_source(source_root)
    oracle_ids = {candidate.occurrence_id for candidate in oracle}
    current_records = tuple(
        record
        for record in inventory.migrations
        if record.status in {MigrationStatus.LIVE, MigrationStatus.EXEMPTED}
    )
    ledger_ids = {
        occurrence_id(
            record.former_module,
            record.former_qualified_name,
            record.former_line,
            record.former_column,
            record.category.value,
        )
        for record in current_records
    }
    issues = [
        MigrationAuditIssue(
            record_id="<oracle-missed-route>",
            reason=(
                "independent raw-source occurrence is absent from the inventory: "
                f"{candidate.module}.{candidate.qualified_name}:{candidate.line}:"
                f"{candidate.column} [{candidate.category.value}]"
            ),
        )
        for candidate in oracle
        if candidate.occurrence_id not in ledger_ids
    ]
    for record in current_records:
        if record.category is not MigrationCategory.PATH_RESOLUTION:
            continue
        route_kind = record.former_route.partition(":")[0]
        if route_kind not in {
            "path-join",
            "configured-path-join",
            "path-construction",
            "configured-path-call",
        }:
            continue
        identity = occurrence_id(
            record.former_module,
            record.former_qualified_name,
            record.former_line,
            record.former_column,
            record.category.value,
        )
        if identity not in oracle_ids:
            issues.append(
                MigrationAuditIssue(
                    record_id="<oracle-false-path>",
                    reason=(
                        "inventory path occurrence has no verified Path/resolver "
                        "dataflow in the independent oracle: "
                        f"{record.former_module}.{record.former_qualified_name}:"
                        f"{record.former_line}:{record.former_column} "
                        f"{record.former_route}"
                    ),
                )
            )
    return tuple(issues)


def _inspect_boundaries(
    inventory: AuditInventory,
    *,
    discovered: Sequence[DiscoveredRuntimeBoundary],
) -> tuple[BoundaryAuditIssue, ...]:
    issues: list[BoundaryAuditIssue] = []
    expected = {
        (boundary.module, boundary.callable_name): boundary
        for boundary in inventory.boundaries
    }
    actual = {
        (boundary.module, boundary.callable_name): boundary for boundary in discovered
    }
    for key in sorted(set(actual) - set(expected)):
        actual_candidate = actual[key]
        issues.append(
            BoundaryAuditIssue(
                boundary_id=f"{actual_candidate.module}:{actual_candidate.callable_name}",
                reason=f"discoverable {actual_candidate.kind.value} boundary is omitted from the immutable manifest",
            )
        )
    for key in sorted(set(expected) - set(actual)):
        expected_candidate = expected[key]
        issues.append(
            BoundaryAuditIssue(
                boundary_id=f"{expected_candidate.module}:{expected_candidate.callable_name}",
                reason="runtime boundary manifest entry is stale or its actual callable is not discoverable",
            )
        )
    for key in sorted(set(expected) & set(actual)):
        expected_boundary = expected[key]
        actual_boundary = actual[key]
        boundary_id = f"{expected_boundary.module}:{expected_boundary.callable_name}"
        if expected_boundary.kind is not actual_boundary.kind:
            issues.append(
                BoundaryAuditIssue(
                    boundary_id,
                    "boundary kind mismatch: "
                    f"manifest={expected_boundary.kind.value}, source={actual_boundary.kind.value}",
                )
            )
        if expected_boundary.executable_module != actual_boundary.executable_module:
            issues.append(
                BoundaryAuditIssue(
                    boundary_id,
                    "executable-module declaration mismatch: "
                    f"manifest={expected_boundary.executable_module}, "
                    f"source={actual_boundary.executable_module}",
                )
            )
        if expected_boundary.validator_key != actual_boundary.validator_key:
            issues.append(
                BoundaryAuditIssue(
                    boundary_id,
                    "validator key mismatch: "
                    f"manifest={expected_boundary.validator_key!r}, "
                    f"source={actual_boundary.validator_key!r}",
                )
            )
        if expected_boundary.validator_callable != actual_boundary.validator_callable:
            issues.append(
                BoundaryAuditIssue(
                    boundary_id,
                    "actual validator callable mismatch: "
                    f"manifest={expected_boundary.validator_callable!r}, "
                    f"source={actual_boundary.validator_callable!r}",
                )
            )
        if actual_boundary.kind is not BoundaryKind.VALIDATION_COMMAND and (
            actual_boundary.validator_key is None
            or actual_boundary.validator_callable is None
        ):
            issues.append(
                BoundaryAuditIssue(
                    boundary_id,
                    "runtime boundary has no actual pre-side-effect validator binding",
                )
            )
        if (
            actual_boundary.subprocess_invokers
            and not actual_boundary.executable_module
        ):
            issues.append(
                BoundaryAuditIssue(
                    boundary_id,
                    "python -m subprocess target has no executable module edge; "
                    "invoked by " + ", ".join(actual_boundary.subprocess_invokers),
                )
            )
    return tuple(sorted(issues))


def _migration_identifier(record: MigrationRecord) -> str:
    serialized = json.dumps(
        [
            record.former_module,
            record.former_qualified_name,
            record.former_line,
            record.former_column,
            record.former_route,
            record.category.value,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:20]


def _inspect_migrations(
    inventory: AuditInventory,
    *,
    current_routes: Counter[tuple[str, str, MigrationCategory, str]],
    current_symbols: set[str],
) -> tuple[MigrationAuditIssue, ...]:
    issues: list[MigrationAuditIssue] = []
    if inventory.rules != EXPECTED_AUDIT_RULES:
        issues.append(
            MigrationAuditIssue(
                record_id="<rules>",
                reason="audit category manifest differs from the immutable manifest",
            )
        )
    if inventory.exemptions != EXPECTED_AUDIT_EXEMPTIONS:
        issues.append(
            MigrationAuditIssue(
                record_id="<exemptions>",
                reason="audit exemption inventory differs from the immutable manifest",
            )
        )
    actual_record_ids = frozenset(record.record_id for record in inventory.migrations)
    if len(inventory.migrations) != EXPECTED_MIGRATION_RECORD_COUNT:
        issues.append(
            MigrationAuditIssue(
                record_id="<ledger>",
                reason=(
                    "migration record count does not match the immutable manifest: "
                    f"expected {EXPECTED_MIGRATION_RECORD_COUNT}, "
                    f"got {len(inventory.migrations)}"
                ),
            )
        )
    missing_record_ids = sorted(EXPECTED_MIGRATION_RECORD_IDS - actual_record_ids)
    unexpected_record_ids = sorted(actual_record_ids - EXPECTED_MIGRATION_RECORD_IDS)
    if missing_record_ids or unexpected_record_ids:
        details: list[str] = []
        if missing_record_ids:
            details.append("missing " + ", ".join(missing_record_ids))
        if unexpected_record_ids:
            details.append("unexpected " + ", ".join(unexpected_record_ids))
        issues.append(
            MigrationAuditIssue(
                record_id="<ledger>",
                reason="migration record identity set mismatch: " + "; ".join(details),
            )
        )
    actual_manifest_digest = migration_manifest_digest(inventory.migrations)
    if actual_manifest_digest != MIGRATION_LEDGER_SHA256:
        issues.append(
            MigrationAuditIssue(
                record_id="<ledger>",
                reason=(
                    "migration manifest digest mismatch: "
                    f"expected {MIGRATION_LEDGER_SHA256}, "
                    f"got {actual_manifest_digest}"
                ),
            )
        )
    if inventory.boundaries != EXPECTED_RUNTIME_BOUNDARIES:
        issues.append(
            MigrationAuditIssue(
                record_id="<boundaries>",
                reason="runtime boundary inventory differs from the immutable manifest",
            )
        )
    boundary_modules = {boundary.module for boundary in inventory.boundaries}
    exemption_keys = {_exemption_key(exemption) for exemption in inventory.exemptions}
    categories = {record.category for record in inventory.migrations}
    ledger_route_keys = {
        (
            record.former_module,
            record.former_qualified_name,
            record.category,
            record.former_route,
        )
        for record in inventory.migrations
    }
    for route_key in sorted(set(current_routes) - ledger_route_keys):
        module, qualified_name, category, route = route_key
        issues.append(
            MigrationAuditIssue(
                record_id="<unrecorded-route>",
                reason=(
                    "current source route has no LIVE/EXEMPTED inventory record: "
                    f"{module}.{qualified_name} [{category.value}] {route}"
                ),
            )
        )
    missing_categories = set(MigrationCategory) - categories
    for category in sorted(missing_categories, key=str):
        issues.append(
            MigrationAuditIssue(
                record_id="<ledger>",
                reason=f"missing migration category {category.value}",
            )
        )
    for boundary in inventory.boundaries:
        callable_symbol = f"{boundary.module}.{boundary.callable_name}"
        if callable_symbol not in current_symbols:
            issues.append(
                MigrationAuditIssue(
                    record_id=f"boundary:{boundary.module}",
                    reason=f"runtime boundary callable is stale: {callable_symbol}",
                )
            )
        if boundary.path_authority not in current_symbols:
            issues.append(
                MigrationAuditIssue(
                    record_id=f"boundary:{boundary.module}",
                    reason=f"path authority is stale: {boundary.path_authority}",
                )
            )
    for record in inventory.migrations:
        if record.record_id != _migration_identifier(record):
            issues.append(
                MigrationAuditIssue(
                    record.record_id,
                    "record ID does not match the exact former site",
                )
            )
        if record.canonical_symbol not in current_symbols:
            issues.append(
                MigrationAuditIssue(
                    record.record_id,
                    f"canonical symbol is stale: {record.canonical_symbol}",
                )
            )
        if (
            record.authority_kind is MigrationAuthorityKind.PATH_RESOLVER
            and record.canonical_symbol
            != "src.utils.configuration.paths.PathResolver.resolve"
        ):
            issues.append(
                MigrationAuditIssue(
                    record.record_id,
                    "path authority must be the shared PathResolver.resolve callable",
                )
            )
        if (
            record.status in {MigrationStatus.LIVE, MigrationStatus.EXEMPTED}
        ):
            expected_kind, expected_symbol, expected_field = _live_route_authority(
                record.former_module,
                record.former_qualified_name,
                record.category,
                record.former_route,
            )
            if record.authority_kind is not expected_kind or (
                record.canonical_symbol != expected_symbol
            ) or record.authority_field != expected_field:
                issues.append(
                    MigrationAuditIssue(
                        record.record_id,
                        "live route must name the authority it actually invokes: "
                        f"{expected_kind.value} {expected_symbol} "
                        f"field={expected_field!r}",
                    )
                )
        unknown_coverage = sorted(set(record.entrypoint_coverage) - boundary_modules)
        if unknown_coverage:
            issues.append(
                MigrationAuditIssue(
                    record.record_id,
                    "unknown entrypoint coverage: " + ", ".join(unknown_coverage),
                )
            )
        expected_coverage = migration_entrypoint_coverage(
            record.domain,
            inventory.boundaries,
        )
        if record.entrypoint_coverage != expected_coverage:
            issues.append(
                MigrationAuditIssue(
                    record.record_id,
                    "entrypoint coverage does not match the domain policy: "
                    f"expected {expected_coverage!r}, "
                    f"got {record.entrypoint_coverage!r}",
                )
            )
        route_key = (
            record.former_module,
            record.former_qualified_name,
            record.category,
            record.former_route,
        )
        actual = current_routes[route_key]
        if actual != record.expected_current_occurrences:
            issues.append(
                MigrationAuditIssue(
                    record.record_id,
                    "former route occurrence count is stale: "
                    f"expected {record.expected_current_occurrences}, got {actual}",
                )
            )
        if record.status is MigrationStatus.MIGRATED and actual != 0:
            issues.append(
                MigrationAuditIssue(
                    record.record_id,
                    f"MIGRATED route still has {actual} live occurrence(s)",
                )
            )
        if (
            record.status in {MigrationStatus.LIVE, MigrationStatus.EXEMPTED}
            and actual == 0
        ):
            issues.append(
                MigrationAuditIssue(
                    record.record_id,
                    f"{record.status.value.upper()} route has no current occurrence",
                )
            )
        exact_exemptions = {
            key
            for key in exemption_keys
            if key[:3]
            == (
                record.former_module,
                record.former_qualified_name,
                record.former_line,
            )
            and migration_route_audit_rule(record.former_route) is key[3]
        }
        if record.status is MigrationStatus.EXEMPTED and not exact_exemptions:
            issues.append(
                MigrationAuditIssue(
                    record.record_id,
                    "EXEMPTED route has no exact source exemption",
                )
            )
        if record.status is MigrationStatus.LIVE and exact_exemptions:
            issues.append(
                MigrationAuditIssue(
                    record.record_id,
                    "LIVE route is classified by an exact exemption and must be EXEMPTED",
                )
            )
    return tuple(sorted(issues))


def _render(findings: Iterable[AuditFinding]) -> str:
    return "\n".join(
        f"{finding.module}:{finding.line}:{finding.column}: "
        f"{finding.rule.value} ({finding.qualified_name})"
        for finding in findings
    )


def _render_migration(record: MigrationRecord) -> str:
    coverage = ",".join(record.entrypoint_coverage)
    return "\t".join(
        (
            record.record_id,
            record.category.value,
            record.status.value,
            record.domain,
            f"{record.former_module}:{record.former_line}:{record.former_column}",
            record.former_qualified_name,
            record.former_route,
            record.canonical_symbol,
            record.migration_target,
            coverage,
        )
    )


def _render_boundary(inventory: AuditInventory) -> str:
    return "\n".join(
        "\t".join(
            (
                boundary.domain,
                boundary.module,
                boundary.callable_name,
                boundary.kind.value,
                str(boundary.executable_module),
                boundary.validator_key or "<unbound>",
                boundary.validator_callable or "<unbound>",
                boundary.configuration_authority or "<unbound>",
                boundary.path_authority,
                boundary.migration_target,
                boundary.required_policy,
                boundary.optional_policy,
                boundary.default_authority,
                boundary.precedence_authority,
            )
        )
        for boundary in inventory.boundaries
    )


def _render_adapter_contracts() -> str:
    """Render every typed adapter field from the single source catalog."""
    from src.configuration_contracts import ADAPTER_CONTRACTS

    return "\n".join(
        "\t".join(
            (
                contract.adapter_symbol,
                field.path,
                "|".join(field.expected_types),
                "required" if field.required else "optional",
                field.default_policy.value,
                field.precedence_authority.value,
            )
        )
        for contract in ADAPTER_CONTRACTS
        for field in contract.inspect()
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the source audit, returning nonzero for unclassified findings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument(
        "--show-ledger",
        action="store_true",
        help="render every exact migration record as tab-separated data",
    )
    parser.add_argument(
        "--show-contracts",
        action="store_true",
        help="render required/optional/default/precedence authorities",
    )
    parser.add_argument(
        "--show-discovered-boundaries",
        action="store_true",
        help="render source-discovered callable, executable, and validator bindings",
    )
    parser.add_argument(
        "--regenerate-ledger",
        action="store_true",
        help="emit deterministic candidate ledger JSON without changing embedded data",
    )
    parser.add_argument(
        "--write-generated-data",
        action="store_true",
        help=(
            "freeze re-anchored exemptions and ledger metadata in migration_data.py "
            "and exemption_data.py; fails on every newly unclassified construct"
        ),
    )
    parser.add_argument(
        "--source-revision",
        help="immutable revision label required by --write-generated-data",
    )
    arguments = parser.parse_args(argv)
    if arguments.write_generated_data:
        if arguments.source_revision is None:
            parser.error("--write-generated-data requires --source-revision")
        try:
            migration_count, exemption_count = write_generated_inventory_data(
                arguments.source_root.resolve(),
                source_revision=arguments.source_revision,
            )
        except RuntimeError as error:
            print(error)
            return 1
        print(
            "Generated strict inventory data: "
            f"{migration_count} migration rows, {exemption_count} exemptions."
        )
        return 0
    report = inspect_source(arguments.source_root.resolve())
    if arguments.show_ledger:
        for record in DEFAULT_AUDIT_INVENTORY.migrations:
            print(_render_migration(record))
    if arguments.show_contracts:
        print(_render_boundary(DEFAULT_AUDIT_INVENTORY))
        print(_render_adapter_contracts())
    if arguments.show_discovered_boundaries:
        for boundary in report.discovered_boundaries:
            print(
                "\t".join(
                    (
                        boundary.module,
                        boundary.callable_name,
                        boundary.kind.value,
                        str(boundary.executable_module),
                        boundary.validator_key or "<unbound>",
                        boundary.validator_callable or "<unbound>",
                        ",".join(boundary.subprocess_invokers),
                    )
                )
            )
    if arguments.regenerate_ledger:
        print(
            json.dumps(
                regenerate_migration_rows(arguments.source_root.resolve()),
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
    if report.unclassified:
        print(_render(report.unclassified))
        print(f"Unclassified findings: {len(report.unclassified)}")
    if report.stale_exemptions:
        for exemption in report.stale_exemptions:
            print(
                f"stale exemption: {exemption.module}:{exemption.line}: "
                f"{exemption.rule.value} ({exemption.qualified_name})"
            )
    if report.migration_issues:
        for issue in report.migration_issues:
            print(f"migration ledger: {issue.record_id}: {issue.reason}")
    if report.boundary_issues:
        for boundary_issue in report.boundary_issues:
            print(
                f"runtime boundary: {boundary_issue.boundary_id}: "
                f"{boundary_issue.reason}"
            )
    if not report.passed:
        return 1
    print(
        "Strict configuration/path audit passed with no unclassified findings; "
        f"migration ledger exact ({len(DEFAULT_AUDIT_INVENTORY.migrations)} records)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - executable audit boundary
    raise SystemExit(main())
