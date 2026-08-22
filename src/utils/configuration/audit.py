"""Current-source AST audit for strict configuration and path contracts."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import src.utils.configuration.discovery as configuration_discovery
from src.utils.configuration.catalog import ADAPTER_CONTRACTS
from src.utils.configuration.inventory import (
    DEFAULT_AUDIT_INVENTORY,
    AuditInventory,
    AuditRule,
)

__all__ = [
    "AuditFinding",
    "AuditOptions",
    "AuditReport",
    "BoundaryAuditIssue",
    "audit_source",
    "inspect_source",
    "run_configuration_audit",
]


@dataclass(frozen=True, slots=True, order=True)
class AuditFinding:
    """One prohibited construct detected in current source."""

    module: str
    qualified_name: str
    line: int
    column: int
    rule: AuditRule


@dataclass(frozen=True, slots=True)
class AuditOptions:
    """Presentation choices for one read-only audit run."""

    show_contracts: bool = False
    show_discovered_boundaries: bool = False


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Current-source findings and runtime-boundary violations."""

    findings: tuple[AuditFinding, ...]
    boundary_issues: tuple[BoundaryAuditIssue, ...] = field(default_factory=tuple)
    discovered_boundaries: tuple[
        configuration_discovery.DiscoveredRuntimeBoundary, ...
    ] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """Whether current source satisfies every audit contract."""

        return not self.findings and not self.boundary_issues


@dataclass(frozen=True, slots=True, order=True)
class BoundaryAuditIssue:
    """One omitted, stale, unbound, or silently non-executable boundary."""

    boundary_id: str
    reason: str


class _Visitor(ast.NodeVisitor):
    def __init__(self, module: str, tree: ast.Module) -> None:
        self.module = module
        self.scope: list[str] = []
        module_mappings = _semantic_mapping_names(
            tree.body,
            (),
            callable_name="",
        )
        self.mapping_names: list[set[str]] = [module_mappings]
        self.configuration_value_names: list[set[str]] = [
            _semantic_configuration_value_names(tree.body, module_mappings)
        ]
        self.findings: list[AuditFinding] = []
        self._parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }

    @property
    def qualified_name(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

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

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        class_mappings = _class_mapping_attributes(node)
        mappings = self.mapping_names[-1] | class_mappings
        self.mapping_names.append(mappings)
        self.configuration_value_names.append(
            _semantic_configuration_value_names(
                node.body,
                mappings,
                initial_values=self.configuration_value_names[-1],
            )
        )
        self.generic_visit(node)
        self.configuration_value_names.pop()
        self.mapping_names.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        mappings = _semantic_mapping_names(
            node.body,
            _function_arguments(node),
            callable_name=node.name,
            initial_mappings=self.mapping_names[-1],
        )
        self.mapping_names.append(mappings)
        self.configuration_value_names.append(
            _semantic_configuration_value_names(
                node.body,
                mappings,
                initial_values=self.configuration_value_names[-1],
            )
        )
        self.generic_visit(node)
        self.configuration_value_names.pop()
        self.mapping_names.pop()
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        mappings = _semantic_mapping_names(
            node.body,
            _function_arguments(node),
            callable_name=node.name,
            initial_mappings=self.mapping_names[-1],
        )
        self.mapping_names.append(mappings)
        self.configuration_value_names.append(
            _semantic_configuration_value_names(
                node.body,
                mappings,
                initial_values=self.configuration_value_names[-1],
            )
        )
        self.generic_visit(node)
        self.configuration_value_names.pop()
        self.mapping_names.pop()
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            configured_receiver = _is_configuration_route(
                node.func.value,
                self.mapping_names[-1],
            )
            if configured_receiver and node.func.attr == "get" and len(node.args) >= 2:
                self._record(node, AuditRule.GET_WITH_FALLBACK)
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
            if configured_receiver and node.func.attr == "setdefault":
                self._record(node, AuditRule.SETDEFAULT)
            if node.func.attr == "to_absolute_path":
                self._record(node, AuditRule.HYDRA_ABSOLUTE_PATH)
            if node.func.attr in {"cwd", "getcwd"} and (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"Path", "os"}
                or isinstance(node.func.value, ast.Attribute)
                and ast.unparse(node.func.value) == "os.path"
            ):
                self._record(node, AuditRule.PROCESS_CWD)
        if isinstance(node.func, ast.Name) and node.func.id == "to_absolute_path":
            self._record(node, AuditRule.HYDRA_ABSOLUTE_PATH)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 3
            and _is_configuration_route(node.args[0], self.mapping_names[-1])
        ):
            self._record(node, AuditRule.GETATTR_WITH_FALLBACK)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "Path"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and _looks_like_runtime_path_literal(node.args[0].value)
        ):
            self._record(node.args[0], AuditRule.RUNTIME_PATH_LITERAL)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "Path"
            and node.args
            and _contains_configuration_value(
                node.args[0],
                self.mapping_names[-1],
                self.configuration_value_names[-1],
            )
        ):
            self._record(node, AuditRule.RAW_PATH_CONSTRUCTION)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if (
            isinstance(node.op, ast.Or)
            and not _is_predicate_expression(node, self._parents)
            and not any(_is_explicit_boolean_value(value) for value in node.values)
            and _contains_configuration_value(
                node,
                self.mapping_names[-1],
                self.configuration_value_names[-1],
            )
        ):
            self._record(node, AuditRule.NULL_COALESCING)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            self.module != "src.utils.paths"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and _contains_dunder_file(node.value.value)
        ):
            self._record(node, AuditRule.FILE_PARENT_INDEX)
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
    return "dictconfig" in rendered or bool(
        re.search(r"(^|[^a-z])(dict|mapping)([^a-z]|$)", rendered)
    )


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
    name = configuration_discovery.ast_call_name(node).lower()
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
        return _mapping_annotation(node.args[0])
    return False


def _semantic_mapping_names(
    body: Sequence[ast.stmt],
    arguments: Sequence[ast.arg],
    *,
    callable_name: str,
    initial_mappings: set[str] | None = None,
) -> set[str]:
    """Infer raw configuration mappings without tainting derived scalar values."""

    mapping_names = set() if initial_mappings is None else set(initial_mappings)
    mapping_names.update({
        argument.arg
        for argument in arguments
        if _mapping_annotation(argument.annotation)
        and (
            "dictconfig" in _annotation_name(argument.annotation)
            or any(token in callable_name for token in ("config", "boundary"))
        )
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
            if _mapping_annotation(statement.annotation) and any(
                token in callable_name for token in ("config", "boundary")
            ):
                mapping_names.update(assigned)
        if (
            isinstance(statement, (ast.For, ast.AsyncFor))
            and _contains_configuration_route(statement.iter, mapping_names)
        ):
            mapping_names.update(_bound_names(statement.target))
    changed = True
    while changed:
        changed = False
        for statement in scoped_nodes:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            value = statement.value
            if value is None:
                continue
            if ast.unparse(value) not in mapping_names and not _call_returns_mapping(
                value, mapping_names
            ):
                continue
            before = len(mapping_names)
            mapping_names.update(_assigned_names(statement))
            changed |= len(mapping_names) != before
    return mapping_names


def _semantic_configuration_value_names(
    body: Sequence[ast.stmt],
    mapping_names: set[str],
    *,
    initial_values: set[str] | None = None,
) -> set[str]:
    """Infer direct scalar aliases sourced from raw configuration mappings."""

    value_names = set() if initial_values is None else set(initial_values)
    assignments = tuple(
        statement
        for statement in _scope_nodes(body)
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        and statement.value is not None
    )
    changed = True
    while changed:
        changed = False
        for statement in assignments:
            value = statement.value
            if value is None:
                continue
            if not _is_raw_configuration_value(
                value,
                mapping_names,
                value_names,
            ):
                continue
            before = len(value_names)
            value_names.update(_assigned_names(statement))
            changed |= len(value_names) != before
    return value_names


def _is_raw_configuration_value(
    node: ast.AST,
    mapping_names: set[str],
    value_names: set[str],
) -> bool:
    rendered = ast.unparse(node)
    if rendered in value_names:
        return True
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _is_configuration_route(node, mapping_names)
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"get", "__getitem__"}
        and _is_configuration_route(node.func.value, mapping_names)
    )


def _class_mapping_attributes(node: ast.ClassDef) -> set[str]:
    """Infer raw configuration mappings retained on an instance."""

    declared_mappings = {
        f"self.{statement.target.id}"
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and _mapping_annotation(statement.annotation)
        and (
            statement.target.id.lower() in {"cfg", "config"}
            or statement.target.id.lower().endswith(("_cfg", "_config"))
        )
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
        return declared_mappings
    mappings = _semantic_mapping_names(
        constructor.body,
        _function_arguments(constructor),
        callable_name=constructor.name,
    )
    return declared_mappings | {
        name for name in mappings if name.startswith("self.")
    }


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


def _contains_configuration_value(
    node: ast.AST,
    mapping_names: set[str],
    value_names: set[str],
) -> bool:
    return _contains_configuration_route(node, mapping_names) or any(
        isinstance(child, (ast.Name, ast.Attribute))
        and ast.unparse(child) in value_names
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
        and configuration_discovery.ast_call_name(node)
        in {"all", "any", "bool", "isinstance", "issubclass"}
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


def audit_source(
    source_root: Path,
    *,
    inventory: AuditInventory = DEFAULT_AUDIT_INVENTORY,
) -> tuple[AuditFinding, ...]:
    """Return prohibited findings under source_root in stable order."""

    return inspect_source(source_root, inventory=inventory).findings


def inspect_source(
    source_root: Path,
    *,
    inventory: AuditInventory = DEFAULT_AUDIT_INVENTORY,
) -> AuditReport:
    """Inspect current source without a checked-in source snapshot."""

    findings: list[AuditFinding] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _Visitor(
            configuration_discovery.source_module_name(source_root, path), tree
        )
        visitor.visit(tree)
        findings.extend(visitor.findings)
    discovered_boundaries = configuration_discovery.discover_runtime_boundaries(
        source_root
    )
    return AuditReport(
        findings=tuple(sorted(set(findings))),
        boundary_issues=_inspect_boundaries(
            inventory,
            discovered=discovered_boundaries,
        ),
        discovered_boundaries=discovered_boundaries,
    )


def _inspect_boundaries(
    inventory: AuditInventory,
    *,
    discovered: Sequence[configuration_discovery.DiscoveredRuntimeBoundary],
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
                reason=f"discoverable {actual_candidate.kind.value} boundary is omitted from the explicit inventory",
            )
        )
    for key in sorted(set(expected) - set(actual)):
        expected_candidate = expected[key]
        issues.append(
            BoundaryAuditIssue(
                boundary_id=f"{expected_candidate.module}:{expected_candidate.callable_name}",
                reason="runtime boundary inventory entry is stale or its actual callable is not discoverable",
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
                    f"inventory={expected_boundary.kind.value}, source={actual_boundary.kind.value}",
                )
            )
        if expected_boundary.executable_module != actual_boundary.executable_module:
            issues.append(
                BoundaryAuditIssue(
                    boundary_id,
                    "executable-module declaration mismatch: "
                    f"inventory={expected_boundary.executable_module}, "
                    f"source={actual_boundary.executable_module}",
                )
            )
        if expected_boundary.validator_key != actual_boundary.validator_key:
            issues.append(
                BoundaryAuditIssue(
                    boundary_id,
                    "validator key mismatch: "
                    f"inventory={expected_boundary.validator_key!r}, "
                    f"source={actual_boundary.validator_key!r}",
                )
            )
        if expected_boundary.validator_callable != actual_boundary.validator_callable:
            issues.append(
                BoundaryAuditIssue(
                    boundary_id,
                    "actual validator callable mismatch: "
                    f"inventory={expected_boundary.validator_callable!r}, "
                    f"source={actual_boundary.validator_callable!r}",
                )
            )
        if (
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


def _render(findings: Iterable[AuditFinding]) -> str:
    return "\n".join(
        f"{finding.module}:{finding.line}:{finding.column}: "
        f"{finding.rule.value} ({finding.qualified_name})"
        for finding in findings
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
                boundary.validation_target,
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


def run_configuration_audit(source_root: Path, options: AuditOptions) -> int:
    """Run the read-only audit and return nonzero for any violation."""

    report = inspect_source(source_root.resolve())
    if options.show_contracts:
        print(_render_boundary(DEFAULT_AUDIT_INVENTORY))
        print(_render_adapter_contracts())
    if options.show_discovered_boundaries:
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
    if report.findings:
        print(_render(report.findings))
        print(f"Prohibited findings: {len(report.findings)}")
    if report.boundary_issues:
        for issue in report.boundary_issues:
            print(f"runtime boundary: {issue.boundary_id}: {issue.reason}")
    if not report.passed:
        return 1
    print(
        "Strict configuration/path audit passed against current source; "
        f"{len(report.discovered_boundaries)} runtime boundaries verified."
    )
    return 0
