"""Source-complete inspectable contracts for configuration authorities."""

from __future__ import annotations

import ast
import importlib
import inspect
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast, get_args, get_type_hints

from src.utils.configuration.schema import (
    ConfigFieldContract,
    ConfigurationAbsencePolicy,
    ConfigurationDefaultPolicy,
    ConfigurationPrecedence,
    StrictConfigSchema,
)


@dataclass(frozen=True, slots=True)
class TypedAdapterContract:
    """Stable field contract for one dataclass or strict-schema authority."""

    name: str
    adapter_symbol: str
    authority_kind: str
    fields: tuple[ConfigFieldContract, ...]
    semantic_constraints: tuple[str, ...]

    def inspect(self) -> tuple[ConfigFieldContract, ...]:
        """Return the complete field declaration."""
        return self.fields


@dataclass(frozen=True, slots=True)
class ContractDeclarations:
    """Raw source declarations used as the catalog completeness oracle."""

    adapter_symbols: frozenset[str]
    schema_symbols: frozenset[str]
    path_boundary_symbols: frozenset[str]

    @property
    def all_symbols(self) -> frozenset[str]:
        """Return every declaration which must appear in the runtime catalog."""
        return (
            self.adapter_symbols
            | self.schema_symbols
            | self.path_boundary_symbols
        )


@dataclass(frozen=True, slots=True)
class RuntimeBoundaryReference:
    """Minimal source identity needed to bind one runtime boundary."""

    boundary_id: str
    module: str
    callable_name: str
    validator_key: str | None
    validator_callable: str | None


@dataclass(frozen=True, slots=True)
class BoundaryAuthorityBinding:
    """Source-reachable contract and semantic authorities for one boundary."""

    authority_symbols: tuple[str, ...]
    semantic_authorities: tuple[str, ...]
    path_role_authorities: tuple[str, ...]


def _decorator_name(decorator: ast.expr) -> str:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _assigned_names(statement: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


def _schema_constructor_name(value: ast.expr | None) -> str:
    if not isinstance(value, ast.Call):
        return ""
    if isinstance(value.func, ast.Name):
        return value.func.id
    if isinstance(value.func, ast.Attribute):
        return value.func.attr
    return ""


def _module_name(source_root: Path, path: Path) -> str:
    return ".".join(path.relative_to(source_root.parent).with_suffix("").parts)


def discover_contract_declarations(source_root: Path) -> ContractDeclarations:
    """Discover runtime adapter/schema declarations directly from raw source.

    Dataclass adapters are either owned by a conventional configuration module
    or explicitly named as a ``*Config``/``*Configuration``/``*Paths`` runtime
    type. Strict schemas are discovered by their module-level ``*_SCHEMA``
    declaration, independent of the source audit and runtime catalog.
    """
    if not source_root.is_absolute() or source_root.name != "src":
        raise ValueError("configuration contract discovery requires an absolute src root")
    adapters: set[str] = set()
    schemas: set[str] = set()
    path_boundaries: set[str] = set()
    for path in sorted(source_root.rglob("*.py")):
        module = _module_name(source_root, path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(source_root).as_posix()
        adapter_module = (
            path.name in {"configuration.py", "config.py", "operations.py"}
            or relative == "utils/schema/court.py"
        )
        for statement in tree.body:
            if isinstance(statement, ast.ClassDef) and any(
                _decorator_name(decorator) == "dataclass"
                for decorator in statement.decorator_list
            ):
                has_field_default = any(
                    isinstance(member, ast.AnnAssign) and member.value is not None
                    for member in statement.body
                )
                explicitly_named_adapter = statement.name.endswith(
                    ("Config", "Configuration")
                ) or (
                    statement.name.endswith("Paths")
                    and not module.startswith("src.utils.configuration")
                )
                if adapter_module or (
                    explicitly_named_adapter and not has_field_default
                ):
                    adapters.add(f"{module}.{statement.name}")
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                constructor = _schema_constructor_name(statement.value)
                for name in _assigned_names(statement):
                    if constructor in {
                        "StrictConfigSchema",
                        "_schema",
                        "_mapping_schema",
                        "_numeric_mapping",
                    } and (name == "SCHEMA" or name.endswith("_SCHEMA")):
                        schemas.add(f"{module}.{name}")
                    if constructor == "NonHydraPathBoundary":
                        path_boundaries.add(f"{module}.{name}")
    return ContractDeclarations(
        frozenset(adapters),
        frozenset(schemas),
        frozenset(path_boundaries),
    )


def inspect_typed_adapter(adapter: type[object]) -> TypedAdapterContract:
    """Build an inspectable contract from a no-runtime-default dataclass."""
    if not is_dataclass(adapter):
        raise TypeError(f"Typed configuration adapter must be a dataclass: {adapter!r}")
    hints = get_type_hints(adapter)
    contracts: list[ConfigFieldContract] = []
    for specification in fields(adapter):
        if specification.default is not MISSING or specification.default_factory is not MISSING:
            raise ValueError(
                f"{adapter.__module__}.{adapter.__qualname__}.{specification.name} "
                "declares a Python runtime default; defaults must be composition-owned."
            )
        annotation = hints[specification.name]
        required = specification.metadata.get("configuration_required", True)
        if type(required) is not bool:
            raise TypeError(
                f"{adapter.__qualname__}.{specification.name} configuration_required "
                "metadata must be exactly bool."
            )
        raw_absence = specification.metadata.get(
            "configuration_absence_policy",
            (
                ConfigurationAbsencePolicy.REQUIRED.value
                if required
                else ConfigurationAbsencePolicy.OPTIONAL_OMITTED.value
            ),
        )
        try:
            absence_policy = ConfigurationAbsencePolicy(raw_absence)
        except ValueError as error:
            raise ValueError(
                f"{adapter.__qualname__}.{specification.name} has unsupported "
                f"absence policy {raw_absence!r}."
            ) from error
        if required and absence_policy is not ConfigurationAbsencePolicy.REQUIRED:
            raise ValueError(
                f"{adapter.__qualname__}.{specification.name} is required but "
                f"declares {absence_policy.value!r}."
            )
        if not required and absence_policy is ConfigurationAbsencePolicy.REQUIRED:
            raise ValueError(
                f"{adapter.__qualname__}.{specification.name} is optional but "
                "declares the required absence policy."
            )
        contracts.append(
            ConfigFieldContract(
                path=f"{adapter.__qualname__}.{specification.name}",
                expected_types=(str(annotation),),
                required=required,
                absence_policy=absence_policy,
                value_constraints=(
                    "exact-declared-type",
                    "required-constructor-input"
                    if required
                    else "optional-execution-input",
                ),
                default_policy=ConfigurationDefaultPolicy.COMPOSITION_OWNED,
                precedence_authority=ConfigurationPrecedence.COMPOSED_VALUE_ONLY,
            )
        )
    symbol = f"{adapter.__module__}.{adapter.__qualname__}"
    return TypedAdapterContract(
        name=adapter.__qualname__,
        adapter_symbol=symbol,
        authority_kind="typed-dataclass",
        fields=tuple(contracts),
        semantic_constraints=tuple(
            f"{symbol}.{name}"
            for name in ("__post_init__", "from_config", "from_mapping", "from_json")
            if name in adapter.__dict__
        ),
    )


def _schema_contract(symbol: str, schema: StrictConfigSchema) -> TypedAdapterContract:
    return TypedAdapterContract(
        name=schema.name,
        adapter_symbol=symbol,
        authority_kind="strict-schema",
        fields=schema.inspect(),
        semantic_constraints=tuple(
            f"{check.__module__}.{check.__qualname__}"
            for check in schema.semantic_checks
        ),
    )


def _literal_text(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _enum_member(node: ast.AST | None, enum_name: str) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == enum_name
    ):
        return node.attr.lower()
    return None


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == name),
        None,
    )


def _path_boundary_contract(
    symbol: str,
    call: ast.Call,
) -> TypedAdapterContract:
    """Inspect a module-level ``NonHydraPathBoundary`` without importing it."""
    name_node = call.args[0] if call.args else _keyword(call, "name")
    boundary_name = _literal_text(name_node)
    if boundary_name is None or not boundary_name:
        raise RuntimeError(f"Path boundary has no literal name: {symbol}")
    fields_node = call.args[1] if len(call.args) >= 2 else _keyword(call, "fields")
    if not isinstance(fields_node, (ast.Tuple, ast.List)):
        raise RuntimeError(f"Path boundary fields are not source-inspectable: {symbol}")

    contracts: list[ConfigFieldContract] = []
    for node in fields_node.elts:
        if not isinstance(node, ast.Call) or _schema_constructor_name(node) != "BoundaryPathField":
            raise RuntimeError(f"Path boundary field is not explicit: {symbol}")
        field_name = _literal_text(node.args[0] if node.args else _keyword(node, "name"))
        role = _enum_member(
            node.args[1] if len(node.args) >= 2 else _keyword(node, "role"),
            "PathRole",
        )
        direction = _enum_member(
            node.args[2] if len(node.args) >= 3 else _keyword(node, "direction"),
            "PathDirection",
        )
        kind = _enum_member(
            node.args[3] if len(node.args) >= 4 else _keyword(node, "kind"),
            "PathKind",
        )
        if field_name is None or role is None or direction is None or kind is None:
            raise RuntimeError(f"Path boundary field metadata is incomplete: {symbol}")
        many_node = _keyword(node, "many")
        many = isinstance(many_node, ast.Constant) and many_node.value is True
        must_exist_node = _keyword(node, "must_exist")
        must_exist = (
            isinstance(must_exist_node, ast.Constant)
            and must_exist_node.value is True
        )
        contracts.append(
            ConfigFieldContract(
                path=f"{boundary_name}.{field_name}",
                expected_types=("sequence[str]" if many else "str",),
                required=True,
                absence_policy=ConfigurationAbsencePolicy.REQUIRED,
                value_constraints=(
                    "exact-runtime-type",
                    "required-key",
                    f"path-role:{role}",
                    f"path-direction:{direction}",
                    f"path-kind:{kind}",
                    *(('must-exist-before-side-effects',) if must_exist else ()),
                    *(('one-or-more-values',) if many else ()),
                ),
                default_policy=ConfigurationDefaultPolicy.COMPOSITION_OWNED,
                precedence_authority=ConfigurationPrecedence.COMPOSED_VALUE_ONLY,
            )
        )
    return TypedAdapterContract(
        name=boundary_name,
        adapter_symbol=symbol,
        authority_kind="non-hydra-path-boundary",
        fields=tuple(contracts),
        semantic_constraints=(
            "src.utils.configuration.paths.NonHydraPathBoundary.validate",
        ),
    )


def _path_boundary_contracts(
    source_root: Path,
    symbols: frozenset[str],
) -> tuple[TypedAdapterContract, ...]:
    by_symbol: dict[str, TypedAdapterContract] = {}
    for path in sorted(source_root.rglob("*.py")):
        module = _module_name(source_root, path)
        if not any(symbol.startswith(f"{module}.") for symbol in symbols):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for statement in tree.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            if not isinstance(statement.value, ast.Call):
                continue
            if _schema_constructor_name(statement.value) != "NonHydraPathBoundary":
                continue
            for name in _assigned_names(statement):
                symbol = f"{module}.{name}"
                if symbol in symbols:
                    by_symbol[symbol] = _path_boundary_contract(
                        symbol,
                        statement.value,
                    )
    missing = symbols - set(by_symbol)
    if missing:
        raise RuntimeError(
            "Discovered path boundaries disappeared during inspection: "
            + ", ".join(sorted(missing))
        )
    return tuple(by_symbol[symbol] for symbol in sorted(by_symbol))


def _import_symbol(symbol: str, modules: dict[str, ModuleType]) -> object:
    module_name, _, attribute = symbol.rpartition(".")
    module = modules.setdefault(module_name, importlib.import_module(module_name))
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise RuntimeError(f"Discovered configuration authority is absent: {symbol}") from error


def _nested_dataclass_types(annotation: object) -> Iterable[type[object]]:
    if inspect.isclass(annotation) and is_dataclass(annotation):
        yield annotation
        return
    for candidate in get_args(annotation):
        if inspect.isclass(candidate) and is_dataclass(candidate):
            yield candidate
        else:
            yield from _nested_dataclass_types(candidate)


@dataclass(frozen=True, slots=True)
class _SourceDefinition:
    module: str
    symbol: str
    node: ast.AST


@dataclass(frozen=True, slots=True)
class _SourceIndex:
    definitions: Mapping[str, _SourceDefinition]
    imports: Mapping[str, Mapping[str, str]]
    registries: Mapping[tuple[str, str], str]
    registry_symbols: frozenset[str]


def _import_bindings(tree: ast.Module, module: str) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom):
            imported_module = statement.module or ""
            if statement.level:
                package = (
                    module.removesuffix(".__init__")
                    if module.endswith(".__init__")
                    else module.rpartition(".")[0]
                )
                package_parts = package.split(".")
                parent_parts = package_parts[: len(package_parts) - statement.level + 1]
                imported_module = ".".join(
                    (*parent_parts, *(imported_module.split(".") if imported_module else ()))
                )
            for alias in statement.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = (
                        f"{imported_module}.{alias.name}"
                    )
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name
    return bindings


def _resolved_reference(
    module: str,
    imports: Mapping[str, str],
    node: ast.AST,
) -> str | None:
    if isinstance(node, ast.Name):
        return imports.get(node.id, f"{module}.{node.id}")
    if isinstance(node, ast.Attribute):
        base = _resolved_reference(module, imports, node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _canonical_source_reference(
    reference: str,
    imports_by_module: Mapping[str, Mapping[str, str]],
) -> str:
    """Follow source-level package re-exports to the defining symbol."""
    resolved = reference
    visited: set[str] = set()
    while resolved not in visited:
        visited.add(resolved)
        candidates = tuple(
            module
            for module in imports_by_module
            if resolved.startswith(f"{module}.")
        )
        if not candidates:
            break
        module = max(candidates, key=len)
        remainder = resolved.removeprefix(f"{module}.")
        name, separator, suffix = remainder.partition(".")
        imported = imports_by_module[module].get(name)
        if imported is None:
            break
        resolved = imported + (f".{suffix}" if separator else "")
    return resolved


def _source_index(source_root: Path) -> _SourceIndex:
    definitions: dict[str, _SourceDefinition] = {}
    imports_by_module: dict[str, Mapping[str, str]] = {}
    registries: dict[tuple[str, str], str] = {}
    registry_symbols: set[str] = set()
    for path in sorted(source_root.rglob("*.py")):
        module = _module_name(source_root, path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = _import_bindings(tree, module)
        imports_by_module[module] = imports
        if module.endswith(".__init__"):
            imports_by_module[module.removesuffix(".__init__")] = imports
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = f"{module}.{statement.name}"
                definitions[symbol] = _SourceDefinition(module, symbol, statement)
                continue
            if isinstance(statement, ast.ClassDef):
                class_symbol = f"{module}.{statement.name}"
                definitions[class_symbol] = _SourceDefinition(
                    module,
                    class_symbol,
                    statement,
                )
                for child in statement.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_symbol = f"{class_symbol}.{child.name}"
                        definitions[method_symbol] = _SourceDefinition(
                            module,
                            method_symbol,
                            child,
                        )
                continue
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            value = statement.value
            if value is None:
                continue
            for name in _assigned_names(statement):
                symbol = f"{module}.{name}"
                definitions[symbol] = _SourceDefinition(module, symbol, value)
            if not isinstance(value, ast.Dict):
                continue
            found_registry_entry = False
            for key_node, value_node in zip(value.keys, value.values, strict=True):
                key = _literal_text(key_node)
                target = _resolved_reference(module, imports, value_node)
                if key is not None and target is not None:
                    registries[(module, key)] = target
                    found_registry_entry = True
            if found_registry_entry:
                registry_symbols.update(
                    f"{module}.{name}" for name in _assigned_names(statement)
                )
    return _SourceIndex(
        definitions,
        imports_by_module,
        registries,
        frozenset(registry_symbols),
    )


def _callable_symbol(expression: str) -> tuple[str, tuple[str, ...]]:
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return expression, ()
    arguments: tuple[str, ...] = ()
    if isinstance(node, ast.Call):
        arguments = tuple(
            value
            for argument in node.args
            if (value := _literal_text(argument)) is not None
        )
        node = node.func
    return ast.unparse(node), arguments


def _definition_references(
    definition: _SourceDefinition,
    index: _SourceIndex,
) -> tuple[str, ...]:
    imports = index.imports[definition.module]
    references: set[str] = set()
    global_names = {
        symbol.removeprefix(f"{definition.module}.").split(".", maxsplit=1)[0]
        for symbol in index.definitions
        if symbol.startswith(f"{definition.module}.")
    }
    for node in ast.walk(definition.node):
        if isinstance(node, ast.Name) and (
            node.id in imports or node.id in global_names
        ):
            resolved = _resolved_reference(definition.module, imports, node)
            if resolved is not None:
                references.add(
                    _canonical_source_reference(resolved, index.imports)
                )
        elif isinstance(node, ast.Attribute):
            root: ast.expr = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and (
                root.id in imports or root.id in global_names
            ):
                resolved = _resolved_reference(definition.module, imports, node)
                if resolved is not None:
                    references.add(
                        _canonical_source_reference(resolved, index.imports)
                    )
    return tuple(sorted(references))


_PATH_ROLE_METHODS = frozenset(
    {
        "artifact",
        "cache",
        "checkpoint",
        "data",
        "external_asset",
        "output",
        "project",
    }
)


def _path_role_references(definition: _SourceDefinition) -> tuple[str, ...]:
    authorities: set[str] = set()
    for node in ast.walk(definition.node):
        if not isinstance(node, ast.Call):
            continue
        role: str | None = None
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in _PATH_ROLE_METHODS:
                role = node.func.attr
            elif node.func.attr in {"resolve", "validate"} and node.args:
                role = _enum_member(node.args[0], "PathRole")
        if role is None:
            role = next(
                (
                    candidate
                    for argument in node.args
                    if (candidate := _enum_member(argument, "PathRole")) is not None
                ),
                None,
            )
        if role is not None:
            authorities.add(
                f"{definition.symbol}:{node.lineno}:path-role:{role}:"
                f"{ast.unparse(node)}"
            )
    return tuple(sorted(authorities))


def discover_boundary_authorities(
    source_root: Path,
    boundaries: Sequence[RuntimeBoundaryReference],
    contracts: Sequence[TypedAdapterContract],
) -> Mapping[str, BoundaryAuthorityBinding]:
    """Bind each boundary only to authorities reachable from its real validator.

    The traversal is source-only.  It starts from both the configured validator
    and the executable callable, follows imported/local definitions, and resolves
    literal validator-key schema registries.  This avoids domain-wide overclaim
    while still covering adapters reached through helper functions.
    """
    index = _source_index(source_root)
    contract_symbols = {contract.adapter_symbol for contract in contracts}
    bindings: dict[str, BoundaryAuthorityBinding] = {}
    for boundary in boundaries:
        pending: deque[str] = deque()
        literal_arguments: tuple[str, ...] = ()
        local_path_contracts = tuple(
            symbol
            for symbol in contract_symbols
            if symbol.startswith(f"{boundary.module}.")
            and next(
                contract.authority_kind
                for contract in contracts
                if contract.adapter_symbol == symbol
            )
            == "non-hydra-path-boundary"
        )
        if boundary.validator_callable is not None:
            validator_symbol, literal_arguments = _callable_symbol(
                boundary.validator_callable
            )
            pending.append(validator_symbol)
            if (
                boundary.validator_callable.endswith(
                    "NonHydraPathBoundary.validate"
                )
                and not local_path_contracts
            ):
                pending.append(f"{boundary.module}.{boundary.callable_name}")
        else:
            pending.append(f"{boundary.module}.{boundary.callable_name}")
        pending.extend(local_path_contracts)
        if boundary.validator_key is not None:
            registered = index.registries.get(
                (
                    (
                        boundary.validator_callable.rpartition(".")[0]
                        if boundary.validator_callable is not None
                        else boundary.module
                    ).split("(", maxsplit=1)[0],
                    boundary.validator_key,
                )
            )
            if registered is None:
                registered = next(
                    (
                        target
                        for (module, key), target in index.registries.items()
                        if key == boundary.validator_key
                        and (
                            boundary.validator_callable is None
                            or boundary.validator_callable.startswith(f"{module}.")
                        )
                    ),
                    None,
                )
            if registered is not None:
                pending.append(registered)
        for argument in literal_arguments:
            registered = next(
                (
                    target
                    for (module, key), target in index.registries.items()
                    if key == argument
                    and boundary.validator_callable is not None
                    and boundary.validator_callable.startswith(f"{module}.")
                ),
                None,
            )
            if registered is not None:
                pending.append(registered)

        visited: set[str] = set()
        reached_contracts: set[str] = set()
        semantics: set[str] = set()
        path_roles: set[str] = set()
        while pending:
            symbol = pending.popleft()
            if symbol in visited:
                continue
            visited.add(symbol)
            matched_contract = next(
                (
                    contract_symbol
                    for contract_symbol in contract_symbols
                    if symbol == contract_symbol
                    or symbol.startswith(f"{contract_symbol}.")
                ),
                None,
            )
            if matched_contract is not None:
                reached_contracts.add(matched_contract)
            definition = index.definitions.get(symbol)
            if definition is None:
                continue
            if isinstance(
                definition.node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                semantics.add(symbol)
            path_roles.update(_path_role_references(definition))
            if definition.symbol in index.registry_symbols:
                continue
            for reference in _definition_references(definition, index):
                matched_contract = next(
                    (
                        contract_symbol
                        for contract_symbol in contract_symbols
                        if reference == contract_symbol
                        or reference.startswith(f"{contract_symbol}.")
                    ),
                    None,
                )
                if matched_contract is not None:
                    reached_contracts.add(matched_contract)
                if reference in index.definitions:
                    pending.append(reference)

        if boundary.validator_callable is not None:
            semantics.add(boundary.validator_callable)
        bindings[boundary.boundary_id] = BoundaryAuthorityBinding(
            authority_symbols=tuple(sorted(reached_contracts)),
            semantic_authorities=tuple(sorted(semantics)),
            path_role_authorities=tuple(sorted(path_roles)),
        )
    return bindings


def discover_configuration_contracts(
    source_root: Path,
) -> tuple[TypedAdapterContract, ...]:
    """Import and inspect every source-declared configuration authority.

    Field-typed dataclasses are followed transitively, so secondary component
    adapters cannot disappear merely because a top-level catalog forgot to list
    them. Source/schema disappearance fails closed instead of shrinking silently.
    """
    declarations = discover_contract_declarations(source_root)
    modules: dict[str, ModuleType] = {}
    adapters: dict[str, type[object]] = {}
    pending: list[type[object]] = []
    for symbol in sorted(declarations.adapter_symbols):
        candidate = _import_symbol(symbol, modules)
        if not inspect.isclass(candidate) or not is_dataclass(candidate):
            raise RuntimeError(f"Discovered adapter is not a dataclass: {symbol}")
        pending.append(candidate)
    while pending:
        adapter = pending.pop()
        symbol = f"{adapter.__module__}.{adapter.__qualname__}"
        if symbol in adapters or not symbol.startswith("src."):
            continue
        adapters[symbol] = adapter
        for annotation in get_type_hints(adapter).values():
            pending.extend(
                nested
                for nested in _nested_dataclass_types(annotation)
                if all(
                    specification.default is MISSING
                    and specification.default_factory is MISSING
                    for specification in fields(cast(Any, nested))
                )
            )

    contracts = [inspect_typed_adapter(adapter) for adapter in adapters.values()]
    for symbol in sorted(declarations.schema_symbols):
        candidate = _import_symbol(symbol, modules)
        if not isinstance(candidate, StrictConfigSchema):
            raise RuntimeError(f"Discovered schema is not StrictConfigSchema: {symbol}")
        contracts.append(_schema_contract(symbol, candidate))
    contracts.extend(
        _path_boundary_contracts(
            source_root,
            declarations.path_boundary_symbols,
        )
    )

    by_symbol = {contract.adapter_symbol: contract for contract in contracts}
    if len(by_symbol) != len(contracts):
        raise RuntimeError("Configuration contract catalog contains duplicate symbols.")
    missing = declarations.all_symbols - set(by_symbol)
    if missing:
        raise RuntimeError(
            "Configuration contract catalog omitted source declarations: "
            + ", ".join(sorted(missing))
        )
    return tuple(by_symbol[symbol] for symbol in sorted(by_symbol))


__all__ = [
    "BoundaryAuthorityBinding",
    "ContractDeclarations",
    "RuntimeBoundaryReference",
    "TypedAdapterContract",
    "discover_boundary_authorities",
    "discover_configuration_contracts",
    "discover_contract_declarations",
    "inspect_typed_adapter",
]
