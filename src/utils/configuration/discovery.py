"""Source-only discovery of executable configuration boundaries."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.utils.configuration.inventory import BoundaryKind

__all__ = ["DiscoveredRuntimeBoundary", "discover_runtime_boundaries"]


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


def source_module_name(source_root: Path, path: Path) -> str:
    """Return the import path for ``path`` below an absolute ``src`` root."""
    return ".".join(path.relative_to(source_root.parent).with_suffix("").parts)


def ast_call_name(node: ast.Call) -> str:
    """Return the terminal source name of an AST call target."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else (statement.target,)
        )
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value.value
    return constants


def _resolved_string(node: ast.AST, constants: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _contains_named_call(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Call) and ast_call_name(child) == name
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
            and ast_call_name(statement.value) in {"main", "_main"}
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
        if isinstance(child, ast.Call) and ast_call_name(child) == "hydra_main":
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
            or ast_call_name(node) != "register_boundary_validator"
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
            and ast_call_name(child) == "register_boundary_validator"
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
            or ast_call_name(value) != "NonHydraPathBoundary"
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
    """Discover runtime callables and validator bindings directly from source."""
    trees: dict[str, ast.Module] = {}
    constants_by_module: dict[str, dict[str, str]] = {}
    registrations: dict[str, str] = {}
    subprocess_invokers: dict[str, set[str]] = {}
    for path in sorted(source_root.rglob("*.py")):
        module = source_module_name(source_root, path)
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
                    BoundaryKind.ARGPARSE
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
                    BoundaryKind.SUBPROCESS_MODULE
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
