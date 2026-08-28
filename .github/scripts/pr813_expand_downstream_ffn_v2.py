from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
BLOCK_CONFIG_NAMES = {"TransformerBlockConfig", "CrossAttnBlockConfig"}
FFN_TYPES = (
    "swiglu",
    "mlp",
    "kimi_k3_situglu",
    "deepseek_v4_swiglu",
    "gpt_oss_swiglu",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    result: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            result[child] = node
    return result


def _enclosing_function(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _enclosing_class(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.ClassDef | None:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, ast.ClassDef):
            return current
        current = parents.get(current)
    return None


def _parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }


def _ensure_import(text: str, module: str, names: Iterable[str]) -> str:
    required = [name for name in names if re.search(rf"\b{name}\b", text)]
    missing = []
    for name in required:
        imported = re.search(
            rf"from\s+{re.escape(module)}\s+import(?:\s*\([^)]*\b{name}\b[^)]*\)|[^\n]*\b{name}\b)",
            text,
            flags=re.DOTALL,
        )
        if imported is None:
            missing.append(name)
    if not missing:
        return text
    line = f"from {module} import {', '.join(sorted(missing))}\n"
    marker = "from __future__ import annotations\n"
    if marker in text:
        return text.replace(marker, marker + "\n" + line, 1)
    match = re.match(r'(?s)(\A(?:""".*?"""|\'\'\'.*?\'\'\')\n*)', text)
    if match is not None:
        return text[: match.end()] + "\n" + line + text[match.end() :]
    return line + "\n" + text


def _widen_contract_text(text: str) -> str:
    text = text.replace('Literal["swiglu", "mlp"]', "FFNType")
    text = text.replace("Literal['swiglu', 'mlp']", "FFNType")
    text = text.replace(
        'cast("Literal[\'swiglu\', \'mlp\']",',
        'cast("FFNType",',
    )
    text = text.replace(
        'cast("Literal[\"swiglu\", \"mlp\"]",',
        'cast("FFNType",',
    )
    text = re.sub(
        r"not\s+in\s+\{\s*['\"]swiglu['\"]\s*,\s*['\"]mlp['\"]\s*\}",
        "not in SUPPORTED_FFN_TYPES",
        text,
    )
    text = re.sub(
        r"in\s+\{\s*['\"]swiglu['\"]\s*,\s*['\"]mlp['\"]\s*\}",
        "in SUPPORTED_FFN_TYPES",
        text,
    )
    text = text.replace(
        "must be 'swiglu' or 'mlp'",
        "must be one of the supported FFN variants",
    )
    return text


def _insert_function_parameter(
    text: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    if "ffn_type" in _parameters(function):
        return text
    offsets = _offsets(text)
    function_start = offsets[function.lineno - 1] + function.col_offset
    body_start = offsets[function.body[0].lineno - 1] + function.body[0].col_offset
    header = text[function_start:body_start]

    insertion_absolute: int
    if function.args.kwarg is not None:
        insertion_absolute = (
            offsets[function.args.kwarg.lineno - 1] + function.args.kwarg.col_offset
        )
        line_start = text.rfind("\n", 0, insertion_absolute) + 1
        indent = re.match(r"\s*", text[line_start:insertion_absolute]).group(0)
        payload = f'ffn_type: FFNType = "swiglu",\n{indent}'
    else:
        close_relative = header.rfind(")")
        if close_relative < 0:
            raise RuntimeError(
                f"Unable to locate signature end for {function.name}:{function.lineno}"
            )
        insertion_absolute = function_start + close_relative
        if "\n" in header[:close_relative]:
            header_lines = header[:close_relative].splitlines()
            parameter_lines = [line for line in header_lines[1:] if line.strip()]
            indent = (
                re.match(r"\s*", parameter_lines[-1]).group(0)
                if parameter_lines
                else "        "
            )
            payload = f'{indent}ffn_type: FFNType = "swiglu",\n'
        else:
            before = text[function_start:insertion_absolute].rstrip()
            separator = "" if before.endswith("(") else ", "
            payload = f'{separator}ffn_type: FFNType = "swiglu"'
    return text[:insertion_absolute] + payload + text[insertion_absolute:]


def _replace_node(text: str, node: ast.AST, replacement: str) -> str:
    offsets = _offsets(text)
    if not hasattr(node, "end_lineno") or node.end_lineno is None:
        raise RuntimeError("AST node lacks an end position")
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return text[:start] + replacement + text[end:]


def _insert_keyword(
    text: str,
    call: ast.Call,
    *,
    expression: str,
) -> str:
    offsets = _offsets(text)
    anchor = next((kw for kw in call.keywords if kw.arg == "ffn_dim"), None)
    if anchor is not None:
        value_end = offsets[anchor.value.end_lineno - 1] + anchor.value.end_col_offset
        comma = text.find(",", value_end)
        if comma >= 0 and comma < offsets[call.end_lineno - 1] + call.end_col_offset:
            line_start = text.rfind("\n", 0, offsets[anchor.value.lineno - 1]) + 1
            indent = re.match(r"\s*", text[line_start:]).group(0)
            return (
                text[: comma + 1]
                + f"\n{indent}ffn_type={expression},"
                + text[comma + 1 :]
            )
    call_start = offsets[call.lineno - 1] + call.col_offset
    call_end = offsets[call.end_lineno - 1] + call.end_col_offset
    segment = text[call_start:call_end]
    close_relative = segment.rfind(")")
    if close_relative < 0:
        raise RuntimeError(f"Cannot locate call end at line {call.lineno}")
    close = call_start + close_relative
    if "\n" in segment:
        lines = segment[:close_relative].splitlines()
        argument_lines = [line for line in lines[1:] if line.strip()]
        indent = (
            re.match(r"\s*", argument_lines[-1]).group(0)
            if argument_lines
            else "            "
        )
        payload = f"{indent}ffn_type={expression},\n"
    else:
        before = text[call_start:close].rstrip()
        separator = "" if before.endswith("(") else ", "
        payload = f"{separator}ffn_type={expression}"
    return text[:close] + payload + text[close:]


def _sibling_ffn_expression(text: str, call: ast.Call) -> str | None:
    keyword = next((item for item in call.keywords if item.arg == "ffn_dim"), None)
    if keyword is None:
        return None
    value = keyword.value
    if isinstance(value, ast.Name):
        return "ffn_type"
    if isinstance(value, ast.Attribute):
        base = ast.get_source_segment(text, value.value)
        return None if base is None else f"{base}.ffn_type"
    if isinstance(value, ast.Subscript):
        source = ast.get_source_segment(text, value)
        if source is None:
            return None
        return re.sub(
            r"(['\"])ffn_dim\1\s*\]$",
            lambda match: f'{match.group(1)}ffn_type{match.group(1)}]',
            source,
        )
    return None


def _install_shared_contract() -> None:
    path = SRC / "utils/models/components/ffn_layers.py"
    text = _read(path)
    if "SUPPORTED_FFN_TYPES" not in text:
        marker = "]\n\n\ndef default_ffn_dim"
        replacement = "]\n\nSUPPORTED_FFN_TYPES: frozenset[str] = frozenset(\n    {\n        \"swiglu\",\n        \"mlp\",\n        \"kimi_k3_situglu\",\n        \"deepseek_v4_swiglu\",\n        \"gpt_oss_swiglu\",\n    }\n)\n\n\ndef resolve_ffn_type(value: str) -> FFNType:\n    \"\"\"Validate and narrow one externally supplied FFN selector.\"\"\"\n    if value not in SUPPORTED_FFN_TYPES:\n        supported = \", \".join(sorted(SUPPORTED_FFN_TYPES))\n        raise ValueError(\n            f\"Unsupported ffn_type={value!r}; expected one of: {supported}\"\n        )\n    return cast(FFNType, value)\n\n\ndef default_ffn_dim"
        if marker not in text:
            raise RuntimeError("Unable to install the shared FFN selector contract")
        text = text.replace(marker, replacement, 1)
        _write(path, text)

    path = SRC / "utils/models/components/__init__.py"
    text = _read(path)
    if "SUPPORTED_FFN_TYPES" not in text:
        text = text.replace(
            "    MLP,\n",
            "    MLP,\n    SUPPORTED_FFN_TYPES,\n",
            1,
        )
    if "resolve_ffn_type" not in text:
        text = text.replace(
            "    default_ffn_dim,\n",
            "    default_ffn_dim,\n    resolve_ffn_type,\n",
            1,
        )
    if '"SUPPORTED_FFN_TYPES"' not in text:
        text = text.replace(
            '    "SwiGLU",\n',
            '    "SwiGLU",\n    "SUPPORTED_FFN_TYPES",\n',
            1,
        )
    if '"resolve_ffn_type"' not in text:
        text = text.replace(
            '    "default_ffn_dim",\n',
            '    "default_ffn_dim",\n    "resolve_ffn_type",\n',
            1,
        )
    _write(path, text)


def _direct_block_consumers() -> tuple[Path, ...]:
    result = []
    for path in SRC.rglob("*.py"):
        if path.name == "block.py":
            continue
        text = _read(path)
        if any(name in text for name in BLOCK_CONFIG_NAMES):
            result.append(path)
    return tuple(result)


def _make_direct_block_selectors_configurable(
    paths: tuple[Path, ...],
) -> set[str]:
    callable_targets: set[str] = set()
    for path in paths:
        text = _widen_contract_text(_read(path))
        while True:
            tree = ast.parse(text, filename=str(path))
            parents = _parents(tree)
            changed = False
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or _call_name(node) not in BLOCK_CONFIG_NAMES:
                    continue
                function = _enclosing_function(node, parents)
                if function is None:
                    raise RuntimeError(f"Block configuration outside a function: {path}:{node.lineno}")
                owner = _enclosing_class(function, parents)
                if function.name == "__init__" and owner is not None:
                    callable_targets.add(owner.name)
                else:
                    callable_targets.add(function.name)
                selector = next(
                    (keyword for keyword in node.keywords if keyword.arg == "ffn_type"),
                    None,
                )
                if selector is None:
                    text = _insert_keyword(text, node, expression="ffn_type")
                    changed = True
                    break
                if isinstance(selector.value, ast.Constant):
                    text = _replace_node(text, selector.value, "ffn_type")
                    changed = True
                    break
                if "ffn_type" not in _parameters(function):
                    text = _insert_function_parameter(text, function)
                    changed = True
                    break
            if not changed:
                break
        tree = ast.parse(text, filename=str(path))
        parents = _parents(tree)
        functions = {
            function
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _call_name(node) in BLOCK_CONFIG_NAMES
            if (function := _enclosing_function(node, parents)) is not None
        }
        for function in sorted(functions, key=lambda item: item.lineno, reverse=True):
            if "ffn_type" not in _parameters(function):
                text = _insert_function_parameter(text, function)
                tree = ast.parse(text, filename=str(path))
                parents = _parents(tree)
        text = _ensure_import(
            text,
            "src.utils.models.components.ffn_layers",
            ("FFNType",),
        )
        _write(path, text)
    return callable_targets


def _propagate_selector_through_call_graph(initial_targets: set[str]) -> set[str]:
    all_targets = set(initial_targets)
    frontier = set(initial_targets)
    for _ in range(16):
        if not frontier:
            return all_targets
        next_frontier: set[str] = set()
        for path in SRC.rglob("*.py"):
            text = _read(path)
            while True:
                tree = ast.parse(text, filename=str(path))
                parents = _parents(tree)
                changed = False
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call) or _call_name(node) not in frontier:
                        continue
                    if any(keyword.arg is None for keyword in node.keywords):
                        continue
                    if any(keyword.arg == "ffn_type" for keyword in node.keywords):
                        continue
                    function = _enclosing_function(node, parents)
                    if function is None:
                        continue
                    expression = _sibling_ffn_expression(text, node)
                    if expression is None:
                        expression = "ffn_type"
                    if expression == "ffn_type" and "ffn_type" not in _parameters(function):
                        text = _insert_function_parameter(text, function)
                        owner = _enclosing_class(function, parents)
                        if function.name == "__init__" and owner is not None:
                            next_frontier.add(owner.name)
                        else:
                            next_frontier.add(function.name)
                        changed = True
                        break
                    text = _insert_keyword(text, node, expression=expression)
                    changed = True
                    break
                if not changed:
                    break
            if "ffn_type" in text:
                text = _ensure_import(
                    text,
                    "src.utils.models.components.ffn_layers",
                    ("FFNType",),
                )
                _write(path, text)
        next_frontier -= all_targets
        all_targets.update(next_frontier)
        frontier = next_frontier
    raise RuntimeError("FFN selector call-graph propagation did not converge")


def _impacted_tasks(consumers: tuple[Path, ...]) -> set[str]:
    result: set[str] = set()
    for path in consumers:
        relative = path.relative_to(SRC)
        if len(relative.parts) >= 2 and relative.parts[0] == "tasks":
            result.add(relative.parts[1])
    return result


def _widen_task_sources(tasks: set[str]) -> None:
    for task in sorted(tasks):
        root = SRC / "tasks" / task
        for path in root.rglob("*.py"):
            text = _read(path)
            if "ffn_type" not in text:
                continue
            widened = _widen_contract_text(text)
            if widened != text:
                widened = _ensure_import(
                    widened,
                    "src.utils.models.components.ffn_layers",
                    ("FFNType", "SUPPORTED_FFN_TYPES"),
                )
                _write(path, widened)


def _normalize_blcs_track_query_contract() -> None:
    path = SRC / "tasks/blcs/configuration.py"
    text = _read(path)
    class_pattern = re.compile(
        r"(?P<header>@dataclass\([^\n]*\)\nclass\s+(?P<name>TrackQuery(?:Reference)?(?:Ablation)?ModelConfig):\n)"
        r"(?P<body>.*?)(?=\n\n@dataclass|\n\n[A-Z_]+\s*=|\n\ndef\s)",
        flags=re.DOTALL,
    )

    def replace_class(match: re.Match[str]) -> str:
        body = match.group("body")
        body = re.sub(r"^\s+ffn_type:\s*FFNType(?:\s*=\s*[^\n]+)?\n", "", body, flags=re.MULTILINE)
        body = body.rstrip() + '\n    ffn_type: FFNType = "swiglu"\n'
        return match.group("header") + body

    text = class_pattern.sub(replace_class, text)
    text = re.sub(
        r'(?P<indent>\s+)"ffn_dim",\n(?!\s+"ffn_type",)',
        lambda match: match.group(0) + f'{match.group("indent")}"ffn_type",\n',
        text,
    )
    text = re.sub(
        r'(?P<indent>\s+)"ffn_dim":\s*int,\n(?!\s+"ffn_type":)',
        lambda match: match.group(0) + f'{match.group("indent")}"ffn_type": str,\n',
        text,
    )
    text = re.sub(
        r'(?P<indent>\s+)ffn_dim=(?P<value>[^\n]+),\n(?!\s+ffn_type=)',
        lambda match: (
            match.group(0)
            + f'{match.group("indent")}ffn_type=cast("FFNType", model["ffn_type"]),\n'
        ),
        text,
    )
    validation_marker = "    result: BLCSModelConfig\n"
    if "model[\"ffn_type\"] not in SUPPORTED_FFN_TYPES" not in text:
        validation = (
            "    if \"ffn_type\" in model and model[\"ffn_type\"] not in SUPPORTED_FFN_TYPES:\n"
            "        supported = \", \".join(sorted(SUPPORTED_FFN_TYPES))\n"
            "        raise SemanticConfigurationError(\n"
            "            f\"model.ffn_type must be one of: {supported}.\"\n"
            "        )\n"
        )
        if validation_marker not in text:
            raise RuntimeError("BLCS parse_model_config marker was not found")
        text = text.replace(validation_marker, validation_marker + validation, 1)
    text = _ensure_import(
        text,
        "src.utils.models.components.ffn_layers",
        ("FFNType", "SUPPORTED_FFN_TYPES"),
    )
    _write(path, text)

    for yaml_path in (SRC / "tasks/blcs/configs/model").rglob("*.yaml"):
        yaml = _read(yaml_path)
        if not re.search(r"^name:\s*blcs_track_query", yaml, flags=re.MULTILINE):
            continue
        if re.search(r"^ffn_type:", yaml, flags=re.MULTILINE):
            continue
        yaml, count = re.subn(
            r"^(ffn_dim:\s*[^\n]+\n)",
            r"\1ffn_type: swiglu\n",
            yaml,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError(f"Unable to add ffn_type to {yaml_path}")
        _write(yaml_path, yaml)


def _remove_plcs_ablation_standard_only_rule() -> None:
    path = SRC / "tasks/plcs/configuration.py"
    text = _read(path)
    text = re.sub(
        r"(?P<indent>\s+)if name in _TRACK_QUERY_ABLATION_MODEL_NAMES:\n"
        r"(?P=indent)    if _string\(mapping, \"ffn_type\", path=\"model\"\) != \"swiglu\":\n"
        r"(?P=indent)        raise SemanticConfigurationError\(\n"
        r"(?P=indent)            \"PLCS track-query ablation requires model.ffn_type='swiglu'.\"\n"
        r"(?P=indent)        \)\n",
        lambda match: f'{match.group("indent")}if name in _TRACK_QUERY_ABLATION_MODEL_NAMES:\n',
        text,
    )
    _write(path, text)


def _add_missing_model_yaml_selectors(tasks: set[str]) -> None:
    for task in sorted(tasks):
        model_root = SRC / "tasks" / task / "configs/model"
        if not model_root.exists():
            continue
        for path in model_root.rglob("*.yaml"):
            text = _read(path)
            if not re.search(r"^ffn_dim:\s*", text, flags=re.MULTILINE):
                continue
            if re.search(r"^ffn_type:\s*", text, flags=re.MULTILINE):
                continue
            text, count = re.subn(
                r"^(ffn_dim:\s*[^\n]+\n)",
                r"\1ffn_type: swiglu\n",
                text,
                count=1,
                flags=re.MULTILINE,
            )
            if count == 1:
                _write(path, text)


def _repair_impacted_test_contracts(tasks: set[str]) -> None:
    roots = [TESTS / "unit/tasks" / task for task in tasks]
    roots.extend(TESTS / "integration/tasks" / task for task in tasks)
    roots.extend(TESTS / "e2e/tasks" / task for task in tasks)
    roots.append(TESTS / "integration/tasks")
    visited: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path in visited:
                continue
            visited.add(path)
            text = _read(path)
            if "ffn_type" in text:
                widened = _widen_contract_text(text)
                if widened != text:
                    widened = _ensure_import(
                        widened,
                        "src.utils.models.components.ffn_layers",
                        ("FFNType", "SUPPORTED_FFN_TYPES"),
                    )
                    text = widened
            if "blcs_track_query" in text:
                text = re.sub(
                    r'(?P<indent>\s+)"ffn_dim":\s*(?P<value>[^\n]+),\n(?!\s+"ffn_type":)',
                    lambda match: (
                        match.group(0)
                        + f'{match.group("indent")}"ffn_type": "swiglu",\n'
                    ),
                    text,
                )
            if "PLCS track-query ablation requires" in text:
                functions = []
                tree = ast.parse(text, filename=str(path))
                offsets = _offsets(text)
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    segment = ast.get_source_segment(text, node) or ""
                    if "PLCS track-query ablation requires" not in segment:
                        continue
                    replacement = segment.replace('"mlp"', '"unknown"')
                    replacement = replacement.replace(
                        "PLCS track-query ablation requires model.ffn_type='swiglu'.",
                        "model.ffn_type",
                    )
                    start = offsets[node.lineno - 1] + node.col_offset
                    end = offsets[node.end_lineno - 1] + node.end_col_offset
                    functions.append((start, end, replacement))
                for start, end, replacement in sorted(functions, reverse=True):
                    text = text[:start] + replacement + text[end:]
            _write(path, text)


def _write_static_audit_test(callable_targets: set[str], tasks: set[str]) -> None:
    path = TESTS / "unit/utils/models/components/test_ffn_downstream_configuration.py"
    targets_literal = repr(sorted(callable_targets))
    tasks_literal = repr(sorted(tasks))
    _write(
        path,
        f'''"""Repository-wide configuration propagation for TransformerBlock FFNs."""

from __future__ import annotations

import ast
from pathlib import Path

from src.utils.models.components.ffn_layers import SUPPORTED_FFN_TYPES

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
CALLABLE_TARGETS = set({targets_literal})
IMPACTED_TASKS = set({tasks_literal})
BLOCK_CONFIG_NAMES = {{"TransformerBlockConfig", "CrossAttnBlockConfig"}}


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_supported_ffn_type_contract_contains_all_architecture_variants() -> None:
    assert SUPPORTED_FFN_TYPES == {{
        "swiglu",
        "mlp",
        "kimi_k3_situglu",
        "deepseek_v4_swiglu",
        "gpt_oss_swiglu",
    }}


def test_direct_block_consumers_use_runtime_ffn_selectors() -> None:
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        if path.name == "block.py":
            continue
        text = path.read_text(encoding="utf-8")
        if not any(name in text for name in BLOCK_CONFIG_NAMES):
            continue
        tree = ast.parse(text, filename=str(path))
        parents: dict[ast.AST, ast.AST] = {{}}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in BLOCK_CONFIG_NAMES:
                continue
            selector = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "ffn_type"),
                None,
            )
            current: ast.AST | None = node
            function: ast.FunctionDef | ast.AsyncFunctionDef | None = None
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function = current
                    break
                current = parents.get(current)
            parameters = set()
            if function is not None:
                parameters = {{
                    argument.arg
                    for argument in (
                        *function.args.posonlyargs,
                        *function.args.args,
                        *function.args.kwonlyargs,
                    )
                }}
            if selector is None or isinstance(selector, ast.Constant) or "ffn_type" not in parameters:
                violations.append(f"{{path.relative_to(REPOSITORY_ROOT)}}:{{node.lineno}}")
    assert not violations, "non-configurable block selectors: " + ", ".join(violations)


def test_block_wrapper_calls_forward_ffn_selector() -> None:
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in CALLABLE_TARGETS:
                continue
            if any(keyword.arg is None for keyword in node.keywords):
                continue
            if not any(keyword.arg == "ffn_type" for keyword in node.keywords):
                violations.append(f"{{path.relative_to(REPOSITORY_ROOT)}}:{{node.lineno}}")
    assert not violations, "wrapper calls missing ffn_type: " + ", ".join(violations)


def test_impacted_task_sources_have_no_stale_two_option_contract() -> None:
    stale = (
        'Literal["swiglu", "mlp"]',
        "Literal['swiglu', 'mlp']",
        '{{"swiglu", "mlp"}}',
        "{{'swiglu', 'mlp'}}",
    )
    violations: list[str] = []
    for task in sorted(IMPACTED_TASKS):
        for path in (SOURCE_ROOT / "tasks" / task).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "ffn_type" in text and any(fragment in text for fragment in stale):
                violations.append(str(path.relative_to(REPOSITORY_ROOT)))
    assert not violations, "stale FFN contracts: " + ", ".join(violations)


def test_model_yamls_with_ffn_width_expose_ffn_selector() -> None:
    violations: list[str] = []
    for task in sorted(IMPACTED_TASKS):
        root = SOURCE_ROOT / "tasks" / task / "configs/model"
        if not root.exists():
            continue
        for path in root.rglob("*.yaml"):
            text = path.read_text(encoding="utf-8")
            if "ffn_dim:" in text and "ffn_type:" not in text:
                violations.append(str(path.relative_to(REPOSITORY_ROOT)))
    assert not violations, "model YAMLs missing ffn_type: " + ", ".join(violations)
''',
    )


def _audit_direct_block_calls(paths: tuple[Path, ...]) -> None:
    violations: list[str] = []
    for path in paths:
        text = _read(path)
        tree = ast.parse(text, filename=str(path))
        parents = _parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in BLOCK_CONFIG_NAMES:
                continue
            selector = next((kw.value for kw in node.keywords if kw.arg == "ffn_type"), None)
            function = _enclosing_function(node, parents)
            if (
                selector is None
                or isinstance(selector, ast.Constant)
                or function is None
                or "ffn_type" not in _parameters(function)
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    if violations:
        raise RuntimeError("Non-configurable block selectors: " + ", ".join(violations))


def main() -> None:
    _install_shared_contract()
    consumers = _direct_block_consumers()
    targets = _make_direct_block_selectors_configurable(consumers)
    targets = _propagate_selector_through_call_graph(targets)
    tasks = _impacted_tasks(consumers)
    _widen_task_sources(tasks)
    _normalize_blcs_track_query_contract()
    _remove_plcs_ablation_standard_only_rule()
    _add_missing_model_yaml_selectors(tasks)
    _repair_impacted_test_contracts(tasks)
    _write_static_audit_test(targets, tasks)
    _audit_direct_block_calls(consumers)


if __name__ == "__main__":
    main()
