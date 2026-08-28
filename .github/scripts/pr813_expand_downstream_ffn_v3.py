from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
BLOCK_CONFIG_NAMES = {"TransformerBlockConfig", "CrossAttnBlockConfig"}
FFN_MODULE = "src.utils.models.components.ffn_layers"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _offsets(text: str) -> list[int]:
    result = [0]
    for line in text.splitlines(keepends=True):
        result.append(result[-1] + len(line))
    return result


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
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


def _parameter_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return {
        arg.arg
        for arg in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }


def _ensure_import(text: str, module: str, names: Iterable[str]) -> str:
    required = [name for name in names if re.search(rf"\b{name}\b", text)]
    missing: list[str] = []
    for name in required:
        pattern = re.compile(
            rf"from\s+{re.escape(module)}\s+import"
            rf"(?:\s*\([^)]*\b{name}\b[^)]*\)|[^\n]*\b{name}\b)",
            flags=re.DOTALL,
        )
        if pattern.search(text) is None:
            missing.append(name)
    if not missing:
        return text
    import_line = f"from {module} import {', '.join(sorted(missing))}\n"
    future = "from __future__ import annotations\n"
    if future in text:
        return text.replace(future, future + "\n" + import_line, 1)
    module_docstring = re.match(
        r"\A(?:\s*)(?:\"\"\".*?\"\"\"|'''(?:.|\n)*?''')\s*\n",
        text,
        flags=re.DOTALL,
    )
    if module_docstring is not None:
        position = module_docstring.end()
        return text[:position] + "\n" + import_line + text[position:]
    return import_line + "\n" + text


def _widen_two_option_contract(text: str) -> str:
    text = text.replace('Literal["swiglu", "mlp"]', "FFNType")
    text = text.replace("Literal['swiglu', 'mlp']", "FFNType")
    text = text.replace(
        'cast("Literal[\'swiglu\', \'mlp\']",',
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


def _replace_node(text: str, node: ast.AST, replacement: str) -> str:
    offsets = _offsets(text)
    if node.end_lineno is None or node.end_col_offset is None:
        raise RuntimeError("AST node has no end position")
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return text[:start] + replacement + text[end:]


def _add_ffn_parameter(
    text: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    if "ffn_type" in _parameter_names(function):
        return text
    offsets = _offsets(text)
    function_start = offsets[function.lineno - 1] + function.col_offset
    body_start = offsets[function.body[0].lineno - 1] + function.body[0].col_offset
    header = text[function_start:body_start]

    if function.args.kwarg is not None:
        kwarg_start = offsets[function.args.kwarg.lineno - 1] + function.args.kwarg.col_offset
        line_start = text.rfind("\n", 0, kwarg_start) + 1
        if line_start > function_start:
            indent = text[line_start:kwarg_start]
            payload = f'{indent}ffn_type: FFNType = "swiglu",\n'
            return text[:line_start] + payload + text[line_start:]
        return (
            text[:kwarg_start]
            + 'ffn_type: FFNType = "swiglu", '
            + text[kwarg_start:]
        )

    close_relative = header.rfind(")")
    if close_relative < 0:
        raise RuntimeError(f"Cannot locate signature end at line {function.lineno}")
    close = function_start + close_relative
    if "\n" not in header[:close_relative]:
        before = text[function_start:close].rstrip()
        separator = "" if before.endswith("(") else ", "
        return (
            text[:close]
            + f'{separator}ffn_type: FFNType = "swiglu"'
            + text[close:]
        )

    close_line_start = text.rfind("\n", 0, close) + 1
    close_indent = text[close_line_start:close]
    argument_lines = [
        line
        for line in header[:close_relative].splitlines()[1:]
        if line.strip() and not line.lstrip().startswith(("/", "*"))
    ]
    if argument_lines:
        parameter_indent = re.match(r"\s*", argument_lines[-1]).group(0)
    else:
        parameter_indent = close_indent + "    "
    replacement = (
        f'{parameter_indent}ffn_type: FFNType = "swiglu",\n{close_indent}'
    )
    return text[:close_line_start] + replacement + text[close:]


def _insert_call_keyword(text: str, call: ast.Call, expression: str) -> str:
    offsets = _offsets(text)
    anchor = next((kw for kw in call.keywords if kw.arg == "ffn_dim"), None)
    call_end = offsets[call.end_lineno - 1] + call.end_col_offset
    if anchor is not None:
        anchor_end = offsets[anchor.value.end_lineno - 1] + anchor.value.end_col_offset
        comma = text.find(",", anchor_end, call_end)
        if comma >= 0:
            line_start = text.rfind("\n", 0, offsets[anchor.value.lineno - 1]) + 1
            indent = re.match(r"\s*", text[line_start:]).group(0)
            return (
                text[: comma + 1]
                + f"\n{indent}ffn_type={expression},"
                + text[comma + 1 :]
            )

    call_start = offsets[call.lineno - 1] + call.col_offset
    segment = text[call_start:call_end]
    close_relative = segment.rfind(")")
    if close_relative < 0:
        raise RuntimeError(f"Cannot locate call end at line {call.lineno}")
    close = call_start + close_relative
    if "\n" not in segment[:close_relative]:
        before = text[call_start:close].rstrip()
        separator = "" if before.endswith("(") else ", "
        return text[:close] + f"{separator}ffn_type={expression}" + text[close:]

    close_line_start = text.rfind("\n", 0, close) + 1
    close_indent = text[close_line_start:close]
    argument_lines = [line for line in segment[:close_relative].splitlines()[1:] if line.strip()]
    argument_indent = (
        re.match(r"\s*", argument_lines[-1]).group(0)
        if argument_lines
        else close_indent + "    "
    )
    replacement = f"{argument_indent}ffn_type={expression},\n{close_indent}"
    return text[:close_line_start] + replacement + text[close:]


def _sibling_selector_expression(text: str, call: ast.Call) -> str | None:
    width = next((kw.value for kw in call.keywords if kw.arg == "ffn_dim"), None)
    if width is None:
        return None
    if isinstance(width, ast.Name):
        return "ffn_type"
    if isinstance(width, ast.Attribute):
        base = ast.get_source_segment(text, width.value)
        return None if base is None else f"{base}.ffn_type"
    if isinstance(width, ast.Subscript):
        source = ast.get_source_segment(text, width)
        if source is None:
            return None
        replaced = re.sub(
            r"(?P<quote>['\"])ffn_dim(?P=quote)\s*\]$",
            lambda match: f'{match.group("quote")}ffn_type{match.group("quote")}]',
            source,
        )
        return replaced if replaced != source else None
    return None


def _install_shared_selector_contract() -> None:
    path = SRC / "utils/models/components/ffn_layers.py"
    text = _read(path)
    if "SUPPORTED_FFN_TYPES" not in text:
        marker = "]\n\n\ndef default_ffn_dim"
        contract = "]\n\nSUPPORTED_FFN_TYPES: frozenset[str] = frozenset(\n    {\n        \"swiglu\",\n        \"mlp\",\n        \"kimi_k3_situglu\",\n        \"deepseek_v4_swiglu\",\n        \"gpt_oss_swiglu\",\n    }\n)\n\n\ndef resolve_ffn_type(value: str) -> FFNType:\n    \"\"\"Validate and narrow an externally supplied FFN selector.\"\"\"\n    if value not in SUPPORTED_FFN_TYPES:\n        supported = \", \".join(sorted(SUPPORTED_FFN_TYPES))\n        raise ValueError(\n            f\"Unsupported ffn_type={value!r}; expected one of: {supported}\"\n        )\n    return cast(FFNType, value)\n\n\ndef default_ffn_dim"
        if marker not in text:
            raise RuntimeError("Shared FFN selector insertion point was not found")
        text = text.replace(marker, contract, 1)
        _write(path, text)

    path = SRC / "utils/models/components/__init__.py"
    text = _read(path)
    if "SUPPORTED_FFN_TYPES" not in text:
        text = text.replace("    MLP,\n", "    MLP,\n    SUPPORTED_FFN_TYPES,\n", 1)
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


def _direct_consumer_paths() -> tuple[Path, ...]:
    result: list[Path] = []
    for path in SRC.rglob("*.py"):
        if path.name == "block.py":
            continue
        text = _read(path)
        if any(name in text for name in BLOCK_CONFIG_NAMES):
            result.append(path)
    return tuple(result)


def _rewrite_direct_consumers(paths: tuple[Path, ...]) -> set[str]:
    targets: set[str] = set()
    for path in paths:
        text = _widen_two_option_contract(_read(path))
        while True:
            tree = ast.parse(text, filename=str(path))
            parents = _parent_map(tree)
            changed = False
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call) or _call_name(call) not in BLOCK_CONFIG_NAMES:
                    continue
                function = _enclosing_function(call, parents)
                if function is None:
                    raise RuntimeError(f"Block config outside function: {path}:{call.lineno}")
                selector = next(
                    (kw for kw in call.keywords if kw.arg == "ffn_type"),
                    None,
                )
                if selector is None:
                    text = _insert_call_keyword(text, call, "ffn_type")
                    changed = True
                    break
                if isinstance(selector.value, ast.Constant):
                    text = _replace_node(text, selector.value, "ffn_type")
                    changed = True
                    break
                if (
                    isinstance(selector.value, ast.Name)
                    and selector.value.id == "ffn_type"
                    and "ffn_type" not in _parameter_names(function)
                ):
                    text = _add_ffn_parameter(text, function)
                    changed = True
                    break
            if not changed:
                break

        tree = ast.parse(text, filename=str(path))
        parents = _parent_map(tree)
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or _call_name(call) not in BLOCK_CONFIG_NAMES:
                continue
            function = _enclosing_function(call, parents)
            if function is None or "ffn_type" not in _parameter_names(function):
                continue
            owner = _enclosing_class(function, parents)
            targets.add(owner.name if function.name == "__init__" and owner else function.name)
        text = _ensure_import(text, FFN_MODULE, ("FFNType",))
        _write(path, text)
    return targets


def _propagate_through_wrappers(initial_targets: set[str]) -> set[str]:
    targets = set(initial_targets)
    frontier = set(initial_targets)
    for _ in range(16):
        if not frontier:
            return targets
        added_targets: set[str] = set()
        for path in SRC.rglob("*.py"):
            text = _read(path)
            while True:
                tree = ast.parse(text, filename=str(path))
                parents = _parent_map(tree)
                changed = False
                for call in ast.walk(tree):
                    if not isinstance(call, ast.Call) or _call_name(call) not in frontier:
                        continue
                    if any(kw.arg is None for kw in call.keywords):
                        continue
                    if any(kw.arg == "ffn_type" for kw in call.keywords):
                        continue
                    function = _enclosing_function(call, parents)
                    if function is None:
                        continue
                    expression = _sibling_selector_expression(text, call) or "ffn_type"
                    if expression == "ffn_type" and "ffn_type" not in _parameter_names(function):
                        text = _add_ffn_parameter(text, function)
                        owner = _enclosing_class(function, parents)
                        added_targets.add(
                            owner.name if function.name == "__init__" and owner else function.name
                        )
                        changed = True
                        break
                    text = _insert_call_keyword(text, call, expression)
                    changed = True
                    break
                if not changed:
                    break
            if "FFNType" in text:
                text = _ensure_import(text, FFN_MODULE, ("FFNType",))
            _write(path, text)
        added_targets -= targets
        targets.update(added_targets)
        frontier = added_targets
    raise RuntimeError("FFN wrapper propagation failed to converge")


def _impacted_tasks(paths: tuple[Path, ...]) -> set[str]:
    tasks: set[str] = set()
    for path in paths:
        relative = path.relative_to(SRC)
        if len(relative.parts) > 1 and relative.parts[0] == "tasks":
            tasks.add(relative.parts[1])
    return tasks


def _widen_impacted_task_sources(tasks: set[str]) -> None:
    for task in sorted(tasks):
        for path in (SRC / "tasks" / task).rglob("*.py"):
            text = _read(path)
            if "ffn_type" not in text:
                continue
            widened = _widen_two_option_contract(text)
            if widened != text:
                widened = _ensure_import(
                    widened,
                    FFN_MODULE,
                    ("FFNType", "SUPPORTED_FFN_TYPES"),
                )
                _write(path, widened)


def _append_missing_config_dataclass_fields(path: Path) -> set[str]:
    text = _read(path)
    class_names: set[str] = set()
    while True:
        tree = ast.parse(text, filename=str(path))
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            fields = [
                child
                for child in node.body
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
            ]
            names = {field.target.id for field in fields}
            if "ffn_dim" not in names:
                continue
            class_names.add(node.name)
            if "ffn_type" in names:
                continue
            last_field = fields[-1]
            offsets = _offsets(text)
            if last_field.end_lineno is None:
                raise RuntimeError(f"Missing class-field end position in {path}")
            end = offsets[last_field.end_lineno]
            indent = " " * last_field.col_offset
            text = text[:end] + f'{indent}ffn_type: FFNType = "swiglu"\n' + text[end:]
            changed = True
            break
        if not changed:
            break
    if class_names:
        text = _ensure_import(text, FFN_MODULE, ("FFNType",))
        _write(path, text)
    return class_names


def _augment_config_key_specs(path: Path) -> None:
    text = _read(path)
    text = re.sub(
        r'(?P<indent>^[ \t]+)"ffn_dim":\s*int,\n(?![ \t]+"ffn_type":)',
        lambda match: match.group(0) + f'{match.group("indent")}"ffn_type": str,\n',
        text,
        flags=re.MULTILINE,
    )
    _write(path, text)


def _add_config_constructor_selectors(path: Path, config_classes: set[str]) -> None:
    text = _read(path)
    while True:
        tree = ast.parse(text, filename=str(path))
        changed = False
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or _call_name(call) not in config_classes:
                continue
            if any(kw.arg == "ffn_type" for kw in call.keywords):
                continue
            expression = _sibling_selector_expression(text, call)
            if expression is None:
                continue
            if "[" in expression:
                expression = f'resolve_ffn_type(cast("str", {expression}))'
            text = _insert_call_keyword(text, call, expression)
            changed = True
            break
        if not changed:
            break
    if "resolve_ffn_type" in text:
        text = _ensure_import(text, FFN_MODULE, ("resolve_ffn_type",))
    _write(path, text)


def _normalize_configuration_contracts(tasks: set[str]) -> None:
    for task in sorted(tasks):
        config_paths = sorted((SRC / "tasks" / task).glob("configuration*.py"))
        config_classes: set[str] = set()
        for path in config_paths:
            text = _widen_two_option_contract(_read(path))
            text = _ensure_import(
                text,
                FFN_MODULE,
                ("FFNType", "SUPPORTED_FFN_TYPES"),
            )
            _write(path, text)
            config_classes.update(_append_missing_config_dataclass_fields(path))
            _augment_config_key_specs(path)
        for path in config_paths:
            _add_config_constructor_selectors(path, config_classes)


def _normalize_blcs_track_query_fields() -> None:
    path = SRC / "tasks/blcs/configuration.py"
    text = _read(path)
    text = re.sub(
        r'(?P<indent>^[ \t]+)"ffn_dim",\n(?![ \t]+"ffn_type",)',
        lambda match: match.group(0) + f'{match.group("indent")}"ffn_type",\n',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'(?P<indent>^[ \t]+)"ffn_dim":\s*int,\n(?![ \t]+"ffn_type":)',
        lambda match: match.group(0) + f'{match.group("indent")}"ffn_type": str,\n',
        text,
        flags=re.MULTILINE,
    )
    marker = "    result: BLCSModelConfig\n"
    if 'model["ffn_type"] not in SUPPORTED_FFN_TYPES' not in text:
        validation = (
            '    if "ffn_type" in model and model["ffn_type"] not in SUPPORTED_FFN_TYPES:\n'
            '        supported = ", ".join(sorted(SUPPORTED_FFN_TYPES))\n'
            '        raise SemanticConfigurationError(\n'
            '            f"model.ffn_type must be one of: {supported}."\n'
            '        )\n'
        )
        if marker not in text:
            raise RuntimeError("BLCS model parser marker was not found")
        text = text.replace(marker, marker + validation, 1)
    text = _ensure_import(
        text,
        FFN_MODULE,
        ("FFNType", "SUPPORTED_FFN_TYPES", "resolve_ffn_type"),
    )
    _write(path, text)


def _remove_plcs_ablation_ffn_pin() -> None:
    path = SRC / "tasks/plcs/configuration.py"
    lines = _read(path).splitlines(keepends=True)
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        output.append(line)
        if "if name in _TRACK_QUERY_ABLATION_MODEL_NAMES:" not in line:
            index += 1
            continue
        outer_indent = len(line) - len(line.lstrip())
        inner_indent = outer_indent + 4
        next_index = index + 1
        if (
            next_index < len(lines)
            and len(lines[next_index]) - len(lines[next_index].lstrip()) == inner_indent
            and 'if _string(mapping, "ffn_type", path="model") != "swiglu":' in lines[next_index]
        ):
            next_index += 1
            while next_index < len(lines):
                candidate = lines[next_index]
                stripped = candidate.strip()
                indent = len(candidate) - len(candidate.lstrip())
                if stripped and indent == inner_indent:
                    break
                next_index += 1
            index = next_index
            continue
        index += 1
    text = "".join(output)
    if "PLCS track-query ablation requires" in text:
        raise RuntimeError("The PLCS standard-SwiGLU-only rule remains")
    _write(path, text)


def _add_model_yaml_selectors(tasks: set[str]) -> None:
    for task in sorted(tasks):
        root = SRC / "tasks" / task / "configs/model"
        if not root.exists():
            continue
        for path in root.rglob("*.yaml"):
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


def _update_impacted_tests(tasks: set[str]) -> None:
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
                widened = _widen_two_option_contract(text)
                if widened != text:
                    text = _ensure_import(
                        widened,
                        FFN_MODULE,
                        ("FFNType", "SUPPORTED_FFN_TYPES"),
                    )
            if "blcs_track_query" in text:
                text = re.sub(
                    r'(?P<indent>^[ \t]+)"ffn_dim":\s*(?P<value>[^\n]+),\n'
                    r'(?![ \t]+"ffn_type":)',
                    lambda match: (
                        match.group(0)
                        + f'{match.group("indent")}"ffn_type": "swiglu",\n'
                    ),
                    text,
                    flags=re.MULTILINE,
                )
            if "PLCS track-query ablation requires" in text:
                tree = ast.parse(text, filename=str(path))
                offsets = _offsets(text)
                changes: list[tuple[int, int, str]] = []
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    source = ast.get_source_segment(text, node) or ""
                    if "PLCS track-query ablation requires" not in source:
                        continue
                    replacement = source.replace('"mlp"', '"unknown"')
                    replacement = re.sub(
                        r'"PLCS track-query ablation requires "\s*\n\s*'
                        r'"model\.ffn_type=\'swiglu\'\."',
                        '"model.ffn_type"',
                        replacement,
                    )
                    replacement = replacement.replace(
                        "PLCS track-query ablation requires model.ffn_type='swiglu'.",
                        "model.ffn_type",
                    )
                    start = offsets[node.lineno - 1] + node.col_offset
                    end = offsets[node.end_lineno - 1] + node.end_col_offset
                    changes.append((start, end, replacement))
                for start, end, replacement in sorted(changes, reverse=True):
                    text = text[:start] + replacement + text[end:]
            _write(path, text)


def _write_audit_test(targets: set[str], tasks: set[str]) -> None:
    path = TESTS / "unit/utils/models/components/test_ffn_downstream_configuration.py"
    target_literal = repr(sorted(targets))
    task_literal = repr(sorted(tasks))
    _write(
        path,
        f'''"""Repository-wide FFN-selector propagation contracts."""

from __future__ import annotations

import ast
from pathlib import Path

from src.utils.models.components.ffn_layers import SUPPORTED_FFN_TYPES

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "src"
BLOCK_CONFIG_NAMES = {{"TransformerBlockConfig", "CrossAttnBlockConfig"}}
WRAPPER_TARGETS = set({target_literal})
IMPACTED_TASKS = set({task_literal})


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_supported_selector_set_is_complete() -> None:
    assert SUPPORTED_FFN_TYPES == {{
        "swiglu",
        "mlp",
        "kimi_k3_situglu",
        "deepseek_v4_swiglu",
        "gpt_oss_swiglu",
    }}


def test_direct_block_configs_do_not_hardcode_ffn_architecture() -> None:
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.name == "block.py":
            continue
        text = path.read_text(encoding="utf-8")
        if not any(name in text for name in BLOCK_CONFIG_NAMES):
            continue
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in BLOCK_CONFIG_NAMES:
                continue
            selector = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "ffn_type"),
                None,
            )
            if selector is None or isinstance(selector, ast.Constant):
                violations.append(f"{{path.relative_to(ROOT)}}:{{node.lineno}}")
    assert not violations, "hardcoded block FFNs: " + ", ".join(violations)


def test_block_wrapper_calls_forward_ffn_selector() -> None:
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in WRAPPER_TARGETS:
                continue
            if any(keyword.arg is None for keyword in node.keywords):
                continue
            if not any(keyword.arg == "ffn_type" for keyword in node.keywords):
                violations.append(f"{{path.relative_to(ROOT)}}:{{node.lineno}}")
    assert not violations, "wrapper calls without ffn_type: " + ", ".join(violations)


def test_impacted_task_sources_have_no_stale_two_option_contract() -> None:
    stale = (
        'Literal["swiglu", "mlp"]',
        "Literal['swiglu', 'mlp']",
        '{{"swiglu", "mlp"}}',
        "{{'swiglu', 'mlp'}}",
    )
    violations: list[str] = []
    for task in sorted(IMPACTED_TASKS):
        for path in (SRC / "tasks" / task).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "ffn_type" in text and any(fragment in text for fragment in stale):
                violations.append(str(path.relative_to(ROOT)))
    assert not violations, "stale selector contracts: " + ", ".join(violations)


def test_model_configs_with_ffn_width_expose_selector() -> None:
    violations: list[str] = []
    for task in sorted(IMPACTED_TASKS):
        root = SRC / "tasks" / task / "configs/model"
        if not root.exists():
            continue
        for path in root.rglob("*.yaml"):
            text = path.read_text(encoding="utf-8")
            if "ffn_dim:" in text and "ffn_type:" not in text:
                violations.append(str(path.relative_to(ROOT)))
    assert not violations, "model configs without ffn_type: " + ", ".join(violations)
''',
    )


def _audit(paths: tuple[Path, ...]) -> None:
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(_read(path), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in BLOCK_CONFIG_NAMES:
                continue
            selector = next((kw.value for kw in node.keywords if kw.arg == "ffn_type"), None)
            if selector is None or isinstance(selector, ast.Constant):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    if violations:
        raise RuntimeError("Hardcoded direct block FFNs: " + ", ".join(violations))


def main() -> None:
    _install_shared_selector_contract()
    direct_paths = _direct_consumer_paths()
    wrapper_targets = _rewrite_direct_consumers(direct_paths)
    wrapper_targets = _propagate_through_wrappers(wrapper_targets)
    tasks = _impacted_tasks(direct_paths)
    _widen_impacted_task_sources(tasks)
    _normalize_configuration_contracts(tasks)
    _normalize_blcs_track_query_fields()
    _remove_plcs_ablation_ffn_pin()
    _add_model_yaml_selectors(tasks)
    _update_impacted_tests(tasks)
    _write_audit_test(wrapper_targets, tasks)
    _audit(direct_paths)


if __name__ == "__main__":
    main()
