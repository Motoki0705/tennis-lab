from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _ensure_import(text: str, names: tuple[str, ...]) -> str:
    module = "src.utils.models.components.ffn_layers"
    required = [name for name in names if re.search(rf"\b{name}\b", text)]
    missing = [
        name
        for name in required
        if not re.search(
            rf"from {re.escape(module)} import [^\n]*\b{name}\b",
            text,
        )
    ]
    if not missing:
        return text
    import_line = f"from {module} import {', '.join(missing)}\n"
    future = "from __future__ import annotations\n"
    if future in text:
        return text.replace(future, future + "\n" + import_line, 1)
    return import_line + "\n" + text


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


def _function_parameters(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    return {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }


def _widen_text_contract(text: str) -> str:
    text = text.replace('Literal["swiglu", "mlp"]', "FFNType")
    text = text.replace("Literal['swiglu', 'mlp']", "FFNType")
    text = text.replace(
        'cast("Literal[\'swiglu\', \'mlp\']",',
        'cast("FFNType",',
    )
    text = re.sub(
        r"not in \{\s*['\"]swiglu['\"]\s*,\s*['\"]mlp['\"]\s*\}",
        "not in SUPPORTED_FFN_TYPES",
        text,
    )
    text = re.sub(
        r"in \{\s*['\"]swiglu['\"]\s*,\s*['\"]mlp['\"]\s*\}",
        "in SUPPORTED_FFN_TYPES",
        text,
    )
    return text


def _install_shared_contract() -> None:
    path = SRC / "utils/models/components/ffn_layers.py"
    text = path.read_text(encoding="utf-8")
    if "SUPPORTED_FFN_TYPES" not in text:
        marker = "]\n\n\ndef default_ffn_dim"
        replacement = "]\n\nSUPPORTED_FFN_TYPES: frozenset[str] = frozenset(\n    {\n        \"swiglu\",\n        \"mlp\",\n        \"kimi_k3_situglu\",\n        \"deepseek_v4_swiglu\",\n        \"gpt_oss_swiglu\",\n    }\n)\n\n\ndef resolve_ffn_type(value: str) -> FFNType:\n    \"\"\"Validate and narrow one externally supplied FFN selector.\"\"\"\n    if value not in SUPPORTED_FFN_TYPES:\n        supported = \", \".join(sorted(SUPPORTED_FFN_TYPES))\n        raise ValueError(\n            f\"Unsupported ffn_type={value!r}; expected one of: {supported}\"\n        )\n    return cast(FFNType, value)\n\n\ndef default_ffn_dim"
        if marker not in text:
            raise RuntimeError("ffn_layers shared-contract marker was not found")
        text = text.replace(marker, replacement, 1)
        _write(path, text)

    path = SRC / "utils/models/components/__init__.py"
    text = path.read_text(encoding="utf-8")
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


def _direct_consumers() -> tuple[Path, ...]:
    return tuple(
        path
        for path in SRC.rglob("*.py")
        if path.name != "block.py"
        and (
            "TransformerBlockConfig" in path.read_text(encoding="utf-8")
            or "CrossAttnBlockConfig" in path.read_text(encoding="utf-8")
        )
    )


def _make_block_calls_configurable(path: Path) -> set[str]:
    text = _widen_text_contract(path.read_text(encoding="utf-8"))
    tree = ast.parse(text, filename=str(path))
    parents = _parent_map(tree)
    offsets = _offsets(text)
    changes: list[tuple[int, int, str]] = []
    functions_needing_parameter: set[ast.FunctionDef | ast.AsyncFunctionDef] = set()
    class_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(
                isinstance(candidate, ast.Call)
                and _call_name(candidate)
                in {"TransformerBlockConfig", "CrossAttnBlockConfig"}
                for candidate in ast.walk(node)
            ):
                class_names.add(node.name)
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) not in {
            "TransformerBlockConfig",
            "CrossAttnBlockConfig",
        }:
            continue
        selector = next(
            (keyword for keyword in node.keywords if keyword.arg == "ffn_type"),
            None,
        )
        if selector is None:
            raise RuntimeError(f"missing ffn_type in {path}:{node.lineno}")
        if not (
            isinstance(selector.value, ast.Constant)
            and selector.value.value in {"swiglu", "mlp"}
        ):
            continue
        function = _enclosing_function(node, parents)
        if function is None:
            raise RuntimeError(f"block config outside a function in {path}:{node.lineno}")
        if "ffn_type" not in _function_parameters(function):
            functions_needing_parameter.add(function)
        start = offsets[selector.value.lineno - 1] + selector.value.col_offset
        end = offsets[selector.value.end_lineno - 1] + selector.value.end_col_offset
        changes.append((start, end, "ffn_type"))

    for function in functions_needing_parameter:
        header_start = offsets[function.lineno - 1] + function.col_offset
        body_start = offsets[function.body[0].lineno - 1] + function.body[0].col_offset
        header = text[header_start:body_start]
        close_relative = header.rfind(")")
        if close_relative < 0:
            raise RuntimeError(f"cannot locate signature end in {path}:{function.lineno}")
        close = header_start + close_relative
        if "\n" in header:
            parameter_lines = [
                line for line in header[:close_relative].splitlines()[1:] if line.strip()
            ]
            indent = (
                re.match(r"\s*", parameter_lines[-1]).group(0)
                if parameter_lines
                else "        "
            )
            payload = f'{indent}ffn_type: FFNType = "swiglu",\n'
        else:
            payload = ', ffn_type: FFNType = "swiglu"'
        changes.append((close, close, payload))

    for start, end, replacement in sorted(changes, reverse=True):
        text = text[:start] + replacement + text[end:]
    text = _ensure_import(text, ("FFNType",))
    _write(path, text)
    return class_names


def _forward_new_constructor_parameters(class_names: set[str]) -> None:
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        offsets = _offsets(text)
        insertions: list[tuple[int, int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in class_names:
                continue
            if any(keyword.arg == "ffn_type" for keyword in node.keywords):
                continue
            width_keyword = next(
                (keyword for keyword in node.keywords if keyword.arg == "ffn_dim"),
                None,
            )
            if width_keyword is None:
                continue
            if isinstance(width_keyword.value, ast.Name):
                value = "ffn_type"
            elif isinstance(width_keyword.value, ast.Attribute):
                base = ast.get_source_segment(text, width_keyword.value.value)
                if base is None:
                    continue
                value = f"{base}.ffn_type"
            else:
                continue
            value_end = offsets[width_keyword.value.end_lineno - 1] + width_keyword.value.end_col_offset
            comma = text.find(",", value_end)
            if comma < 0:
                raise RuntimeError(f"cannot locate ffn_dim comma in {path}:{node.lineno}")
            line_start = text.rfind("\n", 0, offsets[width_keyword.value.lineno - 1]) + 1
            indent = re.match(r"\s*", text[line_start:]).group(0)
            insertions.append((comma + 1, comma + 1, f"\n{indent}ffn_type={value},"))
        for start, end, replacement in sorted(insertions, reverse=True):
            text = text[:start] + replacement + text[end:]
        if insertions:
            _write(path, text)


def _widen_task_contracts(consumers: tuple[Path, ...]) -> set[str]:
    tasks = {
        path.relative_to(SRC).parts[1]
        for path in consumers
        if path.relative_to(SRC).parts[0] == "tasks"
    }
    for task in sorted(tasks):
        for path in (SRC / "tasks" / task).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "ffn_type" not in text:
                continue
            widened = _widen_text_contract(text)
            if widened != text:
                widened = _ensure_import(
                    widened,
                    ("FFNType", "SUPPORTED_FFN_TYPES"),
                )
                _write(path, widened)
    return tasks


def _expose_blcs_track_query_selector() -> None:
    path = SRC / "tasks/blcs/configuration.py"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"(class TrackQuery(?:Reference)?(?:Ablation)?ModelConfig:\n(?:.*\n){0,24}?\s+ffn_dim: int\n)(?!\s+ffn_type:)",
        r"\1    ffn_type: FFNType\n",
        text,
    )
    text = text.replace(
        '            "ffn_dim",\n            "num_queries",',
        '            "ffn_dim",\n            "ffn_type",\n            "num_queries",',
    )
    text = text.replace(
        '                "ffn_dim": int,\n                "num_queries": int,',
        '                "ffn_dim": int,\n                "ffn_type": str,\n                "num_queries": int,',
    )
    text = re.sub(
        r'(\s+ffn_dim=[^\n]*model\["ffn_dim"\][^\n]*,\n)(?!\s+ffn_type=)',
        r'\1            ffn_type=cast("FFNType", model["ffn_type"]),\n',
        text,
    )
    text = _ensure_import(text, ("FFNType", "SUPPORTED_FFN_TYPES"))
    _write(path, text)

    config_root = SRC / "tasks/blcs/configs/model"
    for yaml_path in config_root.rglob("*.yaml"):
        yaml = yaml_path.read_text(encoding="utf-8")
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
            raise RuntimeError(f"missing ffn_dim in {yaml_path}")
        _write(yaml_path, yaml)


def _remove_plcs_standard_only_ablation_rule() -> None:
    path = SRC / "tasks/plcs/configuration.py"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r'(\s+if name in _TRACK_QUERY_ABLATION_MODEL_NAMES:\n)'
        r'\s+if _string\(mapping, "ffn_type", path="model"\) != "swiglu":\n'
        r'\s+raise SemanticConfigurationError\(\n'
        r'\s+"PLCS track-query ablation requires "\n'
        r'\s+"model\.ffn_type=\'swiglu\'\."\n'
        r'\s+\)\n',
        r"\1",
        text,
    )
    _write(path, text)


def _write_contract_test() -> None:
    path = ROOT / "tests/unit/utils/models/components/test_ffn_downstream_configuration.py"
    _write(
        path,
        '''"""Repository-wide configuration propagation for TransformerBlock FFNs."""

from __future__ import annotations

import ast
from pathlib import Path

from src.utils.models.components.ffn_layers import SUPPORTED_FFN_TYPES

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _block_consumer_files() -> tuple[Path, ...]:
    return tuple(
        path
        for path in SOURCE_ROOT.rglob("*.py")
        if path.name != "block.py"
        and (
            "TransformerBlockConfig" in path.read_text(encoding="utf-8")
            or "CrossAttnBlockConfig" in path.read_text(encoding="utf-8")
        )
    )


def test_supported_ffn_type_contract_contains_all_architecture_variants() -> None:
    assert SUPPORTED_FFN_TYPES == {
        "swiglu",
        "mlp",
        "kimi_k3_situglu",
        "deepseek_v4_swiglu",
        "gpt_oss_swiglu",
    }


def test_block_consumers_forward_a_configurable_ffn_selector() -> None:
    violations: list[str] = []
    for path in _block_consumer_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if call_name not in {"TransformerBlockConfig", "CrossAttnBlockConfig"}:
                continue
            selector = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "ffn_type"),
                None,
            )
            if selector is None or isinstance(selector, ast.Constant):
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
                continue
            current: ast.AST | None = node
            function: ast.FunctionDef | ast.AsyncFunctionDef | None = None
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function = current
                    break
                current = parents.get(current)
            if function is None:
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
                continue
            parameters = {
                argument.arg
                for argument in (
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                )
            }
            if "ffn_type" not in parameters:
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
    assert not violations, "non-configurable block FFN selectors: " + ", ".join(violations)


def test_consumer_task_sources_have_no_stale_two_option_contract() -> None:
    tasks = {
        path.relative_to(SOURCE_ROOT).parts[1]
        for path in _block_consumer_files()
        if path.relative_to(SOURCE_ROOT).parts[0] == "tasks"
    }
    violations: list[str] = []
    stale_fragments = (
        'Literal["swiglu", "mlp"]',
        "Literal['swiglu', 'mlp']",
        '{"swiglu", "mlp"}',
        "{'swiglu', 'mlp'}",
    )
    for task in sorted(tasks):
        for path in (SOURCE_ROOT / "tasks" / task).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "ffn_type" in text and any(fragment in text for fragment in stale_fragments):
                violations.append(str(path.relative_to(REPOSITORY_ROOT)))
    assert not violations, "stale two-option FFN contracts: " + ", ".join(violations)
''',
    )


def _audit(consumers: tuple[Path, ...]) -> None:
    violations: list[str] = []
    for path in consumers:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) not in {"TransformerBlockConfig", "CrossAttnBlockConfig"}:
                continue
            selector = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "ffn_type"),
                None,
            )
            if selector is None or isinstance(selector, ast.Constant):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    if violations:
        raise RuntimeError("non-configurable block selectors: " + ", ".join(violations))


def main() -> None:
    _install_shared_contract()
    consumers = _direct_consumers()
    class_names: set[str] = set()
    for consumer in consumers:
        class_names.update(_make_block_calls_configurable(consumer))
    _forward_new_constructor_parameters(class_names)
    _widen_task_contracts(consumers)
    _expose_blcs_track_query_selector()
    _remove_plcs_standard_only_ablation_rule()
    _write_contract_test()
    _audit(consumers)


if __name__ == "__main__":
    main()
