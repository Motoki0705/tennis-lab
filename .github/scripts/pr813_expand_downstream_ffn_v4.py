from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
FFN_MODULE = "src.utils.models.components.ffn_layers"
TASKS = {"ball_detection", "blcs", "court_detection", "plcs", "slcs"}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
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


def _ensure_import(text: str, name: str) -> str:
    pattern = re.compile(
        rf"from\s+{re.escape(FFN_MODULE)}\s+import"
        rf"(?:\s*\([^)]*\b{name}\b[^)]*\)|[^\n]*\b{name}\b)",
        flags=re.DOTALL,
    )
    if pattern.search(text) is not None:
        return text
    line = f"from {FFN_MODULE} import {name}\n"
    marker = "from __future__ import annotations\n"
    if marker in text:
        return text.replace(marker, marker + "\n" + line, 1)
    return line + "\n" + text


def _source(text: str, node: ast.AST) -> str:
    result = ast.get_source_segment(text, node)
    if result is None:
        raise RuntimeError(f"Unable to recover source for {type(node).__name__}")
    return result


def _derive_selector(text: str, node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return "ffn_type" if node.id == "ffn_dim" else None
    if isinstance(node, ast.Attribute):
        if node.attr != "ffn_dim":
            return None
        return f"{_source(text, node.value)}.ffn_type"
    if isinstance(node, ast.Subscript):
        source = _source(text, node)
        replaced = re.sub(
            r"(?P<quote>['\"])ffn_dim(?P=quote)\s*\]$",
            lambda match: f'{match.group("quote")}ffn_type{match.group("quote")}]',
            source,
        )
        return replaced if replaced != source else None
    if isinstance(node, ast.Call):
        for argument in reversed(node.args):
            selector = _derive_selector(text, argument)
            if selector is not None:
                return selector
        for keyword in reversed(node.keywords):
            selector = _derive_selector(text, keyword.value)
            if selector is not None:
                return selector
    if isinstance(node, ast.UnaryOp):
        return _derive_selector(text, node.operand)
    return None


def _insert_keyword(text: str, call: ast.Call, expression: str) -> str:
    offsets = _offsets(text)
    width = next((keyword for keyword in call.keywords if keyword.arg == "ffn_dim"), None)
    if width is None:
        raise RuntimeError(f"Config construction lacks ffn_dim at line {call.lineno}")
    value_end = offsets[width.value.end_lineno - 1] + width.value.end_col_offset
    call_end = offsets[call.end_lineno - 1] + call.end_col_offset
    comma = text.find(",", value_end, call_end)
    if comma < 0:
        raise RuntimeError(f"Cannot locate ffn_dim comma at line {call.lineno}")
    line_start = text.rfind("\n", 0, offsets[width.value.lineno - 1]) + 1
    indent = re.match(r"\s*", text[line_start:]).group(0)
    return (
        text[: comma + 1]
        + f"\n{indent}ffn_type={expression},"
        + text[comma + 1 :]
    )


def _config_paths() -> tuple[Path, ...]:
    result: list[Path] = []
    for task in sorted(TASKS):
        result.extend(sorted((SRC / "tasks" / task).glob("configuration*.py")))
    return tuple(result)


def _config_classes(path: Path) -> set[str]:
    tree = ast.parse(_read(path), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        field_names = {
            field.target.id
            for field in node.body
            if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)
        }
        if {"ffn_dim", "ffn_type"} <= field_names:
            result.add(node.name)
    return result


def _repair_config_constructor_values(paths: tuple[Path, ...]) -> set[str]:
    config_classes: set[str] = set()
    for path in paths:
        config_classes.update(_config_classes(path))
    for path in paths:
        text = _read(path)
        while True:
            tree = ast.parse(text, filename=str(path))
            changed = False
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call) or _call_name(call) not in config_classes:
                    continue
                if any(keyword.arg == "ffn_type" for keyword in call.keywords):
                    continue
                width = next(
                    (keyword.value for keyword in call.keywords if keyword.arg == "ffn_dim"),
                    None,
                )
                if width is None:
                    continue
                selector = _derive_selector(text, width)
                if selector is None:
                    raise RuntimeError(
                        f"Cannot derive ffn_type for {path}:{call.lineno}"
                    )
                if "[" in selector:
                    selector = f'resolve_ffn_type(cast("str", {selector}))'
                    text = _ensure_import(text, "resolve_ffn_type")
                    tree = ast.parse(text, filename=str(path))
                    call = next(
                        candidate
                        for candidate in ast.walk(tree)
                        if isinstance(candidate, ast.Call)
                        and candidate.lineno == call.lineno
                        and _call_name(candidate) in config_classes
                        and not any(
                            keyword.arg == "ffn_type" for keyword in candidate.keywords
                        )
                    )
                text = _insert_keyword(text, call, selector)
                changed = True
                break
            if not changed:
                break
        _write(path, text)
    return config_classes


def _assignment_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    current: ast.AST | None = node
    while current is not None:
        parent = parents.get(current)
        if isinstance(parent, ast.Assign):
            target = parent.targets[0]
            return target.id if isinstance(target, ast.Name) else None
        if isinstance(parent, ast.AnnAssign):
            return parent.target.id if isinstance(parent.target, ast.Name) else None
        if isinstance(parent, ast.Call):
            return None
        current = parent
    return None


def _augment_required_key_collections(paths: tuple[Path, ...]) -> None:
    for path in paths:
        text = _read(path)
        while True:
            tree = ast.parse(text, filename=str(path))
            parents: dict[ast.AST, ast.AST] = {}
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    parents[child] = node
            changed = False
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                    continue
                values = {
                    element.value
                    for element in node.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                }
                if "ffn_dim" not in values or "ffn_type" in values:
                    continue
                assignment = (_assignment_name(node, parents) or "").lower()
                if any(
                    token in assignment
                    for token in ("positive", "integer", "numeric", "dimension")
                ):
                    continue
                parent = parents.get(node)
                if isinstance(parent, ast.BinOp):
                    assignment = (_assignment_name(parent, parents) or assignment).lower()
                if assignment and not any(
                    token in assignment
                    for token in ("field", "key", "required", "allowed", "model", "common")
                ):
                    continue
                ffn_dim_node = next(
                    element
                    for element in node.elts
                    if isinstance(element, ast.Constant) and element.value == "ffn_dim"
                )
                offsets = _offsets(text)
                line_start = text.rfind("\n", 0, offsets[ffn_dim_node.lineno - 1]) + 1
                indent = re.match(r"\s*", text[line_start:]).group(0)
                end = offsets[ffn_dim_node.end_lineno - 1] + ffn_dim_node.end_col_offset
                comma = text.find(",", end)
                if comma < 0:
                    continue
                text = (
                    text[: comma + 1]
                    + f'\n{indent}"ffn_type",'
                    + text[comma + 1 :]
                )
                changed = True
                break
            if not changed:
                break
        _write(path, text)


def _append_audit(config_classes: set[str]) -> None:
    path = ROOT / "tests/unit/utils/models/components/test_ffn_downstream_configuration.py"
    text = _read(path)
    if "test_config_dataclass_construction_preserves_ffn_selector" in text:
        return
    classes = repr(sorted(config_classes))
    text += f'''

CONFIG_CLASSES = set({classes})


def test_config_dataclass_construction_preserves_ffn_selector() -> None:
    violations: list[str] = []
    for task in sorted(IMPACTED_TASKS):
        for path in (SRC / "tasks" / task).glob("configuration*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or _call_name(node) not in CONFIG_CLASSES:
                    continue
                if not any(keyword.arg == "ffn_dim" for keyword in node.keywords):
                    continue
                if not any(keyword.arg == "ffn_type" for keyword in node.keywords):
                    violations.append(f"{{path.relative_to(ROOT)}}:{{node.lineno}}")
    assert not violations, "config constructors drop ffn_type: " + ", ".join(violations)
'''
    _write(path, text)


def _audit(paths: tuple[Path, ...], config_classes: set[str]) -> None:
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(_read(path), filename=str(path))
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or _call_name(call) not in config_classes:
                continue
            if not any(keyword.arg == "ffn_dim" for keyword in call.keywords):
                continue
            if not any(keyword.arg == "ffn_type" for keyword in call.keywords):
                violations.append(f"{path.relative_to(ROOT)}:{call.lineno}")
    if violations:
        raise RuntimeError("Config constructors drop ffn_type: " + ", ".join(violations))


def main() -> None:
    paths = _config_paths()
    _augment_required_key_collections(paths)
    config_classes = _repair_config_constructor_values(paths)
    _append_audit(config_classes)
    _audit(paths, config_classes)


if __name__ == "__main__":
    main()
