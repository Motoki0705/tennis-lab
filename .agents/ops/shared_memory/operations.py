"""Operation specs and their validated, read-only application to a memory tree."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from .schema import (
    ENTRY_FIELDS,
    MEMORY_DIR,
    REQUIRED_KEYS,
    SHARED_LAYERS,
    is_entry_filename,
    iso_date,
    private_data_kinds,
    read_frontmatter,
)
from .tree import Checkout, EditedTree, Tree
from .validate import Problem, plan_sync

_DATES = ("created", "last_verified")
_CONTENT_FIELDS = REQUIRED_KEYS | {"body"}


def load_json(raw: str | bytes) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON keys must be unique")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise ValueError("JSON numbers must be finite")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)


@dataclass(frozen=True)
class Operation:
    op: str
    layer: str
    slug: str
    reason: str
    fields: dict[str, object]

    @property
    def path(self) -> str:
        return f"{MEMORY_DIR}/{self.layer}/{self.slug}.md"


def _operation(spec: Mapping[str, object], today: date) -> Operation:
    op = spec.get("op")
    if op not in ("add", "update", "delete") or not isinstance(op, str):
        raise ValueError("op must be add, update or delete")
    layer, slug = spec.get("layer"), spec.get("slug")
    if not isinstance(layer, str) or layer not in SHARED_LAYERS:
        raise ValueError("layer must be workflow, decision, environment or gotcha")
    if not isinstance(slug, str) or not is_entry_filename(f"{slug}.md"):
        raise ValueError("slug must be kebab-case and cannot be index")
    fields = {
        k: v for k, v in spec.items() if k not in {"op", "layer", "slug", "reason"}
    }
    if fields.keys() - (set() if op == "delete" else _CONTENT_FIELDS):
        raise ValueError(f"unknown {op} fields")
    reason = spec.get("reason", "共有memoryに観察を追加する。" if op == "add" else None)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]", reason):
        raise ValueError("reason contains invalid characters")
    if kinds := private_data_kinds(reason):
        raise ValueError(f"reason contains {', '.join(kinds)}")
    if op == "add":
        if missing := _CONTENT_FIELDS - fields.keys() - set(_DATES):
            raise ValueError(f"missing add fields: {', '.join(sorted(missing))}")
        for key in _DATES:
            fields.setdefault(key, today)
    if op == "update" and not fields:
        raise ValueError("update requires an entry field or body")
    for key in _DATES:
        if key in fields:
            fields[key] = iso_date(fields[key], key)
    if "body" in fields:
        body = fields["body"]
        if not isinstance(body, str) or re.search(r"[\0\ud800-\udfff]", body):
            raise ValueError("body must be valid Markdown text")
        fields["body"] = _body_text(body)
    return Operation(op, layer, slug, reason.strip(), deepcopy(fields))


def parse_operations(specs: object, today: date) -> tuple[Operation, ...]:
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)) or not specs:
        raise ValueError("batch requires a non-empty sequence of operations")
    operations = []
    paths = set()
    for i, spec in enumerate(specs, start=1):
        try:
            if not isinstance(spec, Mapping):
                raise ValueError("expected an operation object")
            operation = _operation(spec, today)
            if operation.path in paths:
                raise ValueError(f"duplicate operation for {operation.path}")
        except ValueError as exc:
            raise ValueError(f"operation {i}: {exc}") from None
        paths.add(operation.path)
        operations.append(operation)
    return tuple(operations)


def _body_text(body: str) -> str:
    return body.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def _edit_entry(tree: Tree, operation: Operation) -> bytes | None:
    path = operation.path
    kind = tree.kind(path, follow_links=False)
    if kind not in (None, "file"):
        raise ValueError("expected a regular entry file")
    if operation.op == "add" and kind is not None:
        raise ValueError("entry already exists")
    if operation.op != "add" and kind is None:
        raise ValueError("entry does not exist")
    if operation.op == "delete":
        return None

    fields = operation.fields.copy()
    body = fields.pop("body", None)
    data: dict[str, object]
    if operation.op == "add":
        data, previous_body = {}, ""
    else:
        data, previous_body = read_frontmatter(tree.read_bytes(path).decode("utf-8"))
        before = {key: data.get(key) for key in fields}
        for key in before.keys() & set(_DATES):
            if isinstance(value := before[key], str) and value == str(fields[key]):
                before[key] = fields[key]
        if fields == before and (body is None or body == _body_text(previous_body)):
            raise ValueError("update has no changes")
    data.update(fields)
    ordered = {key: data[key] for key in ENTRY_FIELDS if key in data}
    ordered.update({key: value for key, value in data.items() if key not in ordered})
    frontmatter = yaml.safe_dump(
        ordered, allow_unicode=True, sort_keys=False, width=4096
    )
    markdown = previous_body if body is None else f"\n\n{body}\n" if body else "\n"
    return f"---\n{frontmatter}---{markdown}".encode()


def validation_error(problems: Sequence[Problem]) -> ValueError:
    detail = "\n".join(f"{p.path}: [{p.code}] {p.message}" for p in problems)
    return ValueError(f"shared memory validation failed\n{detail}")


def plan_operations(
    tree: Tree, operations: Sequence[Operation], *, today: date
) -> dict[str, bytes | None]:
    directories = (".agents", MEMORY_DIR, *(f"{MEMORY_DIR}/{p}" for p in SHARED_LAYERS))
    for directory in directories:
        if tree.kind(directory, follow_links=False) != "directory":
            raise ValueError(f"{directory}: [layout] expected a directory")
    edits = {}
    for operation in operations:
        try:
            edits[operation.path] = _edit_entry(tree, operation)
        except UnicodeError:
            raise ValueError(f"{operation.path}: expected UTF-8 text") from None
        except ValueError as exc:
            raise ValueError(f"{operation.path}: {exc}") from None
    generated, problems = plan_sync(EditedTree(tree, edits), today=today)
    if problems:
        raise validation_error(problems)
    return {**edits, **generated}


def apply_operations(
    source: Path | Tree,
    operations: Sequence[Mapping[str, object]],
    *,
    today: date | None = None,
) -> dict[str, bytes | None]:
    """Return a validated batch of repo-relative file edits without writing to source."""
    today = today or date.today()
    tree = Checkout(source) if isinstance(source, Path) else source
    return plan_operations(tree, parse_operations(operations, today), today=today)
