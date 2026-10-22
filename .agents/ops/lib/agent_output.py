"""Pin and validate structured output without interpreting assistant prose."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agent_events import parse_json


class OutputSchema:
    def __init__(self, path: Path) -> None:
        # Keep ordinary text/profile runs independent of this optional feature.
        try:
            from jsonschema import (  # type: ignore[import-untyped]
                SchemaError,
                validators,
            )
            from referencing import Registry
        except ImportError as exc:
            raise ValueError(
                "--json-schema requires jsonschema; sync project dependencies"
            ) from exc

        self.raw = path.read_bytes()
        schema = parse_json(self.raw.decode("utf-8"))
        if not isinstance(schema, dict):
            raise ValueError("--json-schema must contain a JSON Schema object")
        validator = validators.validator_for(schema, default=None)
        if "$schema" not in schema:
            validator = validators.Draft202012Validator
        if validator is None:
            raise ValueError("--json-schema declares an unsupported schema dialect")
        try:
            validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError("invalid --json-schema: " + exc.message) from exc
        # External resources are neither pinned by this digest nor fetched.
        self.validator = validator(schema, registry=Registry())
        self.sha256 = hashlib.sha256(self.raw).hexdigest()

    def validate(self, value: Any) -> None:
        from jsonschema import ValidationError
        from referencing.exceptions import Unresolvable

        try:
            self.validator.validate(value)
        except ValidationError as exc:
            raise ValueError(
                "structured output does not match --json-schema: " + exc.message
            ) from exc
        except Unresolvable as exc:
            raise ValueError(
                "structured output schema reference is not pinned/resolvable"
            ) from exc
