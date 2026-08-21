"""Safe loading and optional reduction of local Codex rollout-trace bundles."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

SUPPORTED_TRACE_SCHEMA_VERSIONS = frozenset({1})


class TraceBundleError(RuntimeError):
    """Raised when a rollout-trace bundle is missing or internally inconsistent."""


class TraceBundle:
    """Reduced state plus lazy, bundle-confined raw payload access."""

    def __init__(self, bundle_path: Path, state: dict[str, Any]) -> None:
        self.path = bundle_path.resolve()
        self.state = state
        self._payload_cache: dict[str, Any] = {}
        self._validate_state()

    @classmethod
    def load(
        cls,
        source: Path,
        *,
        auto_reduce: bool = True,
        codex_binary: str = "codex",
    ) -> TraceBundle:
        """Load a bundle directory or reduced state file.

        A missing ``state.json`` is reduced with the installed Codex CLI unless
        ``auto_reduce`` is disabled. Reduction failures are surfaced verbatim.
        """

        resolved = source.expanduser().resolve()
        if resolved.is_file():
            state_path = resolved
            bundle_path = resolved.parent
        elif resolved.is_dir():
            bundle_path = resolved
            state_path = bundle_path / "state.json"
        else:
            raise TraceBundleError(f"trace source does not exist: {resolved}")

        if not state_path.is_file():
            if not auto_reduce:
                raise TraceBundleError(
                    f"reduced trace is missing: {state_path}; run "
                    f"'codex debug trace-reduce {bundle_path}'"
                )
            _reduce_bundle(bundle_path, codex_binary=codex_binary)
        try:
            raw_state: Any = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TraceBundleError(
                f"cannot read reduced trace {state_path}: {exc}"
            ) from exc
        if not isinstance(raw_state, dict):
            raise TraceBundleError(f"reduced trace must be a JSON object: {state_path}")
        return cls(bundle_path, raw_state)

    def payload_json(self, raw_payload_id: str) -> Any:
        """Load one referenced JSON payload while rejecting path traversal."""

        if raw_payload_id in self._payload_cache:
            return self._payload_cache[raw_payload_id]
        refs = self.state["raw_payloads"]
        ref = refs.get(raw_payload_id)
        if not isinstance(ref, dict):
            raise TraceBundleError(f"unknown raw payload id: {raw_payload_id}")
        relative_path = ref.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise TraceBundleError(f"raw payload {raw_payload_id} has no path")
        payload_path = (self.path / relative_path).resolve()
        try:
            payload_path.relative_to(self.path)
        except ValueError as exc:
            raise TraceBundleError(
                f"raw payload {raw_payload_id} escapes the trace bundle: {relative_path}"
            ) from exc
        if not payload_path.is_file():
            raise TraceBundleError(
                f"raw payload {raw_payload_id} is missing: {payload_path}"
            )
        try:
            payload: Any = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TraceBundleError(
                f"cannot read raw payload {raw_payload_id}: {exc}"
            ) from exc
        self._payload_cache[raw_payload_id] = payload
        return payload

    def _validate_state(self) -> None:
        required_scalars = {
            "schema_version": int,
            "trace_id": str,
            "rollout_id": str,
            "status": str,
            "root_thread_id": str,
        }
        for field, expected_type in required_scalars.items():
            value = self.state.get(field)
            if not isinstance(value, expected_type):
                raise TraceBundleError(
                    f"reduced trace field {field!r} must be {expected_type.__name__}"
                )
        schema_version = self.state["schema_version"]
        if schema_version not in SUPPORTED_TRACE_SCHEMA_VERSIONS:
            supported = ", ".join(map(str, sorted(SUPPORTED_TRACE_SCHEMA_VERSIONS)))
            raise TraceBundleError(
                f"unsupported rollout-trace schema_version={schema_version}; "
                f"supported: {supported}"
            )
        for field in (
            "conversation_items",
            "inference_calls",
            "code_cells",
            "tool_calls",
            "terminal_operations",
            "raw_payloads",
        ):
            if not isinstance(self.state.get(field), dict):
                raise TraceBundleError(
                    f"reduced trace field {field!r} must be an object"
                )


def _reduce_bundle(bundle_path: Path, *, codex_binary: str) -> None:
    manifest = bundle_path / "manifest.json"
    events = bundle_path / "trace.jsonl"
    if not manifest.is_file() or not events.is_file():
        raise TraceBundleError(
            f"not a rollout-trace bundle: expected {manifest.name} and {events.name} "
            f"under {bundle_path}"
        )
    try:
        result = subprocess.run(
            [codex_binary, "debug", "trace-reduce", str(bundle_path)],
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TraceBundleError(f"failed to run Codex trace reducer: {exc}") from exc
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "no output"
        raise TraceBundleError(
            f"Codex trace reducer exited with {result.returncode}: {details}"
        )
    if not (bundle_path / "state.json").is_file():
        raise TraceBundleError(
            "Codex trace reducer succeeded but state.json is missing"
        )
