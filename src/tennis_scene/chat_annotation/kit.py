"""Build the embedded prompt definitions and the two project texts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .prompt import write_request
from .runtime.contracts import KIT_VERSION, Annotation, Policies


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def build_kit(root: Path, policies: Policies) -> tuple[dict[str, bytes], str]:
    resources = Path(__file__).parent / "resources"

    def resource(name: str) -> bytes:
        return (
            (resources / name)
            .read_text(encoding="utf-8")
            .replace("{{KIT_VERSION}}", KIT_VERSION)
            .replace("{{BALL_MAX_GAP_SECONDS}}", str(policies.ball_max_gap_seconds))
            .encode("utf-8")
        )

    contents = {
        "PROTOCOL.md": resource("PROTOCOL.md"),
        "annotation.schema.json": _json_bytes(Annotation.model_json_schema()),
    }
    files = {
        name: hashlib.sha256(contents[name]).hexdigest() for name in sorted(contents)
    }
    kit_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    contents["kit_manifest.json"] = _json_bytes(
        {"kit_version": KIT_VERSION, "kit_id": kit_id, "files": files}
    )
    root.mkdir(parents=True, exist_ok=True)
    write_request(root.parent, contents, project_directory=root)
    for name in ("PROJECT_INSTRUCTIONS.txt",):
        value = resource(name)
        path = root / name
        if not path.exists() or path.read_bytes() != value:
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(value)
            temporary.replace(path)
    return contents, kit_id
