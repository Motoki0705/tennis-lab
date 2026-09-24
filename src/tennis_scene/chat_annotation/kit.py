"""Build the embedded prompt definitions and the two project texts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .prompt import write_request
from .runtime.contracts import KIT_VERSION, BallAnnotation, PlayerAnnotation, Policies


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

    schemas = {
        "ball_detection": BallAnnotation.model_json_schema(),
        "player_detection": PlayerAnnotation.model_json_schema(),
    }
    contents: dict[str, bytes] = {
        f"{target}_annotation.schema.json": _json_bytes(schema)
        for target, schema in schemas.items()
    }
    project_kits: dict[str, dict[str, bytes]] = {}
    for target in ("ball_detection", "player_detection"):
        target_resources = resources / target
        target_contents = {
            "PROJECT_INSTRUCTIONS.txt": (target_resources / "PROJECT_INSTRUCTIONS.txt")
            .read_text(encoding="utf-8")
            .replace("{{KIT_VERSION}}", KIT_VERSION)
            .encode("utf-8"),
            "REQUEST.txt": (target_resources / "REQUEST.txt")
            .read_text(encoding="utf-8")
            .replace("{{KIT_VERSION}}", KIT_VERSION)
            .replace(
                "{{BALL_MAX_GAP_SECONDS}}", str(policies.ball_max_gap_seconds)
            )
            .encode("utf-8"),
        }
        project_kits[target] = target_contents
        contents[f"{target}_REQUEST.txt"] = target_contents["REQUEST.txt"]
    contents.update(
        {
            "PROTOCOL.md": resource("PROTOCOL.md"),
        }
    )
    kit_files = dict(contents)
    kit_files.update(
        {
            f"project_kits/{target}/{name}": value
            for target, target_contents in project_kits.items()
            for name, value in target_contents.items()
        }
    )
    files = {
        name: hashlib.sha256(kit_files[name]).hexdigest()
        for name in sorted(kit_files)
    }
    kit_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    contents["kit_manifest.json"] = _json_bytes(
        {"kit_version": KIT_VERSION, "kit_id": kit_id, "files": files}
    )
    root.mkdir(parents=True, exist_ok=True)
    write_request(root.parent, contents, project_directory=root)
    for target, project_files in project_kits.items():
        directory = root / target
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in project_files.items():
            path = directory / name
            if not path.exists() or path.read_bytes() != value:
                temporary = path.with_suffix(path.suffix + ".partial")
                temporary.write_bytes(value)
                temporary.replace(path)
    return contents, kit_id
