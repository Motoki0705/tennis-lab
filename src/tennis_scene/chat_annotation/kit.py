"""Build the embedded prompt definitions and the two project texts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .prompt import write_request
from .runtime.contracts import KIT_VERSION, Annotation


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def build_kit(root: Path) -> tuple[dict[str, bytes], str]:
    from src.utils.schema.court import (
        COURT_KP_NAMES,
        COURT_SKELETON,
        STANDARD_COURT_CONFIG,
        court_keypoints_3d,
    )

    resources = Path(__file__).parent / "resources"

    def resource(name: str) -> bytes:
        return (
            (resources / name)
            .read_text(encoding="utf-8")
            .replace("{{KIT_VERSION}}", KIT_VERSION)
            .encode("utf-8")
        )

    contents = {
        "PROTOCOL.md": resource("PROTOCOL.md"),
        "annotation.schema.json": _json_bytes(Annotation.model_json_schema()),
        "court_definition.json": _json_bytes(
            {
                "contract_id": "physical_courtkp20_v1",
                "names": list(COURT_KP_NAMES),
                "points_xyz": court_keypoints_3d(STANDARD_COURT_CONFIG).tolist(),
                "skeleton": COURT_SKELETON,
                "homography_completion_indices": list(range(15)),
            }
        ),
    }
    files = {
        name: hashlib.sha256(contents[name]).hexdigest() for name in sorted(contents)
    }
    kit_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    contents["kit_manifest.json"] = _json_bytes(
        {"kit_version": KIT_VERSION, "kit_id": kit_id, "files": files}
    )
    root.mkdir(parents=True, exist_ok=True)
    for name in ("PROJECT_INSTRUCTIONS.txt",):
        value = resource(name)
        path = root / name
        if not path.exists() or path.read_bytes() != value:
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(value)
            temporary.replace(path)
    write_request(root.parent, contents, project_directory=root)
    return contents, kit_id
