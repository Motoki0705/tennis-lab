"""Build deterministic, code-free annotation requirements and clip attachments."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .runtime.contracts import KIT_VERSION, Annotation, sha256_file, write_json

KIT_CONTENT_FILES = ("PROTOCOL.md", "annotation.schema.json", "court_definition.json")
CLIP_KIT_FILES = (*KIT_CONTENT_FILES, "kit_manifest.json")


def build_kit(root: Path) -> tuple[Path, str]:
    from src.utils.schema.court import (
        COURT_KP_NAMES,
        COURT_SKELETON,
        STANDARD_COURT_CONFIG,
        court_keypoints_3d,
    )

    resources = Path(__file__).parent / "resources"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=root) as directory:
        stage = Path(directory)
        write_json(stage / "annotation.schema.json", Annotation.model_json_schema())
        write_json(
            stage / "court_definition.json",
            {
                "contract_id": "physical_courtkp20_v1",
                "names": list(COURT_KP_NAMES),
                "points_xyz": court_keypoints_3d(STANDARD_COURT_CONFIG).tolist(),
                "skeleton": COURT_SKELETON,
                "homography_completion_indices": list(range(15)),
            },
        )
        protocol = (
            (resources / "PROTOCOL.md")
            .read_text(encoding="utf-8")
            .replace("{{KIT_VERSION}}", KIT_VERSION)
        )
        (stage / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
        files = {name: sha256_file(stage / name) for name in sorted(KIT_CONTENT_FILES)}
        kit_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        write_json(
            stage / "kit_manifest.json",
            {"kit_version": KIT_VERSION, "kit_id": kit_id, "files": files},
        )
        destination = root / kit_id[:16]
        if destination.exists():
            if {file.name for file in destination.iterdir()} != set(CLIP_KIT_FILES):
                raise ValueError(
                    f"existing annotation kit file set modified: {destination}"
                )
            for file in stage.iterdir():
                if not (destination / file.name).is_file() or sha256_file(
                    destination / file.name
                ) != sha256_file(file):
                    raise ValueError(
                        f"existing annotation kit modified: {destination / file.name}"
                    )
        else:
            stage.rename(destination)
    # Convenient local copies; neither Project instructions nor this parent folder
    # is required to use a clip. Only resources/ is maintained by hand.
    for name in ("PROTOCOL.md", "PROJECT_INSTRUCTIONS.txt"):
        value = (
            (resources / name)
            .read_text(encoding="utf-8")
            .replace("{{KIT_VERSION}}", KIT_VERSION)
        )
        path = root / name
        if not path.exists() or path.read_text(encoding="utf-8") != value:
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_text(value, encoding="utf-8")
            temporary.replace(path)
    return destination, kit_id


def copy_clip_kit(kit_directory: Path, destination: Path) -> None:
    """Copy regular files so selecting a clip never depends on its parent kit."""
    for name in CLIP_KIT_FILES:
        shutil.copyfile(kit_directory / name, destination / name)
