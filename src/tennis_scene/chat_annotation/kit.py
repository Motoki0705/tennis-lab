"""Build deterministic Project files from the only maintained runtime sources."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

from .runtime.contracts import KIT_VERSION, Annotation, sha256_file, write_json

_BOOTSTRAP = '''"""Generated annotation kit. Maintained sources with relocated imports; do not edit.
Use --extract-code DIRECTORY to inspect the exact Python sources.
Requires Python >=3.11, pydantic>=2, numpy, opencv-python and av>=18.
"""
import base64
import io
from pathlib import Path
import sys
import tempfile
import zipfile

PAYLOAD = __PAYLOAD__

def run():
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
        if len(sys.argv) == 3 and sys.argv[1] == "--extract-code":
            archive.extractall(sys.argv[2])
            return 0
        with tempfile.TemporaryDirectory(prefix="tennis-annotation-") as directory:
            archive.extractall(directory)
            sys.path.insert(0, directory)
            try:
                from _tennis_annotation.cli import main
                return main(Path(__file__).resolve().parent)
            except ImportError as error:
                reason = " ".join(str(error).splitlines())
                print("状態: failed\\n入力: 未確認\\n元動画: 未確認\\n処理: 0/未確認フレーム、要確認1件\\n成果物: 未生成（共通キットの依存関係不足: " + reason + "）")
                return 1

if __name__ == "__main__":
    raise SystemExit(run())
'''


def build_kit(root: Path) -> tuple[Path, str]:
    from src.utils.schema.court import (
        COURT_KP_NAMES,
        COURT_SKELETON,
        STANDARD_COURT_CONFIG,
        court_keypoints_3d,
    )

    package = Path(__file__).parent
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:

        def add_module(name: str, content: str) -> None:
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(
                entry,
                content.replace(
                    "src.utils.configuration.", "_tennis_shared_configuration."
                ),
            )

        for module in sorted((package / "runtime").glob("*.py")):
            add_module(
                f"_tennis_annotation/{module.name}", module.read_text(encoding="utf-8")
            )
        # Reuse the real path authority; never require a tennis-lab installation in Chat.
        from src.utils.configuration import paths as shared_paths

        shared_root = Path(shared_paths.__file__).parent
        add_module("_tennis_shared_configuration/__init__.py", "")
        for name in ("paths.py", "schema.py", "errors.py"):
            add_module(
                f"_tennis_shared_configuration/{name}",
                (shared_root / name).read_text(encoding="utf-8"),
            )
    encoded = base64.b64encode(memory.getvalue()).decode("ascii")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=root) as directory:
        stage = Path(directory)
        (stage / "annotation_tools.py").write_text(
            _BOOTSTRAP.replace("__PAYLOAD__", repr(encoded)), encoding="utf-8"
        )
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
        for name in ("PROTOCOL.md", "PROJECT_INSTRUCTIONS.txt"):
            value = (
                (package / "resources" / name)
                .read_text(encoding="utf-8")
                .replace("{{KIT_VERSION}}", KIT_VERSION)
            )
            (stage / name).write_text(value, encoding="utf-8")
        files = {file.name: sha256_file(file) for file in sorted(stage.iterdir())}
        kit_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        write_json(
            stage / "kit_manifest.json",
            {"kit_version": KIT_VERSION, "kit_id": kit_id, "files": files},
        )
        destination = root / kit_id[:16]
        if destination.exists():
            for file in stage.iterdir():
                if not (destination / file.name).is_file() or sha256_file(
                    destination / file.name
                ) != sha256_file(file):
                    raise ValueError(
                        f"existing Project kit modified: {destination / file.name}"
                    )
        else:
            stage.rename(destination)
    return destination, kit_id
