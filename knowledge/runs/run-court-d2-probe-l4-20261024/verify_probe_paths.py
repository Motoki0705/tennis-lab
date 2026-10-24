"""Reproduce the failed temporary-path contract on CPU, without any NHT call."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from src.synthetic_data_generation.rendering.nht import (
    NHTRenderCamera,
    NHTRenderCommandRequest,
    NHTRenderRequest,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.scratch.is_absolute():
        raise ValueError("Use a new relative scratch path to reproduce the failure")
    args.scratch.mkdir(parents=True, exist_ok=False)
    scene = (args.scratch / "fixture/reconstruction/export/scene.json").resolve()
    scene.parent.mkdir(parents=True)
    scene.write_text("{}\n")
    camera = NHTRenderCamera(
        camera_id="cpu-path-fixture",
        width=640,
        height=360,
        intrinsics=(300.0, 0.0, 320.0, 0.0, 300.0, 180.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
    )
    request_cameras = NHTRenderRequest((camera,))
    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(dir=args.scratch) as temporary:
        scratch = Path(temporary)
        assert not scratch.is_absolute()
        render_root = scratch / ".render"
        render_root.mkdir()
        with tempfile.TemporaryDirectory(dir=render_root) as inner:
            directory = Path(inner)
            for name, output, request, error in (
                (
                    "original_relative_temporary_root",
                    directory / "renders",
                    directory / "request.json",
                    "output_directory must be an absolute pathlib.Path.",
                ),
                (
                    "resolving_only_output_is_insufficient",
                    directory.resolve() / "renders",
                    directory / "request.json",
                    "arbitrary_request_path must be an absolute pathlib.Path.",
                ),
                (
                    "resolve_the_shared_temporary_root",
                    directory.resolve() / "renders",
                    directory.resolve() / "request.json",
                    None,
                ),
            ):
                try:
                    NHTRenderCommandRequest(
                        scene_path=scene,
                        output_directory=output,
                        arbitrary_cameras=request_cameras,
                        arbitrary_request_path=request,
                        without_source_images=True,
                        batch_size=1,
                    )
                except ValueError as actual:
                    assert str(actual) == error
                    checks.append({"case": name, "error": str(actual)})
                else:
                    assert error is None
                    checks.append({"case": name, "error": None})
    result = {
        "schema": "court_d2_probe_path_verification_v1",
        "checks": checks,
        "constructed_only_no_renderer_instantiated": True,
        "nht_subprocess_calls": 0,
        "gpu_calls": 0,
        "scope": "Path contract only; no assertion about successful GPU rendering or image quality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
