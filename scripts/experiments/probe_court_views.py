"""Measure resident rendering over four scenes and newly proposed camera poses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from scripts.experiments.benchmark_court_ondemand import response
from src.synthetic_data_generation.alignment.validation import load_alignment_result
from src.synthetic_data_generation.dataset.court.components.labels import (
    attach_renderer_visibility,
    project_court_semantics_for_version,
)
from src.synthetic_data_generation.dataset.court.schema import CourtDatasetSchemaVersion
from src.synthetic_data_generation.rendering.nht.contracts import NHTRenderCamera
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.utils.data.float32_store import read_float32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--nht-worker", type=Path, required=True)
    parser.add_argument("--nht-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {}
    for scene_id in ["B00", "B01", "B02", "B03"]:
        scene = args.repo / "data/synthetic_data_generation/scenes" / scene_id
        alignment = load_alignment_result(scene / "alignment/alignment.json")
        dataset = json.loads((scene / "datasets/court/dataset.json").read_text())
        training = [s for s in dataset["samples"] if s["split"] == "train"]
        samples = [training[i] for i in np.linspace(0, len(training) - 1, 8, dtype=int)]
        cameras = [SceneCamera.from_dict(s["camera"]) for s in samples]
        proposals = []
        for i, camera in enumerate(cameras):
            transform = camera.camera_to_scene.matrix().copy()
            transform[:3, 3] += 0.05 * transform[:3, 0]
            proposals.append(
                SceneCamera(
                    camera_id=f"new-{i}",
                    source_frame_index=i,
                    width=camera.width,
                    height=camera.height,
                    intrinsics=camera.intrinsics,
                    camera_to_scene=RigidTransform.from_matrix(transform),
                    image_path=f"generated/new-{i}.png",
                )
            )
        small = []
        for i, camera in enumerate(proposals):
            width = 256
            height = round(camera.height * width / camera.width)
            sx = width / camera.width
            sy = height / camera.height
            K = np.asarray(camera.intrinsics).reshape(3, 3).copy()
            K[0] *= sx
            K[1] *= sy
            K[0, 2] += (sx - 1) / 2
            K[1, 2] += (sy - 1) / 2
            small.append(
                SceneCamera(
                    camera_id=f"small-{i}",
                    source_frame_index=i,
                    width=width,
                    height=height,
                    intrinsics=tuple(K.ravel()),
                    camera_to_scene=camera.camera_to_scene,
                    image_path=f"generated/small-{i}.png",
                )
            )
        all_cameras = cameras + proposals + small
        request = args.output / f"{scene_id}-cameras.json"
        request.write_text(
            json.dumps(
                {
                    "schema": "nht_render_request_v1",
                    "cameras": [
                        NHTRenderCamera(
                            camera_id=c.camera_id,
                            width=c.width,
                            height=c.height,
                            intrinsics=c.intrinsics,
                            camera_to_scene=alignment.metric_adapter.nht_from_metric_camera(
                                c.camera_to_scene
                            ),
                        ).to_dict()
                        for c in all_cameras
                    ],
                }
            )
        )
        with tempfile.TemporaryDirectory(
            prefix="court-views-", dir="/dev/shm"
        ) as temporary:
            buffer = Path(temporary) / "batch.npy"
            process = subprocess.Popen(
                [
                    str(args.nht_python),
                    str(args.nht_worker),
                    "--scene",
                    str(scene / "reconstruction/export/scene.json"),
                    "--cameras",
                    str(request),
                    "--buffer",
                    str(buffer),
                    "--auxiliary",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            assert process.stdin is not None
            summary = {"startup": response(process), "groups": {}}
            try:
                for name, offset in [
                    ("original", 0),
                    ("novel_5cm", 8),
                    ("novel_256px", 16),
                ]:
                    times = []
                    for _repeat in range(4):
                        process.stdin.write(
                            json.dumps(
                                {
                                    "op": "render",
                                    "indices": list(range(offset, offset + 8)),
                                }
                            )
                            + "\n"
                        )
                        process.stdin.flush()
                        times.append(response(process))
                    rgb = np.load(buffer)
                    alpha = np.load(buffer.with_suffix(".alpha.npy"))
                    depth = np.load(buffer.with_suffix(".depth.npy"))
                    if (
                        not np.isfinite(rgb).all()
                        or not np.isfinite(alpha).all()
                        or not np.isfinite(depth).all()
                    ):
                        raise ValueError("Renderer emitted nonfinite values")
                    started = time.perf_counter()
                    visible = []
                    for i, camera in enumerate(all_cameras[offset : offset + 8]):
                        projection = project_court_semantics_for_version(
                            camera,
                            alignment.layout,
                            schema_version=CourtDatasetSchemaVersion.V3,
                        )
                        labeled = attach_renderer_visibility(
                            projection, alpha=alpha[i], depth=depth[i]
                        )
                        visible.append(labeled.visible_point_count)
                    group = {
                        "timings": times,
                        "visible_points": visible,
                        "kp_projection_visibility_seconds": time.perf_counter()
                        - started,
                        "height": rgb.shape[1],
                        "width": rgb.shape[2],
                    }
                    if name == "original":
                        reference = np.stack(
                            [
                                read_float32(scene / "datasets/court" / s["rgb"])
                                for s in samples
                            ]
                        )
                        group["reference_float_mae"] = float(
                            np.abs(rgb - reference).mean()
                        )
                        original_rgb = rgb.copy()
                    if name == "novel_5cm":
                        group["original_float_mae"] = float(
                            np.abs(rgb - original_rgb).mean()
                        )
                        group["u8_changed_fraction"] = float(
                            np.mean(np.round(rgb * 255) != np.round(original_rgb * 255))
                        )
                    summary["groups"][name] = group
            finally:
                if process.poll() is None:
                    process.stdin.write('{"op":"stop"}\n')
                    process.stdin.flush()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            result[scene_id] = summary
            (args.output / "metrics.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(
                scene_id,
                {
                    k: np.median([x["render_seconds"] for x in v["timings"][1:]])
                    for k, v in summary["groups"].items()
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
