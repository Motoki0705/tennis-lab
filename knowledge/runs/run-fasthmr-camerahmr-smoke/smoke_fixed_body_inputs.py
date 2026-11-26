"""Run the FastHMR component on a pinned short segment of the Meiji 4cam cohort."""
from argparse import ArgumentParser
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.submodules.models.fasthmr.assets import FastHMRConfig
from src.tennis_scene.evaluation.player_pose.bundle import load_body_input_manifest, pinned_file
from src.tennis_scene.pipeline.body_types import BodyRecoveryRequest
from src.tennis_scene.pipeline.components.body_view_selection import BodySelection, BodyViewSelectionOutput
from src.tennis_scene.pipeline.components.fasthmr import FastHMRInput, FastHMRModule
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime

COHORT_SHA256 = "3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580"
BENCHMARK = Path("data/tennis_multiview/processed/meiji_4cam/benchmarks")
CLIP = "video_000/clip_003"
PERSON = 0
SAMPLES = 8


def pinned_input():
    cohort = BENCHMARK / "multiview-player-evaluation-v1"
    manifest = load_body_input_manifest(cohort, expected_sha256=COHORT_SHA256)
    source_root = BENCHMARK / "multiview-v1"
    pinned_file(source_root, {"path": "manifest.json", "sha256": manifest["source_benchmark"]["manifest_sha256"]})
    clip = next(item for item in manifest["clips"] if item["clip_id"] == CLIP)
    source = ClipSource(CLIP, tuple(
        SourceVideo(video["camera_id"], pinned_file(source_root, video), video["sha256"],
                    video["num_frames"], video["fps"], clip["width"], clip["height"])
        for video in clip["videos"]
    ))
    with np.load(pinned_file(cohort, clip["fixed_inputs"]), allow_pickle=False) as arrays:
        row = clip["player_ids"].index(PERSON)
        assert np.array_equal(arrays["player_ids"], clip["player_ids"])
        assert np.array_equal(arrays["camera_ids"], source.camera_ids)
        camera = source.videos[int(arrays["selected_view_index"][row])]
        segment_labels = arrays["segment_index"][row]
        first_segment = np.unique(segment_labels[segment_labels >= 0])[0]
        frames = np.flatnonzero(segment_labels == first_segment)[:SAMPLES]
        assert len(frames) == SAMPLES and np.isin(frames, clip["evaluation_frames"]).all()
        observed = int(arrays["selected_view_observed"][row, frames].sum())
        request = BodyRecoveryRequest(camera.path, frames, arrays["keypoints_xyc"][row, frames],
                                      arrays["boxes_xys"][row, frames], source.size,
                                      arrays["selected_intrinsic"][row])
    return FastHMRInput(source, BodyViewSelectionOutput((
        BodySelection(PERSON, camera.camera_id, (request,), (observed,)),
    )))


def main():
    parser = ArgumentParser()
    parser.add_argument("mode", choices=("check-inputs", "pt32", "pt16"))
    mode = parser.parse_args().mode
    inputs = pinned_input()
    selection = inputs.selection.selections[0]
    camera = inputs.source.video(selection.camera_id)
    frame_numbers = selection.requests[0].source_frames
    print(json.dumps({"phase": "inputs", "clip": CLIP, "person": PERSON, "camera": camera.camera_id,
                      "source_frames": frame_numbers.tolist(), "observed": selection.observed_samples[0],
                      "video_sha256": camera.sha256, "cohort_sha256": COHORT_SHA256}), flush=True)
    if mode == "check-inputs":
        return

    output = Path(os.environ["TENNIS_REPRO_DIR"]) / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    print(json.dumps({"phase": "device", "device": torch.cuda.get_device_name(device),
                      "torch": torch.__version__, "cuda": torch.version.cuda, "backend": mode}), flush=True)
    if "5060 Ti" not in torch.cuda.get_device_name(device):
        raise RuntimeError("The pinned GPU smoke uses the local RTX 5060 Ti")
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    cfg = FastHMRConfig(
        Path("third_party/FastHMR").resolve(), Path("ckpt/fasthmr").resolve(),
        Path("ckpt/body_models/gvhmr/smpl_coco17_J_regressor.pt").resolve(), str(device), mode, 4, 0,
    )
    module = FastHMRModule(cfg)
    key = CLIP, "fasthmr"
    with InferenceRuntime({key: module.model_specs()}) as runtime, runtime.node(key) as models:
        print(json.dumps({"phase": "model_prepare", "backend": mode}), flush=True)
        models.borrow("mesh")
        models.borrow("body")
        print(json.dumps({"phase": "model_prepared", "backend": mode}), flush=True)
        result = module.process_with_models(inputs, models)
        result.validate_selection(inputs)
        segment = result.bodies[0].segments[0]
        geometry = models.borrow("body").reconstruct(segment.parameters)

    assert geometry.vertices.shape == (SAMPLES, 6890, 3)
    assert geometry.coco17.shape == (SAMPLES, 17, 3)
    assert np.isfinite(geometry.vertices).all() and np.isfinite(geometry.coco17).all()
    assert not np.allclose(geometry.vertices[0], geometry.vertices[-1])
    path = output / f"fasthmr-smoke-{mode}.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, source_frames=segment.source_frames, **asdict(segment.parameters),
                            vertices=geometry.vertices, coco17=geometry.coco17)
    temporary.replace(path)
    print(json.dumps({"phase": "done", "backend": mode, "samples": SAMPLES, "output": str(path),
                      "vertices_shape": list(geometry.vertices.shape), "valid": True}), flush=True)


if __name__ == "__main__":
    main()
