"""Freeze tensors at the real GVHMR wrapper and denoiser boundaries."""

import argparse
import json
import subprocess
from dataclasses import asdict, fields
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from benchmark import write_json
from inputs import check_split, sha256_file
from locked_inputs import HERE, load_segments, tensor_addresses
from model import CHECKPOINT, attention_mask, smpl_assets

from src.submodules.models import GvhmrMeshRecovery, GvhmrRequest
from src.submodules.vendor.gvhmr.pipeline import GvhmrDemoModel, Pipeline
from src.submodules.vendor.gvhmr.utils.net_utils import length_to_mask
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.gvhmr import GVHMRInput
from src.tennis_scene.pipeline.components.hmr2_features import HMR2FeaturesOutput
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec

SPLIT_PINS = HERE.parent / "run-i2297-clipreid-tensorrt/inputs.json"
DATASET = Path("data/tennis_multiview/processed/meiji_4cam/dataset")
ARCHIVE = (
    "video-batch-benchmark/tennis_scene/evaluate/video-batch-benchmark-20261119/"
    "serial-fb6255a-first-1/attempts/1-first/dataset"
)


class Captured(Exception):
    def __init__(self, tensors):
        self.tensors = tensors


class CaptureDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(()), requires_grad=False)

    def forward(self, **inputs):
        raise Captured(inputs)


class CaptureDemo(GvhmrDemoModel):
    """Stop the actual vendored pipeline just before neural inference."""

    def predict(self, data, *, static_cam):
        try:
            return super().predict(data, static_cam=static_cam)
        except Captured as capture:
            length = capture.tensors["length"]
            frames = data["kp2d"].shape[0]
            raise Captured(
                {
                    **data,
                    **capture.tensors,
                    "length_raw": data["length"],
                    "f_imgseq_raw": data["f_imgseq"],
                    "static_cam": torch.tensor(static_cam),
                    "sequence_mask": length_to_mask(length, frames),
                    "keypoint_visible": capture.tensors["obs"][..., 2] > 0.5,
                    "attention_mask": attention_mask(frames, length.device),
                }
            ) from None


def load_clip(archive, dataset, media_pin, assets):
    clip = media_pin["clip"]
    directory = archive / "videos" / Path(clip).parent / "clips" / Path(clip).name
    store_path = directory / "annotations/tennis_scene"
    saved = json.loads((store_path / "scene.json").read_text())["source"]
    source = ClipSource(
        saved["clip_id"],
        tuple(
            SourceVideo(**{**video, "path": Path(video["path"])})
            for video in saved["videos"]
        ),
    )
    if source.clip_id != clip or source.num_frames != media_pin["num_frames"]:
        raise ValueError(f"Unexpected clip identity/timeline in {store_path}")
    if source.camera_ids != tuple(stream["camera"] for stream in media_pin["streams"]):
        raise ValueError(f"Unexpected camera order in {store_path}")
    for stream, video in zip(media_pin["streams"], source.videos, strict=True):
        if (
            video.sha256 != stream["sha256"]
            or sha256_file(dataset / stream["media"]) != video.sha256
        ):
            raise ValueError(
                f"Camera media do not match the protocol pin: {clip}/{video.camera_id}"
            )

    store = ClipStore(store_path, saved, memory_entries=0)
    selection_ref = store.active("body_view_selection")
    features_ref = store.active("hmr2_features")
    gvhmr_ref = store.active("gvhmr")
    for ref, schema, version in (
        (selection_ref, "body_view_selection", 1),
        (features_ref, "hmr2_features", 1),
        (gvhmr_ref, "body_parameters", 2),
    ):
        if ref is None or (ref.schema, ref.version) != (schema, version):
            raise ValueError(f"Missing or incompatible {schema} in {store_path}")
    selection_doc = store.descriptor(selection_ref)
    features_doc = store.descriptor(features_ref)
    gvhmr_doc = store.descriptor(gvhmr_ref)
    if features_doc["dependencies"]["selection"] != asdict(selection_ref) or gvhmr_doc[
        "dependencies"
    ] != {
        "features": asdict(features_ref),
        "selection": asdict(selection_ref),
    }:
        raise ValueError(f"Inconsistent body artifact lineage: {store_path}")
    feature_spec = features_doc["identity"]["settings"]["models"]["features"]
    gvhmr_spec = gvhmr_doc["identity"]["settings"]["models"]["mesh"]
    if (
        feature_spec["dtype"] != "float32"
        or gvhmr_spec["dtype"] != "float32"
        or not gvhmr_spec["settings"]["static_cam"]
    ):
        raise ValueError(
            f"These pins require standard FP32 body stages and a static camera: {store_path}"
        )
    if (
        feature_spec["assets"]["checkpoint"]["sha256"] != assets["hmr2"]["sha256"]
        or feature_spec["assets"]["mean_params"]["sha256"]
        != assets["hmr2_mean_params"]["sha256"]
        or gvhmr_spec["assets"]["gvhmr"]["sha256"] != assets["gvhmr"]["sha256"]
    ):
        raise ValueError(f"Saved body stages used different weights: {store_path}")

    value = GVHMRInput(
        source,
        store.load(selection_ref, ArtifactCodec(BodyViewSelectionOutput)),
        store.load(features_ref, ArtifactCodec(HMR2FeaturesOutput)),
    )
    value.validate()
    upstream = {
        "body_view_selection": selection_ref.sha256,
        "hmr2_features": features_ref.sha256,
        "gvhmr": gvhmr_ref.sha256,
        "selection_settings": selection_doc["identity"]["settings"],
        "hmr2_settings": feature_spec["settings"],
        "selection_dependencies": {
            key: ref["sha256"] for key, ref in selection_doc["dependencies"].items()
        },
    }
    return value, upstream


def capture_segment(mesh, demo, request, segment):
    args = GvhmrRequest(
        kp2d=torch.from_numpy(request.keypoints.copy()),
        bbx_xys=torch.from_numpy(request.boxes_xys.copy()),
        f_imgseq=torch.from_numpy(segment.features.copy()),
        width=request.size[0],
        height=request.size[1],
        static_cam=True,
        K_fullimg=torch.from_numpy(request.intrinsic.astype(np.float32)),
    )
    with patch.object(mesh, "_model", demo):
        try:
            mesh._predict_impl(args)
        except Captured as capture:
            return {
                "source_frames": request.source_frames,
                **{
                    name: tensor.detach().numpy()
                    for name, tensor in capture.tensors.items()
                },
            }
    raise AssertionError("GVHMR bypassed the denoiser tap")


def freeze(args):
    media = json.loads(SPLIT_PINS.read_text())
    split = check_split(media)
    assets = smpl_assets(args.checkpoints)
    paths = {
        "gvhmr": args.checkpoints / CHECKPOINT,
        "hmr2": args.checkpoints / "hmr2/epoch=10-step=25000.ckpt",
        "smplx": args.checkpoints / "body_models/smplx/SMPLX_NEUTRAL.npz",
        **{field.name: getattr(assets, field.name) for field in fields(assets)},
    }
    identities = {
        key: {
            "path": str(path.relative_to(args.checkpoints)),
            "sha256": sha256_file(path),
        }
        for key, path in paths.items()
    }
    args.output.mkdir(parents=True)
    mesh = GvhmrMeshRecovery(
        paths["gvhmr"], paths["smplx"].parents[1], device="cpu", smpl_assets=assets
    )
    demo = CaptureDemo(Pipeline(CaptureDenoiser(), None))
    clips, rows = {}, []
    for pin in media["clips"]:
        value, upstream = load_clip(args.archive, args.dataset, pin, identities)
        source = value.source
        clips[source.clip_id] = {
            "split": pin["split"],
            "fps": source.fps,
            "store": f"videos/{Path(source.clip_id).parent}/clips/{Path(source.clip_id).name}/annotations/tennis_scene",
            "media": [
                {
                    key: stream[key]
                    for key in ("camera", "media", "sha256", "source_frames_inclusive")
                }
                for stream in pin["streams"]
            ],
            **upstream,
        }
        grid = sampled_frame_indices(
            source.num_frames,
            source.fps,
            max_frames=upstream["selection_settings"]["max_frames"],
        )
        for selected, features in zip(
            value.selection.selections, value.features.bodies, strict=True
        ):
            for request, segment, observed in zip(
                selected.requests,
                features.segments,
                selected.observed_samples,
                strict=True,
            ):
                frames = request.source_frames
                first = int(np.searchsorted(grid, frames[0]))
                if not np.array_equal(frames, grid[first : first + len(frames)]):
                    raise ValueError(
                        f"Body selection is not a continuous segment of the production grid: {source.clip_id}"
                    )
                stream = next(
                    stream
                    for stream in pin["streams"]
                    if stream["camera"] == selected.camera_id
                )
                offset = stream["source_frames_inclusive"][0]
                name = f"{Path(source.clip_id).name}-p{selected.person_id}-{selected.camera_id}-s{segment.segment_index}"
                arrays = capture_segment(mesh, demo, request, segment)
                np.savez(args.output / f"{name}.npz", **arrays)
                rows.append(
                    {
                        "id": name,
                        "clip": source.clip_id,
                        "split": pin["split"],
                        "camera": selected.camera_id,
                        "person": selected.person_id,
                        "segment": segment.segment_index,
                        "clip_frames_inclusive": [int(frames[0]), int(frames[-1])],
                        "raw_frames_inclusive": [
                            offset + int(frames[0]),
                            offset + int(frames[-1]),
                        ],
                        "length": len(frames),
                        "observed": observed,
                        "tensors": tensor_addresses(arrays),
                    }
                )
        if not rows or rows[-1]["clip"] != source.clip_id:
            raise ValueError(f"No selected body segments: {source.clip_id}")
    return {
        "version": 1,
        "dataset": media["dataset"],
        "archive": ARCHIVE,
        "protocol_media_pins_sha256": sha256_file(SPLIT_PINS),
        "nonoverlap": split,
        "assets": identities,
        "clips": clips,
        "segments": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            text=True,
        ).strip()
    ).parent
    parser.add_argument(
        "--archive", type=Path, default=root / ".worktree-outputs" / ARCHIVE
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--checkpoints", type=Path, default=Path("ckpt"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.checkpoints = args.checkpoints.resolve()
    torch.set_num_threads(2)
    lock = freeze(args)
    list(load_segments(args.output, lock))
    write_json(args.output / "tensor-lock.json", lock)
    print([(row["id"], row["length"]) for row in lock["segments"]], flush=True)


if __name__ == "__main__":
    main()
