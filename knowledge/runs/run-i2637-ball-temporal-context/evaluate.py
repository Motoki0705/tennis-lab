"""Fixed-anchor inference on the long-sequence validation snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
from shape_baseline import OFFSETS, shape_peak

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.all_frames import (
    LABEL_KINDS,
    decode_label_runs,
)
from src.tasks.ball_detection.evaluation.temporal_heatmap_metrics import (
    calibrate,
    frame_metrics,
)
from src.tasks.ball_detection.evaluation.temporal_metrics import prediction_identities

SEED = 42
ANCHOR = 15
FRAMES = 32
MANIFEST_SHA = "af1773174bdbeed86ee5339846bb6f30ac42d86e26abd1e60c0c685b9c8b6fae"
STORE_SHA = {
    "metadata.json": "7a804cc4e3b3006d0128fbd953b0696712d45acee9dcef8a96c02ec46379037c",
    "index.npz": "0bb00facd10f9961081f9c47af1b62f5c3ca4f5d349a2ac011573d9e510a9feb",
}
MODELS = {
    "x3d_t32": dict(
        file="x3d-csn-dpt-T32-common-u10000-scene.pt",
        sha256="c46f724cd0e333d676ee35823b9526e6039e4aa471f97dc24e76871910814242",
        size=(1280, 720), frames=32, threshold=0.5380124449729919, precision="bf16",
    ),
    "i935_t8": dict(
        file="i935-mixed-ft-s42-epoch09.ckpt",
        sha256="37f4c59aead00062829280ad591b3874886978891104a290f704a2c33c3c1b36",
        size=(512, 288), frames=8, threshold=0.5, precision="fp32",
    ),
}
CONDITIONS = ("ordered", "repeat_anchor", "shuffle_context")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_sha(path: Path, digest: str) -> None:
    if sha(path) != digest:
        raise ValueError(f"Input identity mismatch: {path}")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rank(*parts) -> bytes:
    return hashlib.sha256(json.dumps([SEED, *parts], separators=(",", ":")).encode()).digest()


def sample_anchors(manifest: dict) -> list[dict]:
    pools = defaultdict(list)
    for record in manifest["clips"]:
        clip = record["clip"]
        if clip["split"] != "val":
            continue
        identity = clip["camera_id"] if clip["source"] == "meiji" else clip["group_id"]
        codes = decode_label_runs(record["label_runs"], clip["frame_count"])
        for start, stop in record["common_intervals"]:
            stop_anchor = stop - (FRAMES - 1 - ANCHOR)
            for frame in range(start + ANCHOR, stop_anchor):
                kind = LABEL_KINDS[codes[frame]]
                if kind != "masked":
                    pools[clip["source"], identity, kind].append((record, frame))

    observed, empty, outside = LABEL_KINDS[1:]
    chat_cal = {observed: 8, empty: 8, outside: 4}
    chat_eval = {observed: 4, empty: 2, outside: 2}
    groups = sorted({group for source, group, _ in pools if source == "chat_annotation"}, key=rank)
    eligible = [
        group for group in groups
        if all(len(pools["chat_annotation", group, kind]) >= count for kind, count in chat_cal.items())
    ]
    if len(eligible) < 18:
        raise ValueError(f"Need 18 eligible Chat groups; found {len(eligible)}")
    calibration_groups, evaluation_groups = eligible[:6], eligible[6:18]
    samples = []

    def add(split: str, source: str, group: str, counts: dict[str, int]) -> None:
        for kind, count in counts.items():
            pool = pools[source, group, kind]
            selected = sorted(
                pool, key=lambda p: rank(split, p[0]["clip"]["clip_id"], p[1]),
            )[:count]
            if len(selected) != count:
                raise ValueError(f"Insufficient {source}/{group}/{kind} support")
            for record, frame in selected:
                clip = record["clip"]
                samples.append(dict(
                    split=split, source=source, source_group=clip["group_id"],
                    camera_id=clip["camera_id"], clip_id=clip["clip_id"], clip_index=clip["index"],
                    native_fps=clip["fps"], frame_index=frame, kind=kind,
                    window_start=frame - ANCHOR, rgb_shard_sha256=record["rgb_shard_sha256"],
                ))

    for group in calibration_groups:
        add("calibration", "chat_annotation", group, chat_cal)
    for group in evaluation_groups:
        add("evaluation", "chat_annotation", group, chat_eval)
    for camera in ("cam0", "cam1", "cam2"):
        add("evaluation", "meiji", camera, {observed: 8})
    add("evaluation", "tracknet", "game9", {observed: 16, empty: 8})
    samples.sort(key=lambda r: (r["split"], r["clip_id"], r["frame_index"]))
    if len({(r["clip_id"], r["frame_index"]) for r in samples}) != 264:
        raise ValueError("Expected 264 distinct anchors")
    return samples


def freeze_plan(store: BallFrameStore, manifest_path: Path, output: Path) -> None:
    require_sha(manifest_path, MANIFEST_SHA)
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["label_kinds"] != list(LABEL_KINDS) or manifest["ball_store"]["hashes"] != STORE_SHA:
        raise ValueError("Unexpected evaluation labels or RGB store")
    samples = sample_anchors(manifest)
    checked = set()
    for clip_id, entries in groupby(sorted(samples, key=lambda r: r["clip_id"]), lambda r: r["clip_id"]):
        clip = store.clip_by_id(clip_id)
        states = annotation_states(store, clip)
        for entry in entries:
            if clip.index not in checked:
                require_sha(store.image_path(clip), entry["rgb_shard_sha256"])
                checked.add(clip.index)
            frame = entry["frame_index"]
            if clip.split != "val" or clip.index != entry["clip_index"] or clip.fps != entry["native_fps"]:
                raise ValueError(f"Manifest clip identity mismatch: {clip_id}")
            if entry["kind"] == "observed":
                if not states.supervision[frame]:
                    raise ValueError(f"Observed label disagrees with RGB store: {clip_id} #{frame}")
                # Exactly the target pixel conversion used by FrozenBallWindowDataset.
                endpoint = np.asarray((clip.source_width - 1, clip.source_height - 1), np.float32)
                uv = (states.xy[frame] / (clip.scale * endpoint)).astype(np.float32)
                entry["teacher_xy"] = (uv * endpoint).tolist()
            else:
                entry["teacher_xy"] = None
            entry["source_size"] = [clip.source_width, clip.source_height]
    write_json(output, dict(
        schema="ball_context_anchor_plan.v1", seed=SEED, manifest_sha256=MANIFEST_SHA,
        store_sha256=STORE_SHA, anchors=samples,
    ))


def context_order(frames: int, condition: str, clip_id: str, frame: int) -> list[int]:
    anchor = frames // 2 - 1
    indices = list(range(frames))
    if condition == "ordered":
        return indices
    if condition == "repeat_anchor":
        return [anchor] * frames
    if condition == "shuffle_context":
        context = [i for i in indices if i != anchor]
        permutation = sorted(context, key=lambda i: rank(clip_id, frame, frames, i))
        if permutation == context:
            permutation = permutation[1:] + permutation[:1]
        return permutation[:anchor] + [anchor] + permutation[anchor:]
    raise ValueError(f"Unknown condition: {condition}")


def add_prediction(anchor: dict, uv, score: float) -> dict:
    width, height = anchor["source_size"]
    xy = float(uv[0]) * (width - 1), float(uv[1]) * (height - 1)
    if not all(math.isfinite(v) for v in (*xy, score)) or not 0 <= score <= 1:
        raise ValueError(f"Invalid prediction for {anchor['clip_id']} #{anchor['frame_index']}")
    return dict(
        anchor, score=score, pred_xy=xy,
        error_px=math.dist(xy, anchor["teacher_xy"]) if anchor["kind"] == "observed" else None,
    )


def summarize(records: list[dict], threshold: dict) -> dict:
    return dict(
        **prediction_identities(records), **frame_metrics(records, threshold, distance_px=20.0),
        calibration=threshold,
        by_source={
            source: frame_metrics(
                [r for r in records if r["source"] == source], threshold, distance_px=20.0,
            )
            for source in sorted({r["source"] for r in records})
        },
    )


def checked_anchors(store: BallFrameStore, plan_path: Path, *, evaluation_only: bool) -> list[dict]:
    plan = json.loads(plan_path.read_bytes())
    if plan["manifest_sha256"] != MANIFEST_SHA or plan["store_sha256"] != STORE_SHA or plan["seed"] != SEED:
        raise ValueError("Unexpected anchor plan")
    anchors = [r for r in plan["anchors"] if not evaluation_only or r["split"] == "evaluation"]
    if Counter(r["split"] for r in anchors) != (
        {"evaluation": 144} if evaluation_only else {"evaluation": 144, "calibration": 120}
    ):
        raise ValueError("Unexpected anchor support")
    shards = {r["clip_id"]: r["rgb_shard_sha256"] for r in anchors}
    for clip_id, digest in sorted(shards.items()):
        require_sha(store.image_path(store.clip_by_id(clip_id)), digest)
    return anchors


def cpu_evaluate(store: BallFrameStore, plan_path: Path, output: Path) -> None:
    anchors = checked_anchors(store, plan_path, evaluation_only=False)
    predictions = defaultdict(list)
    for anchor in anchors:
        clip = store.clip_by_id(anchor["clip_id"])
        frames = [store.read_bgr(store.row_of(clip, anchor["frame_index"] + delta)) for delta in OFFSETS]
        if any(image.shape != (720, 1280, 3) for image in frames):
            raise ValueError(f"Unexpected stored frame size: {clip.clip_id}")
        uv, score = shape_peak(frames)
        predictions[anchor["split"]].append(add_prediction(anchor, uv, score))
    threshold = calibrate(predictions["calibration"], split="val")
    write_json(output, dict(
        plan_sha256=sha(plan_path), opencv=cv2.__version__,
        metrics=summarize(predictions["evaluation"], threshold), predictions=predictions,
    ))


def gpu_evaluate(store: BallFrameStore, plan_path: Path, checkpoint: Path, output: Path) -> None:
    import torch

    from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor
    from src.tasks.ball_detection.inference.x3d_dpt import X3DDPTPredictor
    from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
    from src.utils.configuration import PathResolver, RuntimePathRoots

    anchors = checked_anchors(store, plan_path, evaluation_only=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.70)
    root = Path.cwd()
    resolver = PathResolver(RuntimePathRoots.from_mapping(dict(
        project_root=str(root), data_root="data", checkpoint_root=str(checkpoint.parent.resolve()),
        artifact_root="outputs", output_root="outputs", cache_root=".cache", external_asset_root="third_party",
    ), repository_root=root))
    report = dict(plan_sha256=sha(plan_path), torch=torch.__version__, opencv=cv2.__version__,
                  gpu=torch.cuda.get_device_name(), models={})
    candidate_config = BallCandidateConfig(max_candidates=1, nms_kernel=5, patch_size=5)
    for model, cfg in MODELS.items():
        path = checkpoint / cfg["file"]
        require_sha(path, cfg["sha256"])
        width, height = cfg["size"]
        frames = cfg["frames"]
        if model == "x3d_t32":
            predictor = X3DDPTPredictor.load_from_checkpoint(
                path, device="cuda", image_size=(height, width), frames=frames, input_channels=5,
                normalize_imagenet=False, precision="bf16", subpixel_refine=True, compile_mode="off",
            )
        else:
            predictor = BallDetectionPredictor.load_from_checkpoint(
                path.resolve(), resolver=resolver, device="cuda", subpixel_refine=True,
                strict=True, weights_only=False,
            )
        if predictor.configured_frames != frames:
            raise ValueError(f"Checkpoint T disagrees for {model}")
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        predictions = {condition: [] for condition in CONDITIONS}
        offset = frames // 2 - 1
        for count, anchor in enumerate(anchors, 1):
            clip = store.clip_by_id(anchor["clip_id"])
            first = anchor["frame_index"] - offset
            images = []
            for frame in range(first, first + frames):
                bgr = store.read_bgr(store.row_of(clip, frame))
                if bgr.shape != (720, 1280, 3):
                    raise ValueError(f"Unexpected input dimensions for {clip.clip_id}")
                resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
                images.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            rgb = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
            for condition, records in predictions.items():
                order = context_order(frames, condition, clip.clip_id, anchor["frame_index"])
                inputs = rgb[order].unsqueeze(0)
                if model == "i935_t8":
                    inputs = inputs.float().div_(255)
                result = predictor.predict(inputs, candidate_config=candidate_config)
                records.append(add_prediction(anchor, result.coords[0, offset], float(result.confidence[0, offset])))
            if count % 24 == 0:
                print(f"{model}: {count}/{len(anchors)}", flush=True)
        torch.cuda.synchronize()
        metrics = {
            condition: summarize(records, dict(mode="threshold", threshold=cfg["threshold"]))
            for condition, records in predictions.items()
        }
        report["models"][model] = dict(
            checkpoint=cfg, seconds=time.perf_counter() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            metrics=metrics, predictions=predictions,
        )
        write_json(output, report)
        del predictor
        torch.cuda.empty_cache()


def self_check() -> None:
    for frames in (8, 32):
        anchor = frames // 2 - 1
        ordered, repeat, shuffle = [context_order(frames, c, "clip", 60) for c in CONDITIONS]
        assert ordered == list(range(frames)) and repeat == [anchor] * frames
        assert shuffle[anchor] == anchor and sorted(shuffle) == ordered and shuffle != ordered
    base = dict(source_size=[257, 513], source="fixture", clip_id="test", native_fps="60", kind="observed")
    pos = add_prediction(dict(base, frame_index=0, teacher_xy=[128., 256.]), (144 / 256, 268 / 512), 0.1)
    neg = add_prediction(dict(base, frame_index=1, kind="out_of_frame", teacher_xy=None), (0.2, 0.3), 0.6)
    report = summarize([pos, neg], dict(mode="threshold", threshold=0.5))
    assert math.isclose(report["mean_error_px"], 20) and report["within_distance_rate"] == 1
    assert report["negative_fpr"] == report["observed_fnr"] == 1
    still = np.full((106, 120, 3), 42, np.uint8)
    cv2.circle(still, (24, 63), 3, (245, 245, 245), -1)
    motion = still.copy()
    cv2.circle(motion, (84, 36), 3, (190, 208, 204), -1)
    uv, score = shape_peak([still, still, still, motion, still, still, still])
    assert math.dist((uv[0] * 119, uv[1] * 105), (84, 36)) <= 1 and score > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "shape", "gpu", "self-check"))
    parser.add_argument("--store", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("ckpt/ball_detection"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "self-check":
        self_check()
        return
    if args.store is None or args.output is None:
        parser.error("--store and --output are required")
    cv2.setNumThreads(2)
    for file, digest in STORE_SHA.items():
        require_sha(args.store / file, digest)
    store = BallFrameStore(args.store.resolve())
    if args.stage == "plan":
        if args.manifest is None:
            parser.error("--manifest is required for plan")
        freeze_plan(store, args.manifest, args.output)
    elif args.plan is None:
        parser.error("--plan is required for shape and gpu")
    elif args.stage == "shape":
        cpu_evaluate(store, args.plan, args.output)
    else:
        gpu_evaluate(store, args.plan, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
