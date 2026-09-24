"""Full-scene association smoke/evaluation. GPU invocation must use training-queue.

GT identities and camera sides are used only for scoring after prediction.
No dataset files, checkpoints or training settings are modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.base.model_io.association_contracts import AssociationObservationRequest
from src.tasks.blcs.generate_dataset.io.dataset_io import load_scene as load_blcs_scene
from src.tasks.blcs.inference.association_predictor import BLCSAssociationPredictor
from src.tasks.plcs.generate_dataset.io.scene_loader import (
    load_scene as load_plcs_scene,
)
from src.tasks.plcs.inference.association_predictor import PLCSAssociationPredictor
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.utilts.timeline import association_frame_indices
from src.utils.checksum import dual_sha256


def load_request(task: str, path: Path, views: int) -> tuple[AssociationObservationRequest, torch.Tensor, torch.Tensor]:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    scene: Any = (load_plcs_scene if task == "plcs" else load_blcs_scene)(path, court_keypoint_contract=contract)
    presence = np.asarray(scene["person_present" if task == "plcs" else "ball_present"], bool)
    fps = float(scene["meta"]["fps" if task == "plcs" else "fps_out"])
    indices = association_frame_indices(len(presence), fps)
    cameras = scene["cameras"][:views]
    if len(cameras) != views:
        raise ValueError("Requested views are not available")
    records = [c.court_view if task == "plcs" else c["court_view"] for c in cameras]
    ids = tuple(record.camera_id for record in records)
    reference = min(ids)
    uv_rows, vis_rows, court_rows, cv_rows = [], [], [], []
    for camera in cameras:
        if task == "plcs":
            uv = np.asarray(camera.human_kp_uv[indices], np.float32)
            visible = np.asarray(camera.human_kp_vis[indices], bool) & presence[indices, :, None]
            court = np.asarray(camera.court_kp_uv[indices, :14], np.float32)
            court_vis = np.asarray(camera.court_kp_vis[indices, :14], bool)
        else:
            uv = np.asarray(camera["ball_uv"][indices], np.float32)[..., None, :]
            visible = (np.asarray(camera["ball_vis"][indices], bool) & presence[indices])[..., None]
            court = np.broadcast_to(camera["court_kp_uv"][:14], (len(indices), 14, 2)).astype(np.float32)
            court_vis = np.broadcast_to(camera["court_kp_vis"][:14], (len(indices), 14)).astype(bool)
        uv_rows.append(torch.from_numpy(np.where(visible[..., None], uv, 0).copy()))
        vis_rows.append(torch.from_numpy(visible.copy()))
        court_rows.append(torch.from_numpy(np.where(court_vis[..., None], court, 0).copy()))
        cv_rows.append(torch.from_numpy(court_vis.copy()))
    request = AssociationObservationRequest(torch.stack(uv_rows), torch.stack(vis_rows), torch.stack(court_rows), torch.stack(cv_rows), ids, reference, torch.from_numpy(indices))
    gt_ids = torch.arange(presence.shape[1], dtype=torch.int64)[None, None].expand(views, len(indices), -1)
    gt_ids = torch.where(request.object_vis.any(-1), gt_ids, -1)
    absolute = torch.tensor([record.camera_center_court_m[1] > 0 for record in records])
    side = absolute ^ absolute[ids.index(reference)]
    return request, gt_ids, side


def identity_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    valid = target >= 0
    table: NDArray[np.int64] = np.zeros((10, 10), np.int64)
    for gt in range(10):
        for pred in range(10):
            table[gt, pred] = int(((target == gt) & (predicted == pred)).sum())
    rows, columns = linear_sum_assignment(-table)
    correct = int(table[rows, columns].sum())
    total = int(valid.sum())
    return {"identity_correct": correct, "identity_count": total, "identity_accuracy": correct / total if total else None,
            "accepted_count": int(((predicted >= 0) & valid).sum()), "acceptance_fraction": float(((predicted >= 0) & valid).sum()) / total if total else None,
            "fp_precision": None, "fp_recall": None, "fp_note": "Unaugmented source scenes contain no labelled false-positive detections"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("plcs", "blcs"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--completion-checkpoint", type=Path, help="Last checkpoint from the same completed 60-epoch run")
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", required=True, help="Explicit test scene names; use all for the complete test split")
    parser.add_argument("--views", nargs="+", type=int, default=[3, 5], choices=(3, 4, 5))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--boundary-padding", action="store_true", help="Also test explicit padding to 512/1024 on full scenes, without cropping")
    args = parser.parse_args()
    test_file = args.scene_root / "splits" / "test.txt"
    if not test_file.is_file():
        test_file = args.scene_root / "test.txt"
    test_names = [line.strip() for line in test_file.read_text().splitlines() if line.strip()]
    requested = test_names if args.scenes == ["all"] else args.scenes
    if any(name not in test_names for name in requested):
        raise ValueError("Evaluation requires scenes from the held-out test split")
    started = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    epoch = int(checkpoint["epoch"])
    run_dir = str(checkpoint["hyper_parameters"]["config"]["run"]["output_dir"])
    del checkpoint
    gc.collect()
    training_finished = None
    if args.completion_checkpoint is not None:
        completion = torch.load(args.completion_checkpoint, map_location="cpu", weights_only=False)
        if str(completion["hyper_parameters"]["config"]["run"]["output_dir"]) != run_dir:
            raise ValueError("Completion proof must come from the selected checkpoint's run")
        training_finished = int(completion["epoch"]) >= 59
        del completion
        gc.collect()
    cls = PLCSAssociationPredictor if args.task == "plcs" else BLCSAssociationPredictor
    predictor = cls.load(args.checkpoint, device=args.device)
    if hasattr(predictor.model, "_orig_mod"):
        raise RuntimeError("Integration inference must use the eager model")
    report: dict[str, Any] = {"task": args.task, "checkpoint": str(args.checkpoint), "checkpoint_sha256": dual_sha256(args.checkpoint),
        "checkpoint_epoch": epoch, "training_finished_60_epochs": training_finished, "device": args.device, "cases": [],
        "torch": str(torch.__version__), "architecture": OmegaConf.to_container(predictor.module.config.model, resolve=True)}
    for scene_name in requested:
        for views in args.views:
            try:
                request, gt, side = load_request(args.task, args.scene_root / "scenes" / scene_name, views)
            except ReconstructionUnavailable as exc:
                report["cases"].append({"scene": scene_name, "views": views, "status": "unsupported", "reason": exc.reason})
                continue
            lengths = [max(512, len(request.frame_indices))]
            if args.boundary_padding:
                exact = 512 if lengths[0] <= 512 else 1024
                if exact not in lengths:
                    lengths.append(exact)
            for model_frames in lengths:
                from src.tasks.base.model_io.association_contracts import (
                    AssociationInferencePolicy,
                )
                policy = replace(AssociationInferencePolicy(), min_frames=max(512, model_frames))
                if args.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                tick = time.monotonic()
                result = predictor.predict_observations(request, policy=policy)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                evaluated = torch.ones(views, dtype=torch.bool)
                evaluated[request.camera_ids.index(request.reference_camera)] = False
                case = {"scene": scene_name, "views": views, "source_frames": len(request.frame_indices),
                    "model_frames": max(policy.min_frames, len(request.frame_indices)), "status": "finite",
                    "variant": "baseline" if model_frames == lengths[0] else "extra_padding",
                    "seconds": time.monotonic() - tick, **identity_metrics(result.raw_ids, gt),
                    "side_correct": int((result.view_half_turns[evaluated] == side[evaluated]).sum()), "side_count": int(evaluated.sum()),
                    "side_logits": result.side_logits.tolist(), "predicted_sides": result.view_half_turns.tolist(), "target_sides": side.tolist(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else 0}
                if args.device == "cuda":
                    case["total_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
                    case["free_vram_after_inference_bytes"] = torch.cuda.mem_get_info()[0]
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
                write_json_atomic(args.output, report)
    finite = [case for case in report["cases"] if case["status"] == "finite" and case["variant"] == "baseline"]
    side_total = sum(case["side_count"] for case in finite)
    object_total = sum(case["identity_count"] for case in finite)
    report["side_accuracy"] = sum(case["side_correct"] for case in finite) / side_total if side_total else None
    report["identity_accuracy"] = sum(case["identity_correct"] for case in finite) / object_total if object_total else None
    report["association_metrics_pass"] = bool(side_total and object_total and report["side_accuracy"] >= .95 and report["identity_accuracy"] >= .9 and args.scenes == ["all"])
    report["release_qualified"] = False
    report["release_note"] = "Model-only evaluation cannot certify real-video Court calibration, 3D coverage or complete pipeline E2E"
    report["elapsed_seconds"] = time.time() - started
    write_json_atomic(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("cases", "architecture")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
