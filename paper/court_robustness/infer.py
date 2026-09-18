"""CPU-only, immutable-checkpoint comparison; preserve raw outputs for auditing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from common import MAIN, REPO, ROOT, sha256, sources, write_json

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

BASELINE = REPO / ".cache/court-report/TennisCourtDetector"
CHECKPOINT = (
    MAIN
    / "outputs/court_detection/multiscale-depth3-local-rtx/logs/version_4/checkpoints/court-detection-epoch=17.ckpt"
)
CHECKPOINT_SHA256 = "e11b494366a2be4266f3f034973ef08c6a91530530d116764689ec35b2843455"
BASELINE_SHA256 = "09aa8c4338459ba1d643f2dc329f45f464dedec3720fccc1a4abfd1f7b464d04"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINE))
from court_reference import CourtReference  # noqa: E402
from postprocess import postprocess, refine_kps  # noqa: E402
from tracknet import BallTrackerNet  # noqa: E402

REFERENCE = CourtReference()
REF_KP = np.asarray(REFERENCE.key_points, np.float32)
EDGES = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]


def fit_court(points: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Upstream 12-configuration fitting, with NumPy/SciPy compatibility fixes.

    Same mean held-out-point criterion as upstream homography.py. Reject a
    singular fit or a configuration with no held-out observation explicitly.
    """
    best, best_error = None, float("inf")
    valid = np.isfinite(points).all(axis=1)
    for conf in REFERENCE.court_conf.values():
        indices = [REFERENCE.key_points.index(point) for point in conf]
        if not valid[indices].all():
            continue
        matrix, _ = cv2.findHomography(REF_KP[indices], points[indices], method=0)
        if (
            matrix is None
            or not np.isfinite(matrix).all()
            or np.linalg.matrix_rank(matrix) < 3
        ):
            continue
        predicted = cv2.perspectiveTransform(REF_KP[:, None, :], matrix)[:, 0]
        other = [i for i in range(12) if i not in indices and valid[i]]
        if not other:
            continue
        error = np.linalg.norm(points[other] - predicted[other], axis=1).mean()
        if error < best_error:
            best, best_error = matrix, float(error)
    if best is None:
        return points.copy(), None
    return cv2.perspectiveTransform(REF_KP[:, None, :], best)[:, 0], best


def infer_baseline(image: np.ndarray, model: torch.nn.Module) -> dict[str, np.ndarray]:
    height, width = image.shape[:2]
    # Published BGR / [0,1] input and exact 640 x 360 network resolution.
    resized = cv2.resize(image[:, :, ::-1], (640, 360))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float()[None] / 255
    with torch.inference_mode():
        probability = model(tensor)[0].sigmoid().numpy()
    original_720 = cv2.resize(image[:, :, ::-1], (1280, 720))
    raw = np.full((14, 2), np.nan)
    refined = raw.copy()
    for k in range(14):
        x, y = postprocess(
            (probability[k] * 255).astype(np.uint8), low_thresh=170, max_radius=25
        )
        if x is None or y is None:
            continue
        raw[k] = [x, y]
        if k not in [8, 12, 9] and x and y:
            x, y = refine_kps(original_720, int(y), int(x))
        refined[k] = [x, y]
    fitted, matrix = fit_court(refined)
    scale = np.asarray([width / 1280, height / 720])
    flat = probability[:14].reshape(14, -1).argmax(1)
    argmax_points = np.column_stack((flat % 640, flat // 640)).astype(float) * 2
    argmax_fitted, argmax_matrix = fit_court(argmax_points)
    return {
        "raw_kp": raw * scale,
        "refined_kp": refined * scale,
        "aligned_kp": fitted * scale,
        "homography_found": np.asarray(matrix is not None),
        "kp_probability": probability[:14],
        "input_hw": np.asarray([360, 640]),
        "argmax_kp": argmax_points * scale,
        "argmax_aligned_kp": argmax_fitted * scale,
        "argmax_homography_found": np.asarray(argmax_matrix is not None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--model", choices=["both", "baseline", "ours"], default="both")
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(42)
    records = sources()
    records = [r for r in records if not args.ids or r["id"] in args.ids]
    output = ROOT / "evidence/predictions"
    output.mkdir(exist_ok=True)
    metadata = {
        "device": "cpu",
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "seed": 42,
        "ids": [r["id"] for r in records],
        "checkpoint": str(CHECKPOINT),
    }
    if args.model in {"both", "baseline"}:
        model = BallTrackerNet(out_channels=15).cpu().eval()
        baseline_path = REPO / ".cache/court-report/baseline.pth"
        if sha256(baseline_path) != BASELINE_SHA256:
            raise ValueError("TCD checkpoint differs from the pinned paper weights")
        model.load_state_dict(
            torch.load(baseline_path, map_location="cpu", weights_only=True),
            strict=True,
        )
        metadata["baseline_sha256"] = sha256(baseline_path)
        metadata["baseline_checkpoint"] = str(baseline_path)
        metadata["baseline_code_commit"] = subprocess.check_output(
            ["git", "-C", str(BASELINE), "rev-parse", "HEAD"], text=True
        ).strip()
        if (
            metadata["baseline_code_commit"]
            != "e5cd4f1ce26b15361700d3d89e068cbf0e82749e"
        ):
            raise ValueError("The paper requires the pinned upstream TCD code version")
        for record in records:
            start = time.monotonic()
            image = np.asarray(
                ImageOps.exif_transpose(
                    Image.open(ROOT / record["paper_path"])
                ).convert("RGB")
            )
            result = infer_baseline(image, model)
            np.savez_compressed(output / f"{record['id']}_baseline.npz", **result)
            print(
                "baseline",
                record["id"],
                "detected",
                np.isfinite(result["raw_kp"]).all(1).sum(),
                "H",
                result["homography_found"],
                "seconds",
                round(time.monotonic() - start, 2),
                flush=True,
            )
        control_record = json.loads((MAIN / "data/court/data_val.json").read_text())[0]
        control_path = next(
            (MAIN / "data/court/images").glob(control_record["id"] + ".*")
        )
        control_image = np.asarray(Image.open(control_path).convert("RGB"))
        control_result = infer_baseline(control_image, model)
        control_gt = np.asarray(control_record["kps"])
        np.savez_compressed(
            ROOT / "evidence/control_baseline.npz", **control_result, gt=control_gt
        )
        metadata["control_sample"] = control_record["id"]
        metadata["control_mean_error_px"] = float(
            np.linalg.norm(control_result["aligned_kp"] - control_gt, axis=1).mean()
        )
        del model
    if args.model in {"both", "ours"}:
        from omegaconf import OmegaConf, open_dict

        from src.tasks.court_detection.model_io.images import prepare_court_image
        from src.tasks.court_detection.training.lightning_module import (
            CourtDetectionLightningModule,
        )

        if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint differs from the pinned paper weights")
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint changed while loading")
        config = checkpoint["hyper_parameters"]["config"]
        write_json(
            ROOT / "evidence/checkpoint_config.json",
            OmegaConf.to_container(config, resolve=True),
        )
        write_json(
            ROOT / "evidence/target_bundle.json",
            checkpoint["hyper_parameters"]["target_bundle_state"],
        )
        metadata["checkpoint_train_scales"] = list(
            config.data.augmentation.train_scales
        )
        metadata["inference_config_compatibility"] = (
            "Current train-only pose validator requires train_scales=[val_short_side]. Set only this unused training field to instantiate. Model, state_dict, validation resolution, normalization and inference unchanged; no training performed."
        )
        with open_dict(config):
            config.paths.project_root = str(REPO)
            config.paths.external_asset_root = str(MAIN / "third_party")
            config.paths.data_root = str(MAIN / "data")
            config.training.compile.enabled = False
            config.data.augmentation.train_scales = [
                config.data.augmentation.val_short_side
            ]
        module = CourtDetectionLightningModule(
            config=config,
            target_bundle_state=checkpoint["hyper_parameters"]["target_bundle_state"],
        )
        module.load_state_dict(checkpoint["state_dict"], strict=True)
        module.cpu().eval()
        metadata["epoch"] = checkpoint["epoch"]
        metadata["global_step"] = checkpoint["global_step"]
        metadata["ours_sha256"] = CHECKPOINT_SHA256
        metadata["short_side"] = module.model_io.spec.short_side
        del checkpoint
        for record in records:
            start = time.monotonic()
            image = ImageOps.exif_transpose(
                Image.open(ROOT / record["paper_path"])
            ).convert("RGB")
            tensor, h, w = prepare_court_image(
                image,
                short_side=module.model_io.spec.short_side,
                device=torch.device("cpu"),
            )
            with torch.inference_mode():
                call = module.model_io.prepare_images(tensor)
                raw = module.model(*call.model_args)
                module.model_io.validate_logits(raw, call)
                logits = raw.dense_logits
                kp = module.model_io.decode_prediction(
                    "kp",
                    logits["kp"],
                    original_size_hw=(h, w),
                    subpixel_refine=True,
                    max_peaks=1,
                )
            points = kp.keypoints[:, 0].numpy()
            points[~kp.valid[:, 0].numpy()] = np.nan
            # Use the same 12-configuration criterion, at the same fitting scale.
            scale = np.asarray([w / 1280, h / 720])
            fitted, matrix = fit_court(points / scale)
            result = {
                "raw_kp": points,
                "aligned_kp": fitted * scale,
                "homography_found": np.asarray(matrix is not None),
                "kp_scores": kp.scores[:, 0].numpy(),
                "line_probability": logits["line"][0, 0].sigmoid().numpy(),
                "input_hw": np.asarray(tensor.shape[-2:]),
                "pose_raw": raw.pose.values[0].numpy(),
            }
            np.savez_compressed(output / f"{record['id']}_ours.npz", **result)
            print(
                "ours",
                record["id"],
                "detected",
                np.isfinite(points).all(1).sum(),
                "H",
                matrix is not None,
                "seconds",
                round(time.monotonic() - start, 2),
                flush=True,
            )
    metadata["cuda_initialized"] = torch.cuda.is_initialized()
    if metadata["cuda_initialized"]:
        raise RuntimeError("This comparison must remain CPU-only.")
    if args.model in {"both", "ours"} and sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint changed during inference")
    if args.model in {"both", "baseline"} and sha256(baseline_path) != BASELINE_SHA256:
        raise ValueError("TCD checkpoint changed during inference")
    metadata["configuration_sha256"] = {
        name: sha256(ROOT / "evidence" / name)
        for name in ("checkpoint_config.json", "target_bundle.json")
        if args.model in {"both", "ours"}
    }
    metadata["input_manifest_sha256"] = sha256(ROOT / "evidence/inputs.json")
    metadata["inputs"] = {r["id"]: r["sha256"] for r in records}
    metadata["prediction_sha256"] = {
        p.name: sha256(p) for p in sorted(output.glob("*.npz"))
    }
    write_json(ROOT / f"evidence/inference_{args.model}.json", metadata)


if __name__ == "__main__":
    main()
