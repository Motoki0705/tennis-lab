"""CLI for evaluating Tennis multi-cam 3D pose models and rendering videos.

This script:
- Loads a hierarchical training config (same style as train.py / train_v2.py).
- Uses ConfigLoader to build the TennisPoseDataModule (train/val/test datasets).
- Automatically finds a checkpoint under runs/<experiment_name>/version_*/checkpoints.
- Runs inference on selected splits (default: train + test) and windows.
- Reprojects predicted 3D poses to 2D using camera metadata and renders videos
  via src.visualize.tennis_multi_cam_3d_pose.render_video.

The goal is visual, qualitative inspection of model predictions.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.tennis.geometry.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    NET_HEIGHT_POST,
)
from src.training.utils.config import ConfigLoader, load_cfg
from src.visualize.tennis_multi_cam_3d_pose import render_video


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference for tennis_multi_cam_3d_pose and render predicted videos "
            "for qualitative inspection."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to the top-level YAML config used for training "
            "(e.g. configs/tennis_multi_cam_3d_pose.yaml or _v2.yaml)."
        ),
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Optional config overrides using dot notation.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to evaluate (default: train test).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tennis_eval_videos",
        help="Directory where rendered videos will be saved.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of windows to render per split.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting dataset index per split.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help=(
            "Camera index to render. If negative, the first camera that sees any "
            "player is chosen automatically."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for rendered videos.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Base directory where TensorBoard runs and checkpoints are stored.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=(
            "Optional explicit path to a .ckpt file. If not set, a checkpoint is "
            "auto-discovered under runs/ based on experiment_name."
        ),
    )
    parser.add_argument(
        "--exist-threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold on exist_conf for showing a predicted query as a player "
            "in the rendered video."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help=(
            "Device for inference (e.g. cpu, cuda, cuda:0). "
            "Default: cuda if available, else cpu."
        ),
    )
    return parser.parse_args(argv)


def _select_camera_index(sample: Mapping[str, Tensor], explicit_idx: int) -> int:
    keypoints_2d = sample["keypoints_2d"]
    player_mask = sample["player_mask"]
    _, V, _, _, _ = keypoints_2d.shape
    if explicit_idx >= 0:
        if explicit_idx >= V:
            msg = f"Requested camera_index={explicit_idx} but only V={V} cameras are present"
            raise ValueError(msg)
        return int(explicit_idx)

    mask_any = player_mask.any(dim=0).any(dim=1)  # [V]
    for v in range(V):
        if bool(mask_any[v]):
            return int(v)
    return 0


def _denormalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = points.astype(np.float32).copy()
    if width <= 0 or height <= 0:
        return pts
    pts[..., 0] = (pts[..., 0] + 1.0) * 0.5 * float(width)
    pts[..., 1] = (pts[..., 1] + 1.0) * 0.5 * float(height)
    return pts


def _denorm_pose3d(pose_norm: Tensor) -> Tensor:
    scales = pose_norm.new_tensor(
        [HALF_DOUBLES_WIDTH, HALF_LENGTH, NET_HEIGHT_POST],
        dtype=pose_norm.dtype,
    )
    return pose_norm * scales


def _project_world_points(
    cam_C: Tensor,
    cam_R: Tensor,
    cam_intr: Tensor,
    xyz_world: Tensor,
) -> tuple[Tensor, Tensor]:
    rel = xyz_world - cam_C.view(1, 3)
    Xc = rel @ cam_R.t()
    z = Xc[:, 2]
    mask = z > 1e-6
    z_safe = torch.where(mask, z, torch.ones_like(z))
    f = cam_intr[0]
    cx = cam_intr[1]
    cy = cam_intr[2]
    u = f * (Xc[:, 0] / z_safe) + cx
    v = f * (-Xc[:, 1] / z_safe) + cy
    uv = torch.stack([u, v], dim=-1)
    return uv, mask


def _build_scene_from_prediction(
    sample: Mapping[str, Tensor],
    pose_pred_qt: Tensor,
    exist_mask: Tensor,
    camera_index: int,
    fps: float,
) -> tuple[dict[str, Any], int, int]:
    """Build a simulator-like scene dict from model predictions.

    Args:
        sample (Mapping[str, Tensor]): Single-window sample from TennisSceneWindowDataset.
        pose_pred_qt (Tensor): Predicted pose_3d with shape [Q, T, J, 3].
        exist_mask (Tensor): Bool mask [Q] indicating which queries to render.
        camera_index (int): Requested camera index (or -1 for auto).
        fps (float): Video FPS.

    Returns:
        tuple[dict[str, Any], int, int]: Scene dict, width, and height.

    Raises:
        ValueError: If prediction length doesn't match dataset window or
            if expected joint count is insufficient.

    """
    keypoints_2d = sample["keypoints_2d"]  # [T,V,M,J,2] (shape reference only)
    court_2d = sample["court_2d"]  # [V,20,2]
    camera_C = sample["camera_C"]  # [V,3]
    camera_R = sample["camera_R"]  # [V,3,3]
    camera_intr = sample["camera_intr"]  # [V,3]
    image_size = sample["image_size"]  # [V,2]

    T, V, _, _, _ = keypoints_2d.shape
    Q, T_pred, J, _ = pose_pred_qt.shape
    if T_pred != T:
        msg = f"Prediction length T={T_pred} does not match dataset window_T={T}"
        raise ValueError(msg)
    if J < 20:
        msg = "Expected at least 20 joints (17 body + 3 racket) in predictions"
        raise ValueError(msg)

    v_idx = _select_camera_index(sample, camera_index)
    size_tensor = image_size[v_idx]
    width = int(size_tensor[0].item())
    height = int(size_tensor[1].item())
    if width <= 0 or height <= 0:
        width, height = 1280, 720

    court = court_2d[v_idx].detach().cpu().numpy()
    court_px = _denormalize_points(court, width, height)

    cam_C = camera_C[v_idx].to(device=pose_pred_qt.device, dtype=pose_pred_qt.dtype)
    cam_R = camera_R[v_idx].to(device=pose_pred_qt.device, dtype=pose_pred_qt.dtype)
    cam_intr = camera_intr[v_idx].to(
        device=pose_pred_qt.device, dtype=pose_pred_qt.dtype
    )

    frames: list[dict[str, Any]] = []
    for t in range(T):
        players_joints: list[list[list[float]]] = []
        players_joints_vis: list[list[int]] = []
        players_racket: list[list[list[float]]] = []
        players_racket_vis: list[list[int]] = []

        pose_t = pose_pred_qt[:, t]  # [Q,J,3]
        for q in range(Q):
            if not bool(exist_mask[q].item()):
                continue
            pose_norm = pose_t[q]
            pose_world = _denorm_pose3d(pose_norm)
            uv, vis = _project_world_points(cam_C, cam_R, cam_intr, pose_world)
            uv_np = uv.detach().cpu().numpy().astype("float32")
            vis_np = vis.detach().cpu().numpy().astype("uint8")

            body_uv = uv_np[:17]
            racket_uv = uv_np[17:20]
            body_vis = vis_np[:17].tolist()
            racket_vis = vis_np[17:20].tolist()

            players_joints.append(body_uv.tolist())
            players_joints_vis.append(body_vis)
            players_racket.append(racket_uv.tolist())
            players_racket_vis.append(racket_vis)

        cam_payload = {
            "court_keypoints_2d": {
                "points": court_px.tolist(),
                "visibility": [1] * int(court_px.shape[0]),
            },
            "player_keypoints_2d": {
                "joints": players_joints,
                "visibility": players_joints_vis,
            },
            "racket_keypoints_2d": {
                "points": players_racket,
                "visibility": players_racket_vis,
            },
        }
        frames.append({"cam_0": cam_payload})

    scene = {
        "fps": float(fps),
        "cameras": [{"image_size": [width, height]}],
        "frames": frames,
    }
    return scene, width, height


def _build_scene_from_prediction_for_camera(
    sample: Mapping[str, Tensor],
    pose_pred_qt: Tensor,
    exist_mask: Tensor,
    camera_index: int,
    fps: float,
) -> tuple[dict[str, Any], int, int]:
    keypoints_2d = sample["keypoints_2d"]
    court_2d = sample["court_2d"]
    camera_C = sample["camera_C"]
    camera_R = sample["camera_R"]
    camera_intr = sample["camera_intr"]
    image_size = sample["image_size"]

    T, V, _, _, _ = keypoints_2d.shape
    Q, T_pred, J, _ = pose_pred_qt.shape
    if T_pred != T:
        msg = f"Prediction length T={T_pred} does not match dataset window_T={T}"
        raise ValueError(msg)
    if J < 20:
        msg = "Expected at least 20 joints (17 body + 3 racket) in predictions"
        raise ValueError(msg)
    if not (0 <= camera_index < V):
        msg = f"camera_index {camera_index} out of range (0..{V - 1})"
        raise ValueError(msg)

    size_tensor = image_size[camera_index]
    width = int(size_tensor[0].item())
    height = int(size_tensor[1].item())
    if width <= 0 or height <= 0:
        width, height = 1280, 720

    court = court_2d[camera_index].detach().cpu().numpy()
    court_px = _denormalize_points(court, width, height)

    cam_C = camera_C[camera_index].to(
        device=pose_pred_qt.device,
        dtype=pose_pred_qt.dtype,
    )
    cam_R = camera_R[camera_index].to(
        device=pose_pred_qt.device,
        dtype=pose_pred_qt.dtype,
    )
    cam_intr = camera_intr[camera_index].to(
        device=pose_pred_qt.device,
        dtype=pose_pred_qt.dtype,
    )

    frames: list[dict[str, Any]] = []
    for t in range(T):
        players_joints: list[list[list[float]]] = []
        players_joints_vis: list[list[int]] = []
        players_racket: list[list[list[float]]] = []
        players_racket_vis: list[list[int]] = []

        pose_t = pose_pred_qt[:, t]
        for q in range(Q):
            if not bool(exist_mask[q].item()):
                continue
            pose_norm = pose_t[q]
            pose_world = _denorm_pose3d(pose_norm)
            uv, vis = _project_world_points(cam_C, cam_R, cam_intr, pose_world)
            uv_np = uv.detach().cpu().numpy().astype("float32")
            vis_np = vis.detach().cpu().numpy().astype("uint8")

            body_uv = uv_np[:17]
            racket_uv = uv_np[17:20]
            body_vis = vis_np[:17].tolist()
            racket_vis = vis_np[17:20].tolist()

            players_joints.append(body_uv.tolist())
            players_joints_vis.append(body_vis)
            players_racket.append(racket_uv.tolist())
            players_racket_vis.append(racket_vis)

        cam_payload = {
            "court_keypoints_2d": {
                "points": court_px.tolist(),
                "visibility": [1] * int(court_px.shape[0]),
            },
            "player_keypoints_2d": {
                "joints": players_joints,
                "visibility": players_joints_vis,
            },
            "racket_keypoints_2d": {
                "points": players_racket,
                "visibility": players_racket_vis,
            },
        }
        frames.append({"cam_0": cam_payload})

    scene = {
        "fps": float(fps),
        "cameras": [{"image_size": [width, height]}],
        "frames": frames,
    }
    return scene, width, height


def _auto_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _discover_checkpoints(runs_dir: Path, experiment_name: str) -> list[Path]:
    exp_dir = runs_dir / experiment_name
    if not exp_dir.exists():
        msg = f"Experiment directory not found under runs/: {exp_dir}"
        raise FileNotFoundError(msg)

    version_dirs = sorted(
        d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("version_")
    )
    if not version_dirs:
        msg = f"No version_* directories found under {exp_dir}"
        raise FileNotFoundError(msg)

    candidates: list[Path] = []
    for version_dir in version_dirs:
        ckpt_dir = version_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        for ckpt in sorted(ckpt_dir.glob("*.ckpt")):
            candidates.append(ckpt)

    if not candidates:
        msg = f"No .ckpt files found under {exp_dir}"
        raise FileNotFoundError(msg)
    return candidates


def _select_checkpoint_interactive(candidates: Sequence[Path]) -> Path:
    print("[tennis-eval] Discovered the following checkpoints:")
    for idx, ckpt in enumerate(candidates):
        print(f"  [{idx}] {ckpt}")

    default_idx = len(candidates) - 1
    prompt = f"Select checkpoint index [default: {default_idx}]: "

    try:
        raw = input(prompt).strip()
    except EOFError:
        print(
            f"[tennis-eval] No input received (EOF); "
            f"using default index {default_idx}.",
        )
        return candidates[default_idx]

    if not raw:
        return candidates[default_idx]

    try:
        idx = int(raw)
    except ValueError:
        print(
            f"[tennis-eval] Invalid input '{raw}', using default index {default_idx}.",
        )
        return candidates[default_idx]

    if idx < 0 or idx >= len(candidates):
        print(
            f"[tennis-eval] Index {idx} out of range, "
            f"using default index {default_idx}.",
        )
        return candidates[default_idx]

    return candidates[idx]


def _get_exist_threshold(cfg: Any, cli_value: float) -> float:
    try:
        logging_cfg = cfg.get("logging", {})
        viz_cfg = logging_cfg.get("visualizer", {})
        val = viz_cfg.get("exist_threshold")
        if val is None:
            return float(cli_value)
        return float(val)
    except Exception:
        return float(cli_value)


def main(argv: Sequence[str] | None = None) -> int:
    """Run tennis multi-cam 3D pose evaluation.

    Args:
        argv (Sequence[str] | None): Command line arguments. If None, uses sys.argv.

    Returns:
        int: Exit code (0 for success, non-zero for error).

    Raises:
        SystemExit: If configuration task is not 'tennis_multi_cam_3d_pose'.

    """
    args = _parse_args(argv)

    cfg = load_cfg(args.config, args.overrides)
    task = str(cfg.get("task") or "").strip().lower()
    if task != "tennis_multi_cam_3d_pose":
        msg = (
            "cfg.task must be 'tennis_multi_cam_3d_pose' for this evaluator. "
            "Pass --set task=tennis_multi_cam_3d_pose or use the tennis config."
        )
        raise SystemExit(msg)

    device = _auto_device(args.device)

    loader = ConfigLoader(cfg)
    datamodule = loader.build_datamodule()
    datamodule.setup("fit")
    datamodule.setup("test")

    lit_module = loader.build_lit_module()

    experiment_name = cfg.get("experiment_name") or "tennis_multi_cam_3d_pose"
    runs_dir = Path(args.runs_dir)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        print(f"[tennis-eval] Using checkpoint from --checkpoint: {ckpt_path}")
    else:
        candidates = _discover_checkpoints(runs_dir, str(experiment_name))
        ckpt_path = _select_checkpoint_interactive(candidates)
        print(f"[tennis-eval] Using checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    lit_module.load_state_dict(state["state_dict"], strict=True)
    lit_module.to(device)
    lit_module.eval()

    exist_threshold = _get_exist_threshold(cfg, args.exist_threshold)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    splits = list(dict.fromkeys(args.splits))

    datasets: dict[str, Any] = {}
    if "train" in splits:
        if datamodule.train_dataset is None:
            msg = "train_dataset is not initialized; check dataset config."
            raise SystemExit(msg)
        datasets["train"] = datamodule.train_dataset
    if "val" in splits:
        if datamodule.val_dataset is None:
            msg = "val_dataset is not initialized; check dataset config."
            raise SystemExit(msg)
        datasets["val"] = datamodule.val_dataset
    if "test" in splits:
        if datamodule.test_dataset is None:
            msg = "test_dataset is not initialized; check dataset config."
            raise SystemExit(msg)
        datasets["test"] = datamodule.test_dataset

    with torch.no_grad():
        for split_name, dataset in datasets.items():
            start = int(args.start_index)
            num = int(args.num_samples)
            if start < 0 or num <= 0:
                raise SystemExit("start_index must be >= 0 and num_samples > 0")

            max_index = min(start + num, len(dataset))
            if start >= len(dataset):
                msg = (
                    f"start_index={start} is out of range for split '{split_name}' "
                    f"(len={len(dataset)})"
                )
                raise SystemExit(msg)

            for idx in range(start, max_index):
                sample = dataset[idx]
                batch = {
                    key: (
                        value.unsqueeze(0).to(device)
                        if isinstance(value, Tensor)
                        else value
                    )
                    for key, value in sample.items()
                }
                outputs = lit_module(batch)

                pose_pred = outputs["pose_3d"][0]  # [Q,T,J,3]
                exist_conf = outputs.get("exist_conf")
                if exist_conf is not None and exist_conf.shape[0] > 0:
                    exist_vec = exist_conf[0, :, 0]
                    exist_mask = exist_vec >= float(exist_threshold)
                else:
                    Q = pose_pred.shape[0]
                    exist_mask = torch.ones(
                        Q, dtype=torch.bool, device=pose_pred.device
                    )
                keypoints_2d_sample = sample["keypoints_2d"]
                _, V, _, _, _ = keypoints_2d_sample.shape
                image_size = sample["image_size"]

                cam_arg = int(args.camera_index)
                if cam_arg >= 0:
                    if cam_arg >= V:
                        msg = (
                            f"camera_index={cam_arg} is out of range for split "
                            f"'{split_name}' (V={V})"
                        )
                        raise SystemExit(msg)
                    cam_indices = [cam_arg]
                else:
                    cam_indices = [
                        v
                        for v in range(V)
                        if int(image_size[v, 0].item()) > 0
                        and int(image_size[v, 1].item()) > 0
                    ]
                    if not cam_indices:
                        cam_indices = list(range(V))

                scene_id = (
                    int(sample["scene_id"].item()) if "scene_id" in sample else -1
                )
                t_start = int(sample["t_start"].item()) if "t_start" in sample else 0
                t_end = int(sample["t_end"].item()) if "t_end" in sample else 0

                for v_idx in cam_indices:
                    scene, width, height = _build_scene_from_prediction_for_camera(
                        sample,
                        pose_pred,
                        exist_mask,
                        camera_index=v_idx,
                        fps=float(args.fps),
                    )

                    out_path = (
                        out_root / f"{split_name}_idx{idx:06d}_scene{scene_id}_"
                        f"t{t_start:04d}-{t_end:04d}_cam{v_idx:02d}_pred.mp4"
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    render_video(
                        scene,
                        str(out_path),
                        camera_index=0,
                        width=width,
                        height=height,
                        fps=int(args.fps),
                    )
                    print(f"[tennis-eval] Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
