"""Run offline inference for the ball multi-task model.

Example:
    `uv run python -m src.ball_multitask.scripts.predict`
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.ball_multitask.inference.predictor import BallMultitaskPredictor
from src.common.data.scene_cache import load_npz_scene


def _load_scene_inputs(scene_path: Path, camera: int) -> dict[str, torch.Tensor]:
    scene = load_npz_scene(scene_path)
    meta = scene.get("meta", {}) if isinstance(scene.get("meta", {}), dict) else {}

    num_cameras = int(scene.get("num_cameras", 1))
    cam_idx = min(max(int(camera), 0), max(num_cameras - 1, 0))
    prefix = f"cam_{cam_idx}_"

    ball_uv = torch.from_numpy(scene[f"{prefix}ball_uv"]).float()
    ball_vis = torch.from_numpy(scene[f"{prefix}ball_visible"]).float()
    court_kp = torch.from_numpy(scene[f"{prefix}court_kp_uv"]).float()
    court_vis = torch.from_numpy(scene[f"{prefix}court_kp_visible"]).float()

    seq_len = int(meta.get("num_frames", ball_uv.shape[0]))
    seq_len = min(seq_len, int(ball_uv.shape[0]))

    return {
        "ball_uv": ball_uv[:seq_len],
        "ball_vis": ball_vis[:seq_len],
        "court_kp": court_kp,
        "court_vis": court_vis,
        "seq_len": torch.tensor(seq_len, dtype=torch.long),
    }


def _save_outputs(output_path: Path, outputs: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        uv_completed=outputs["uv_completed"],
        position_3d=outputs["position_3d"],
        event_logits=outputs["event_logits"],
        event_probs=outputs["event_probs"],
    )
    peaks_path = output_path.with_suffix(".events.json")
    peaks_payload = {
        "event_names": outputs["event_names"],
        "event_peaks": outputs["event_peaks"],
        "event_peak_scores": outputs["event_peak_scores"],
    }
    peaks_path.write_text(json.dumps(peaks_payload, indent=2))


@hydra.main(config_path="../configs", config_name="predict", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    infer_cfg = cfg.get("inference", {}) or {}
    scene_path = Path(str(infer_cfg.get("scene_path", "data/blcs/scenes/rally_000000.npz")))
    checkpoint = str(infer_cfg.get("checkpoint", ""))
    if not checkpoint:
        raise ValueError("inference.checkpoint is required")

    predictor = BallMultitaskPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint,
        device=str(infer_cfg.get("device", "cpu")),
    )

    inputs = _load_scene_inputs(scene_path, camera=int(infer_cfg.get("camera", 0)))
    outputs = predictor.predict(
        inputs["ball_uv"],
        inputs["court_kp"],
        ball_vis=inputs.get("ball_vis"),
        court_vis=inputs.get("court_vis"),
        seq_len=inputs.get("seq_len"),
        threshold=float(infer_cfg.get("threshold", 0.5)),
        min_distance=int(infer_cfg.get("min_distance", 1)),
        top_k=infer_cfg.get("top_k"),
        denormalize=bool(infer_cfg.get("denormalize", True)),
    )

    output_path = infer_cfg.get("output")
    if output_path:
        _save_outputs(Path(str(output_path)), outputs)
        print(f"Saved outputs to {output_path}")
    else:
        print("Inference complete.")
        print(f"uv_completed: {tuple(outputs['uv_completed'].shape)}")
        print(f"position_3d: {tuple(outputs['position_3d'].shape)}")
        print(f"event_logits: {tuple(outputs['event_logits'].shape)}")


if __name__ == "__main__":
    main()
