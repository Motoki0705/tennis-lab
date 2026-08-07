"""Evaluate a ball detection checkpoint on validation and test splits.

Usage:
    python -m src.tasks.ball_detection.scripts.eval
    python -m src.tasks.ball_detection.scripts.eval run.checkpoint_path=path/to/checkpoint.ckpt
    python -m src.tasks.ball_detection.scripts.eval evaluation.splits=[val]

Notes:
    - Hydra loads configuration from `src/tasks/ball_detection/configs/eval.yaml`.
    - The script saves the resolved config and JSON metrics under `run.output_dir`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.tasks.ball_detection import configuration as _configuration  # noqa: F401
from src.tasks.ball_detection.configuration import (
    BallRuntimePaths,
    DetailedEvaluationConfig,
)
from src.tasks.ball_detection.data import build_ball_detection_datamodule
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)
from src.tasks.ball_detection.training.metrics import BallDetectionMetrics
from src.utils.data.heatmaps import heatmaps_to_argmax
from src.utils.device import resolve_device
from src.utils.hydra import hydra_main


@dataclass
class SplitAnalysis:
    """Aggregated split-level detector behavior summary."""

    total_frames: int = 0
    visible_frames: int = 0
    invisible_frames: int = 0
    matched_frames: int = 0
    suppressed_miss_frames: int = 0
    localization_error_frames: int = 0
    absent_false_positive_frames: int = 0
    batch_count: int = 0
    loss_sum: float = 0.0
    visible_peak_values: list[float] = field(default_factory=list)
    invisible_peak_values: list[float] = field(default_factory=list)
    visible_distances_px: list[float] = field(default_factory=list)
    localization_distances_px: list[float] = field(default_factory=list)
    speed_values_px: list[float] = field(default_factory=list)
    speed_match_flags: list[bool] = field(default_factory=list)
    speed_suppressed_flags: list[bool] = field(default_factory=list)
    speed_localization_flags: list[bool] = field(default_factory=list)
    edge_visible_frames: int = 0
    edge_matched_frames: int = 0
    center_visible_frames: int = 0
    center_matched_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert accumulated stats into JSON-serializable values."""
        speed_arr = np.asarray(self.speed_values_px, dtype=np.float32)
        speed_match = np.asarray(self.speed_match_flags, dtype=bool)
        speed_suppressed = np.asarray(self.speed_suppressed_flags, dtype=bool)
        speed_localization = np.asarray(self.speed_localization_flags, dtype=bool)

        speed_summary: dict[str, Any] = {
            "mean_speed_px_per_frame": _safe_mean(self.speed_values_px),
            "matched_mean_speed_px_per_frame": _safe_masked_mean(
                speed_arr, speed_match
            ),
            "suppressed_mean_speed_px_per_frame": _safe_masked_mean(
                speed_arr, speed_suppressed
            ),
            "localization_mean_speed_px_per_frame": _safe_masked_mean(
                speed_arr,
                speed_localization,
            ),
        }
        if speed_arr.size > 0:
            high_speed_threshold = float(np.quantile(speed_arr, 0.75))
            high_speed_mask = speed_arr >= high_speed_threshold
            low_speed_mask = ~high_speed_mask
            speed_summary.update(
                {
                    "high_speed_threshold_px_per_frame_q75": high_speed_threshold,
                    "high_speed_error_rate": _safe_rate(
                        int(
                            (
                                speed_suppressed[high_speed_mask]
                                | speed_localization[high_speed_mask]
                            ).sum()
                        ),
                        int(high_speed_mask.sum()),
                    ),
                    "low_speed_error_rate": _safe_rate(
                        int(
                            (
                                speed_suppressed[low_speed_mask]
                                | speed_localization[low_speed_mask]
                            ).sum()
                        ),
                        int(low_speed_mask.sum()),
                    ),
                }
            )

        return {
            "mean_loss": self.loss_sum / max(self.batch_count, 1),
            "frame_counts": {
                "total": self.total_frames,
                "visible": self.visible_frames,
                "invisible": self.invisible_frames,
            },
            "visible_breakdown": {
                "matched_rate": _safe_rate(self.matched_frames, self.visible_frames),
                "suppressed_miss_rate": _safe_rate(
                    self.suppressed_miss_frames,
                    self.visible_frames,
                ),
                "localization_error_rate": _safe_rate(
                    self.localization_error_frames,
                    self.visible_frames,
                ),
            },
            "absent_frame_false_positive_rate": _safe_rate(
                self.absent_false_positive_frames,
                self.invisible_frames,
            ),
            "peak_value_summary": {
                "visible_mean": _safe_mean(self.visible_peak_values),
                "visible_median": _safe_quantile(self.visible_peak_values, 0.5),
                "invisible_mean": _safe_mean(self.invisible_peak_values),
                "invisible_median": _safe_quantile(self.invisible_peak_values, 0.5),
            },
            "distance_summary_px": {
                "visible_mean": _safe_mean(self.visible_distances_px),
                "visible_p90": _safe_quantile(self.visible_distances_px, 0.9),
                "localization_error_mean": _safe_mean(self.localization_distances_px),
            },
            "edge_summary": {
                "edge_visible_rate": _safe_rate(
                    self.edge_visible_frames, self.visible_frames
                ),
                "edge_match_rate": _safe_rate(
                    self.edge_matched_frames, self.edge_visible_frames
                ),
                "center_match_rate": _safe_rate(
                    self.center_matched_frames, self.center_visible_frames
                ),
            },
            "speed_summary": speed_summary,
        }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _safe_quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float32), quantile))


def _safe_masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    if values.size == 0 or mask.size == 0 or not np.any(mask):
        return None
    return float(values[mask].mean())


def _tensor_to_float_list(tensor: Tensor) -> list[float]:
    if tensor.numel() == 0:
        return []
    return cast(list[float], tensor.detach().float().cpu().reshape(-1).tolist())


def _tensor_to_bool_list(tensor: Tensor) -> list[bool]:
    if tensor.numel() == 0:
        return []
    return cast(list[bool], tensor.detach().cpu().reshape(-1).bool().tolist())


def _move_batch_to_device(
    batch: dict[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _forward_batch(
    module: BallDetectionLightningModule,
    batch: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    images = batch["images"]
    target_heatmaps = batch["heatmaps"]

    model_io = module.model_io
    model_call = model_io.prepare_model_call(images)
    logits = model_io.resized_logits(
        module.model(*model_call.model_args),
        model_call,
        target_size_hw=cast(tuple[int, int], tuple(target_heatmaps.shape[-2:])),
    )

    loss = module.loss_fn(logits, target_heatmaps)
    pred_heatmaps = torch.sigmoid(logits)
    return loss, pred_heatmaps


def _to_original_coords(
    pred_coords_normalized: Tensor, original_size: Tensor
) -> Tensor:
    batch_size = pred_coords_normalized.shape[0]
    original_width = original_size[:, 0].view(batch_size, 1)
    original_height = original_size[:, 1].view(batch_size, 1)

    pred_coords_original = torch.empty_like(pred_coords_normalized)
    pred_coords_original[..., 0] = pred_coords_normalized[..., 0] * torch.clamp(
        original_width - 1.0,
        min=0.0,
    )
    pred_coords_original[..., 1] = pred_coords_normalized[..., 1] * torch.clamp(
        original_height - 1.0,
        min=0.0,
    )
    return pred_coords_original


def _select_primary_targets(
    target_coords: Tensor,
    target_visibility: Tensor,
) -> tuple[Tensor, Tensor]:
    """Select one visible target per frame for legacy diagnostic summaries."""
    visible = target_visibility > 0.5
    frame_visible = visible.any(dim=-1)
    primary_indices = visible.to(torch.int64).argmax(dim=-1)
    gather_indices = primary_indices[..., None, None].expand(
        *primary_indices.shape,
        1,
        2,
    )
    primary_coords = target_coords.gather(dim=2, index=gather_indices).squeeze(2)
    primary_coords = torch.where(
        frame_visible[..., None],
        primary_coords,
        torch.zeros_like(primary_coords),
    )
    return primary_coords, frame_visible


def _compute_edge_mask(
    target_coords: Tensor,
    original_size: Tensor,
    target_visible: Tensor,
    edge_threshold_ratio: float,
) -> Tensor:
    width = torch.clamp(original_size[:, 0].view(-1, 1) - 1.0, min=1.0)
    height = torch.clamp(original_size[:, 1].view(-1, 1) - 1.0, min=1.0)

    x_norm = torch.clamp(target_coords[..., 0] / width, min=0.0, max=1.0)
    y_norm = torch.clamp(target_coords[..., 1] / height, min=0.0, max=1.0)
    distance_to_edge = torch.minimum(
        torch.minimum(x_norm, 1.0 - x_norm),
        torch.minimum(y_norm, 1.0 - y_norm),
    )
    return target_visible & (distance_to_edge < edge_threshold_ratio)


def _compute_speed_px(
    target_coords: Tensor, target_visible: Tensor
) -> tuple[Tensor, Tensor]:
    speed_px = torch.zeros_like(target_visible, dtype=target_coords.dtype)
    valid_mask = torch.zeros_like(target_visible, dtype=torch.bool)
    if target_coords.shape[1] < 2:
        return speed_px, valid_mask

    delta = torch.norm(target_coords[:, 1:] - target_coords[:, :-1], dim=-1)
    visible_pairs = target_visible[:, 1:] & target_visible[:, :-1]
    speed_px[:, 1:] = delta
    valid_mask[:, 1:] = visible_pairs
    return speed_px, valid_mask


def _collect_split_result(
    module: BallDetectionLightningModule,
    dataloader: torch.utils.data.DataLoader,
    cfg: DictConfig,
    evaluation: DetailedEvaluationConfig,
    split_name: str,
    device: torch.device,
) -> dict[str, Any]:
    analysis = SplitAnalysis()
    metrics = BallDetectionMetrics(
        peak_threshold=float(cfg.metrics.peak_threshold),
        ball_distance_threshold=float(cfg.metrics.ball_distance_threshold),
        nms_kernel=int(cfg.metrics.nms_kernel),
        max_predictions_per_frame=int(cfg.metrics.max_predictions_per_frame),
        subpixel_refine=bool(cfg.metrics.subpixel_refine),
    ).to(device)

    module.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if (
                evaluation.max_batches_per_split is not None
                and batch_idx >= evaluation.max_batches_per_split
            ):
                break

            batch_on_device = _move_batch_to_device(batch, device)
            loss, pred_heatmaps = _forward_batch(module, batch_on_device)
            metrics.update(
                pred_heatmaps,
                batch_on_device["coords"],
                batch_on_device["visibility"],
                batch_on_device["original_size"],
            )
            primary_coords, target_visible = _select_primary_targets(
                batch_on_device["coords"],
                batch_on_device["visibility"],
            )

            pred_coords_normalized, peak_values = heatmaps_to_argmax(pred_heatmaps)
            pred_coords_original = _to_original_coords(
                pred_coords_normalized,
                batch_on_device["original_size"],
            )
            distances_px = torch.norm(pred_coords_original - primary_coords, dim=-1)

            pred_visible = peak_values > float(cfg.metrics.peak_threshold)
            matched = (
                pred_visible
                & target_visible
                & (distances_px < float(cfg.metrics.ball_distance_threshold))
            )
            suppressed = (~pred_visible) & target_visible
            localization = pred_visible & target_visible & ~matched
            absent_false_positive = pred_visible & ~target_visible
            edge_mask = _compute_edge_mask(
                primary_coords,
                batch_on_device["original_size"],
                target_visible,
                edge_threshold_ratio=evaluation.edge_threshold_ratio,
            )
            center_mask = target_visible & ~edge_mask
            speed_px, speed_valid_mask = _compute_speed_px(
                primary_coords, target_visible
            )
            speed_mask = speed_valid_mask & target_visible

            analysis.total_frames += int(peak_values.numel())
            analysis.visible_frames += int(target_visible.sum().item())
            analysis.invisible_frames += int((~target_visible).sum().item())
            analysis.matched_frames += int(matched.sum().item())
            analysis.suppressed_miss_frames += int(suppressed.sum().item())
            analysis.localization_error_frames += int(localization.sum().item())
            analysis.absent_false_positive_frames += int(
                absent_false_positive.sum().item()
            )
            analysis.batch_count += 1
            analysis.loss_sum += float(loss.detach().cpu().item())

            analysis.visible_peak_values.extend(
                _tensor_to_float_list(peak_values[target_visible])
            )
            analysis.invisible_peak_values.extend(
                _tensor_to_float_list(peak_values[~target_visible])
            )
            analysis.visible_distances_px.extend(
                _tensor_to_float_list(distances_px[target_visible])
            )
            analysis.localization_distances_px.extend(
                _tensor_to_float_list(distances_px[localization])
            )
            analysis.edge_visible_frames += int(edge_mask.sum().item())
            analysis.edge_matched_frames += int((matched & edge_mask).sum().item())
            analysis.center_visible_frames += int(center_mask.sum().item())
            analysis.center_matched_frames += int((matched & center_mask).sum().item())
            analysis.speed_values_px.extend(_tensor_to_float_list(speed_px[speed_mask]))
            analysis.speed_match_flags.extend(_tensor_to_bool_list(matched[speed_mask]))
            analysis.speed_suppressed_flags.extend(
                _tensor_to_bool_list(suppressed[speed_mask])
            )
            analysis.speed_localization_flags.extend(
                _tensor_to_bool_list(localization[speed_mask])
            )

    split_metrics = {
        name: float(value.detach().cpu().item())
        for name, value in metrics.compute().items()
    }
    return {
        "split": split_name,
        "metrics": split_metrics,
        "analysis": analysis.to_dict(),
    }


def _build_dataloader(
    datamodule: pl.LightningDataModule,
    split_name: str,
) -> torch.utils.data.DataLoader:
    if split_name == "val":
        datamodule.setup(stage="fit")
        dataloader = datamodule.val_dataloader()
        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise TypeError("Ball validation dataloader must be a DataLoader.")
        return dataloader
    if split_name == "test":
        datamodule.setup(stage="test")
        dataloader = datamodule.test_dataloader()
        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise TypeError("Ball test dataloader must be a DataLoader.")
        return dataloader
    raise ValueError(f"Unsupported split: {split_name}")


def _print_split_summary(split_result: dict[str, Any]) -> None:
    split_name = str(split_result["split"])
    metrics = split_result["metrics"]
    analysis = split_result["analysis"]
    visible_breakdown = analysis["visible_breakdown"]
    edge_summary = analysis["edge_summary"]
    speed_summary = analysis["speed_summary"]

    print(
        f"[{split_name}] precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"mean_distance_px={metrics['mean_distance_px']:.4f}"
    )
    print(
        f"[{split_name}] matched_rate={_format_optional(visible_breakdown['matched_rate'])} "
        f"suppressed_miss_rate={_format_optional(visible_breakdown['suppressed_miss_rate'])} "
        f"localization_error_rate={_format_optional(visible_breakdown['localization_error_rate'])}"
    )
    print(
        f"[{split_name}] absent_false_positive_rate="
        f"{_format_optional(analysis['absent_frame_false_positive_rate'])} "
        f"edge_match_rate={_format_optional(edge_summary['edge_match_rate'])} "
        f"center_match_rate={_format_optional(edge_summary['center_match_rate'])}"
    )
    print(
        f"[{split_name}] high_speed_error_rate={_format_optional(speed_summary.get('high_speed_error_rate'))} "
        f"low_speed_error_rate={_format_optional(speed_summary.get('low_speed_error_rate'))}"
    )


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


@hydra_main(
    config_path="../configs",
    config_name="eval",
    version_base="1.3",
    validation_boundary="ball.eval",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point for ball detection evaluation."""
    paths = BallRuntimePaths.from_config(cfg)
    evaluation = DetailedEvaluationConfig.from_config(cfg)
    output_dir = paths.output(str(cfg.run.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    pl.seed_everything(int(cfg.run.seed))

    checkpoint_path = paths.checkpoint(str(cfg.run.checkpoint_path))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    requested_device = "cuda:0" if int(cfg.run.gpus) > 0 else "cpu"
    device = resolve_device(requested_device)
    torch.set_float32_matmul_precision(str(cfg.training.matmul_precision))

    datamodule = build_ball_detection_datamodule(cfg)
    module = BallDetectionLightningModule.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
        strict=bool(cfg.run.strict),
        weights_only=bool(cfg.run.weights_only),
    )
    module.to(device)
    module.eval()

    split_results: dict[str, Any] = {}
    for split_name in evaluation.splits:
        dataloader = _build_dataloader(datamodule, split_name)
        split_result = _collect_split_result(
            module, dataloader, cfg, evaluation, split_name, device
        )
        split_results[split_name] = split_result
        _print_split_summary(split_result)

    result = {
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "peak_threshold": float(cfg.metrics.peak_threshold),
        "ball_distance_threshold": float(cfg.metrics.ball_distance_threshold),
        "splits": split_results,
    }
    result_path = output_dir / evaluation.output_json_name
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved evaluation summary to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
