"""Ensemble heatmap-based ball predictor for WASB models (inference).

This predictor keeps per-frame heatmaps as distributions, applies TTA (with
inverse warp back to a shared grid), calibrates logits, fuses models via PoE,
and smooths the sequence with a forward-backward filter. Final coordinates
are decoded from the smoothed distribution (expected/MAP/fit).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from src.base.api import BasePredictor

if TYPE_CHECKING:
    from src.wasb.training import WASBLightningModule

_F = TypeVar("_F", bound=Callable[..., Any])


def _torch_no_grad(func: _F) -> _F:
    return cast(_F, torch.no_grad()(func))


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def _as_tuple_hw(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    return (int(value[0]), int(value[1]))


def _expand_per_model(value: Any, n: int, *, name: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} must have {n} entries, got {len(value)}")
        return [float(v) for v in value]
    return [float(value) for _ in range(n)]


def _normalize_heatmap(prob: Tensor, eps: float) -> Tensor:
    denom = prob.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return prob / denom


def _uniform_heatmap(hw: tuple[int, int], device: torch.device, dtype: torch.dtype) -> Tensor:
    h, w = hw
    base = torch.full((1, h, w), 1.0 / float(h * w), device=device, dtype=dtype)
    return base


def _entropy_score(prob: Tensor, eps: float) -> Tensor:
    ent = -(prob * (prob + eps).log()).sum(dim=(-2, -1))
    max_ent = math.log(float(prob.shape[-2] * prob.shape[-1]))
    return ent / max_ent


def _peak_score(prob: Tensor) -> Tensor:
    return prob.view(prob.shape[0], -1).max(dim=-1).values


def _parse_tta_transforms(tta_cfg: Any) -> list[dict[str, Any]]:
    enabled = bool(_cfg_get(tta_cfg, "enabled", False))
    if not enabled:
        return [{"type": "identity"}]
    transforms = _cfg_get(tta_cfg, "transforms", None)
    if not transforms:
        return [{"type": "identity"}]
    parsed: list[dict[str, Any]] = []
    for spec in transforms:
        spec = dict(spec)
        spec_type = str(spec.get("type", "identity"))
        spec["type"] = spec_type
        parsed.append(spec)
    return parsed


def _affine_matrix_from_params(
    *,
    angle_deg: float,
    translate: tuple[float, float],
    scale: float,
    shear_deg: tuple[float, float],
    center: tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    angle = math.radians(angle_deg)
    shear_x = math.radians(shear_deg[0])
    shear_y = math.radians(shear_deg[1])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    shx = math.tan(shear_x)
    shy = math.tan(shear_y)
    cx, cy = center
    tx, ty = translate

    t_center = torch.tensor(
        [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], device=device, dtype=dtype
    )
    t_center_inv = torch.tensor(
        [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], device=device, dtype=dtype
    )
    t_translate = torch.tensor(
        [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], device=device, dtype=dtype
    )
    s_mat = torch.tensor(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
    )
    sh_mat = torch.tensor(
        [[1.0, shx, 0.0], [shy, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
    )
    r_mat = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return t_translate @ t_center_inv @ r_mat @ sh_mat @ s_mat @ t_center


def _build_affine_grid(
    matrix: Tensor,
    *,
    out_hw: tuple[int, int],
    in_hw: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    h_out, w_out = out_hw
    h_in, w_in = in_hw
    ys, xs = torch.meshgrid(
        torch.arange(h_out, device=device, dtype=dtype),
        torch.arange(w_out, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    coords = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3).t()
    inv = torch.linalg.inv(matrix)
    src = inv @ coords
    x = src[0].reshape(h_out, w_out)
    y = src[1].reshape(h_out, w_out)
    x_norm = (2.0 * x + 1.0) / float(w_in) - 1.0
    y_norm = (2.0 * y + 1.0) / float(h_in) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def _apply_affine(
    tensor: Tensor,
    matrix: Tensor,
    *,
    out_hw: tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tensor:
    b, c, h_in, w_in = tensor.shape
    grid = _build_affine_grid(
        matrix,
        out_hw=out_hw,
        in_hw=(h_in, w_in),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(
        tensor,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )


def _apply_tta_transform(frames: Tensor, spec: dict[str, Any]) -> Tensor:
    t_type = spec.get("type", "identity")
    if t_type == "identity":
        return frames
    if t_type == "hflip":
        return torch.flip(frames, dims=[-1])
    if t_type == "vflip":
        return torch.flip(frames, dims=[-2])
    if t_type == "affine":
        angle = float(spec.get("angle", 0.0))
        translate = spec.get("translate", [0.0, 0.0])
        translate_xy = (float(translate[0]), float(translate[1]))
        scale = float(spec.get("scale", 1.0))
        shear = spec.get("shear", [0.0, 0.0])
        shear_xy = (float(shear[0]), float(shear[1]))
        h, w = frames.shape[-2], frames.shape[-1]
        center = (float(w - 1) / 2.0, float(h - 1) / 2.0)
        matrix = _affine_matrix_from_params(
            angle_deg=angle,
            translate=translate_xy,
            scale=scale,
            shear_deg=shear_xy,
            center=center,
            device=frames.device,
            dtype=frames.dtype,
        )
        return _apply_affine(frames, matrix, out_hw=(h, w))
    raise ValueError(f"Unknown TTA transform type: {t_type}")


def _inverse_warp_heatmap(
    heatmap: Tensor, spec: dict[str, Any], *, target_hw: tuple[int, int]
) -> Tensor:
    t_type = spec.get("type", "identity")
    if t_type == "identity":
        return heatmap
    if t_type == "hflip":
        return torch.flip(heatmap, dims=[-1])
    if t_type == "vflip":
        return torch.flip(heatmap, dims=[-2])
    if t_type == "affine":
        angle = float(spec.get("angle", 0.0))
        translate = spec.get("translate", [0.0, 0.0])
        translate_xy = (float(translate[0]), float(translate[1]))
        scale = float(spec.get("scale", 1.0))
        shear = spec.get("shear", [0.0, 0.0])
        shear_xy = (float(shear[0]), float(shear[1]))
        h, w = heatmap.shape[-2], heatmap.shape[-1]
        center = (float(w - 1) / 2.0, float(h - 1) / 2.0)
        matrix = _affine_matrix_from_params(
            angle_deg=angle,
            translate=translate_xy,
            scale=scale,
            shear_deg=shear_xy,
            center=center,
            device=heatmap.device,
            dtype=heatmap.dtype,
        )
        inv = torch.linalg.inv(matrix)
        warped = _apply_affine(
            heatmap.unsqueeze(1),
            inv,
            out_hw=target_hw,
            mode="bilinear",
            padding_mode="zeros",
        )
        return warped.squeeze(1)
    raise ValueError(f"Unknown TTA transform type: {t_type}")


def _build_kernel(kernel_cfg: Any, device: torch.device, dtype: torch.dtype) -> Tensor:
    kernel_type = str(_cfg_get(kernel_cfg, "type", "disk"))
    if kernel_type == "disk":
        radius = int(_cfg_get(kernel_cfg, "radius", 4))
        size = radius * 2 + 1
        yy, xx = torch.meshgrid(
            torch.arange(size, device=device, dtype=dtype),
            torch.arange(size, device=device, dtype=dtype),
            indexing="ij",
        )
        cy = radius
        cx = radius
        dist = (yy - cy) ** 2 + (xx - cx) ** 2
        kernel = (dist <= float(radius * radius)).to(dtype=dtype)
    elif kernel_type == "gaussian":
        sigma = float(_cfg_get(kernel_cfg, "sigma", 2.0))
        radius = int(math.ceil(3.0 * sigma))
        size = radius * 2 + 1
        yy, xx = torch.meshgrid(
            torch.arange(size, device=device, dtype=dtype),
            torch.arange(size, device=device, dtype=dtype),
            indexing="ij",
        )
        cy = radius
        cx = radius
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        kernel = torch.exp(-dist2 / (2.0 * sigma * sigma))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel


def _convolve_heatmap(prob: Tensor, kernel: Tensor) -> Tensor:
    k = kernel.unsqueeze(0).unsqueeze(0)
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    out = F.conv2d(prob.unsqueeze(1), k, padding=(pad_y, pad_x))
    return out.squeeze(1)


def _forward_backward_smooth(
    emissions: Tensor,
    *,
    kernel: Tensor,
    accel_weight: float,
    warmup_tau: float,
    forward_state: Tensor | None,
    eps: float,
) -> tuple[Tensor, Tensor]:
    b, h, w = emissions.shape
    device = emissions.device
    dtype = emissions.dtype
    uniform = _uniform_heatmap((h, w), device=device, dtype=dtype).squeeze(0)
    forward = torch.empty_like(emissions)
    prev = forward_state if forward_state is not None else uniform
    prev2: Tensor | None = None
    for t in range(b):
        pred = _convolve_heatmap(prev.unsqueeze(0), kernel).squeeze(0)
        if accel_weight > 0.0 and prev2 is not None:
            pred2 = _convolve_heatmap(
                _convolve_heatmap(prev2.unsqueeze(0), kernel), kernel
            ).squeeze(0)
            pred = (1.0 - accel_weight) * pred + accel_weight * pred2
        pred = _normalize_heatmap(pred.unsqueeze(0), eps).squeeze(0)
        ft = emissions[t] * pred
        ft = _normalize_heatmap(ft.unsqueeze(0), eps).squeeze(0)
        forward[t] = ft
        prev2 = prev
        prev = ft

    backward = torch.empty_like(emissions)
    beta = uniform
    for t in range(b - 1, -1, -1):
        if t < b - 1:
            nxt = emissions[t + 1] * beta
            beta = _convolve_heatmap(nxt.unsqueeze(0), kernel).squeeze(0)
            beta = _normalize_heatmap(beta.unsqueeze(0), eps).squeeze(0)
        backward[t] = beta

    smoothed = _normalize_heatmap(forward * backward, eps)
    if warmup_tau > 0.0:
        idx = torch.arange(b, device=device, dtype=dtype)
        alpha = torch.exp(-idx / float(warmup_tau))
        alpha = alpha.view(b, 1, 1)
        smoothed = (1.0 - alpha) * smoothed + alpha * uniform.unsqueeze(0)
    return smoothed, prev


def _softmax_spatial(logp: Tensor) -> Tensor:
    b, h, w = logp.shape
    flat = logp.view(b, -1)
    prob = torch.softmax(flat, dim=-1).view(b, h, w)
    return prob


def _fit_offset_1d(values: Tensor, *, mode: str, eps: float) -> float:
    if mode == "gaussian":
        values = torch.clamp(values, min=eps).log()
    elif mode != "quadratic":
        raise ValueError(f"Unknown fit_mode: {mode}")
    positions = torch.arange(values.shape[0], device=values.device, dtype=values.dtype)
    positions = positions - float(values.shape[0] // 2)
    a = torch.stack([positions ** 2, positions, torch.ones_like(positions)], dim=1)
    coeff, *_ = torch.linalg.lstsq(a, values)
    a_coef = coeff[0].item()
    b_coef = coeff[1].item()
    if abs(a_coef) < eps:
        return 0.0
    return float(-b_coef / (2.0 * a_coef))


def _fit_peak(
    heatmap: Tensor,
    *,
    fit_mode: str,
    fit_window: int,
    eps: float,
) -> Tensor:
    b, h, w = heatmap.shape
    offsets = torch.zeros((b, 2), device=heatmap.device, dtype=heatmap.dtype)
    half = fit_window // 2
    flat = heatmap.view(b, -1)
    idx = torch.argmax(flat, dim=-1)
    ys = (idx // w).to(dtype=torch.int64)
    xs = (idx % w).to(dtype=torch.int64)
    for i in range(b):
        yi = int(ys[i].item())
        xi = int(xs[i].item())
        if xi < half or xi >= w - half or yi < half or yi >= h - half:
            continue
        x_vals = heatmap[i, yi, xi - half : xi + half + 1]
        y_vals = heatmap[i, yi - half : yi + half + 1, xi]
        dx = _fit_offset_1d(x_vals, mode=fit_mode, eps=eps)
        dy = _fit_offset_1d(y_vals, mode=fit_mode, eps=eps)
        offsets[i, 0] = dx
        offsets[i, 1] = dy
    return offsets


@dataclass
class _Runner:
    module: "WASBLightningModule"
    frames_in: int
    resize_hw: tuple[int, int] | None
    heatmap_hw: tuple[int, int] | None
    buffer: list[Tensor]

    def reset(self) -> None:
        self.buffer = []

    def _frame_to_tensor(self, frame_rgb: np.ndarray) -> Tensor:
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB uint8 frame [H,W,3], got {frame_rgb.dtype} {frame_rgb.shape}")

        if self.resize_hw is not None:
            h, w = self.resize_hw
            frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        t = torch.from_numpy(frame_rgb).to(dtype=torch.float32)
        t = t.permute(2, 0, 1).contiguous() / 255.0
        return t

    def _build_windows(self, new_frames: list[Tensor]) -> Tensor:
        all_frames = self.buffer + new_frames
        all_t = torch.stack(all_frames, dim=0)  # [L, C, H, W]
        windows: list[Tensor] = []
        prev_len = len(self.buffer)
        for i in range(len(new_frames)):
            pos = prev_len + i
            start = pos - self.frames_in + 1
            if start < 0:
                idxs = [0] * (-start) + list(range(0, pos + 1))
            else:
                idxs = list(range(start, pos + 1))
            window = all_t[idxs]  # [T, C, H, W]
            windows.append(window)
        window_batch = torch.stack(windows, dim=0)  # [B, T, C, H, W]
        return window_batch

    def _update_buffer(self, new_frames: list[Tensor]) -> None:
        all_frames = self.buffer + new_frames
        keep = max(self.frames_in - 1, 0)
        self.buffer = all_frames[-keep:] if keep > 0 else []

    @_torch_no_grad
    def predict_batch_heatmaps_tta(
        self,
        frames_rgb: np.ndarray,
        *,
        device: torch.device,
        tta_transforms: list[dict[str, Any]],
        target_hw: tuple[int, int] | None,
    ) -> list[Tensor]:
        """Predict per-frame heatmap logits for each TTA variant."""
        if len(frames_rgb) == 0:
            return [torch.zeros((0, 1, 1), dtype=torch.float32)]

        new_frames = [self._frame_to_tensor(f) for f in frames_rgb]
        window_batch = self._build_windows(new_frames).to(device=device)
        b, t, c, h, w = window_batch.shape
        outputs: list[Tensor] = []
        for spec in tta_transforms:
            frames = window_batch.reshape(b * t, c, h, w)
            frames = _apply_tta_transform(frames, spec)
            frames = frames.view(b, t, c, h, w)
            frames_input = self.module.prepare_frames(frames)
            logits = self.module.extract_heatmaps(self.module.model(frames_input))
            if logits.dim() != 4:
                raise ValueError(f"Expected heatmaps [B,T,H,W], got {tuple(logits.shape)}")
            logits = logits[:, -1]
            if self.heatmap_hw is not None and logits.shape[-2:] != self.heatmap_hw:
                logits = F.interpolate(
                    logits.unsqueeze(1),
                    size=self.heatmap_hw,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            target = (
                (int(logits.shape[-2]), int(logits.shape[-1]))
                if target_hw is None
                else target_hw
            )
            logits = _inverse_warp_heatmap(logits, spec, target_hw=target)
            if logits.shape[-2:] != target:
                logits = F.interpolate(
                    logits.unsqueeze(1),
                    size=target,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            outputs.append(logits.detach())

        self._update_buffer(new_frames)
        return outputs


class HeatmapEnsemblePredictor(BasePredictor):
    """Ensemble predictor over multiple trained WASB Lightning checkpoints."""

    def __init__(
        self,
        runners: list[_Runner],
        *,
        device: torch.device,
        output_heatmap_hw: tuple[int, int] | None,
        tta_transforms: list[dict[str, Any]],
        calibration_t: list[float],
        calibration_b: list[float],
        fusion_cfg: dict[str, Any],
        smoothing_cfg: dict[str, Any],
        decode_cfg: dict[str, Any],
    ) -> None:
        if not runners:
            raise ValueError("runners must be non-empty")
        self.runners = runners
        self.device = device
        self.output_heatmap_hw = output_heatmap_hw
        self.tta_transforms = tta_transforms
        self.calibration_t = calibration_t
        self.calibration_b = calibration_b
        self.fusion_cfg = fusion_cfg
        self.smoothing_cfg = smoothing_cfg
        self.decode_cfg = decode_cfg
        self._expected_next_frame_index = 0
        self._forward_state: Tensor | None = None

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_paths: list[str | Path] | tuple[str | Path, ...],
        *,
        device: str | torch.device = "cpu",
        ensemble_cfg: Any | None = None,
        output_heatmap_hw: tuple[int, int] | None = None,
        tta_cfg: Any | None = None,
        calibration_cfg: Any | None = None,
        fusion_cfg: Any | None = None,
        smoothing_cfg: Any | None = None,
        decode_cfg: Any | None = None,
    ) -> Self:
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            torch_device = torch.device("cpu")

        ckpts = [Path(p) for p in checkpoint_paths]
        if not ckpts:
            raise ValueError("checkpoint_paths must be non-empty")
        for p in ckpts:
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")

        if ensemble_cfg is not None:
            if output_heatmap_hw is None:
                output_heatmap_hw = _as_tuple_hw(_cfg_get(ensemble_cfg, "output_heatmap_hw", None))
            tta_cfg = tta_cfg or _cfg_get(ensemble_cfg, "tta", None)
            calibration_cfg = calibration_cfg or _cfg_get(ensemble_cfg, "calibration", None)
            fusion_cfg = fusion_cfg or _cfg_get(ensemble_cfg, "fusion", None)
            smoothing_cfg = smoothing_cfg or _cfg_get(ensemble_cfg, "smoothing", None)
            decode_cfg = decode_cfg or _cfg_get(ensemble_cfg, "decode", None)

        tta_transforms = _parse_tta_transforms(tta_cfg)
        fusion_cfg = dict(fusion_cfg or {})
        smoothing_cfg = dict(smoothing_cfg or {})
        decode_cfg = dict(decode_cfg or {})
        calibration_cfg = dict(calibration_cfg or {})

        runners: list[_Runner] = []
        for ckpt in ckpts:
            from src.wasb.training import WASBLightningModule

            module = WASBLightningModule.load_from_checkpoint(
                str(ckpt), map_location=torch_device
            )
            module.eval()

            cfg = getattr(module, "config", {}) or {}
            model_cfg = _cfg_get(cfg, "model", {})
            data_cfg = _cfg_get(cfg, "data", {})

            frames_in = int(_cfg_get(model_cfg, "frames_in", _cfg_get(data_cfg, "frames_in", 1)))

            resize_hw = None
            inp_h = _cfg_get(model_cfg, "inp_height", None)
            inp_w = _cfg_get(model_cfg, "inp_width", None)
            if inp_h is not None and inp_w is not None:
                resize_hw = (int(inp_h), int(inp_w))
            else:
                rhw = _cfg_get(data_cfg, "resize_hw", None)
                if rhw is not None:
                    resize_hw = (int(rhw[0]), int(rhw[1]))

            heatmap_hw = None
            hhw = _cfg_get(data_cfg, "heatmap_hw", None)
            if hhw is not None:
                heatmap_hw = (int(hhw[0]), int(hhw[1]))

            runners.append(
                _Runner(
                    module=module,
                    frames_in=max(frames_in, 1),
                    resize_hw=resize_hw,
                    heatmap_hw=heatmap_hw,
                    buffer=[],
                )
            )

        calib_t = _expand_per_model(calibration_cfg.get("temperature", 1.0), len(runners), name="temperature")
        calib_b = _expand_per_model(calibration_cfg.get("bias", 0.0), len(runners), name="bias")

        return cls(
            runners=runners,
            device=torch_device,
            output_heatmap_hw=output_heatmap_hw,
            tta_transforms=tta_transforms,
            calibration_t=calib_t,
            calibration_b=calib_b,
            fusion_cfg=fusion_cfg,
            smoothing_cfg=smoothing_cfg,
            decode_cfg=decode_cfg,
        )

    def reset_tracker(self) -> None:
        for r in self.runners:
            r.reset()
        self._expected_next_frame_index = 0
        self._forward_state = None

    def _infer_target_hw(self, candidates: list[Tensor]) -> tuple[int, int]:
        if self.output_heatmap_hw is not None:
            return self.output_heatmap_hw
        if not candidates:
            raise ValueError("candidates must be non-empty")
        h, w = candidates[0].shape[-2:]
        return (int(h), int(w))

    def _model_weights(self, per_model_probs: list[Tensor], eps: float) -> Tensor:
        mode = str(_cfg_get(self.fusion_cfg, "weight_mode", "fixed"))
        weights = _cfg_get(self.fusion_cfg, "model_weights", [1.0] * len(per_model_probs))
        device = per_model_probs[0].device
        dtype = per_model_probs[0].dtype
        fixed = torch.tensor(
            _expand_per_model(weights, len(per_model_probs), name="model_weights"),
            device=device,
            dtype=dtype,
        )
        if mode == "fixed":
            return fixed
        if mode == "entropy":
            ent = torch.stack([_entropy_score(p, eps) for p in per_model_probs], dim=1)
            conf = 1.0 - ent
            return conf / conf.sum(dim=1, keepdim=True).clamp_min(eps)
        if mode == "peak":
            peak = torch.stack([_peak_score(p) for p in per_model_probs], dim=1)
            return peak / peak.sum(dim=1, keepdim=True).clamp_min(eps)
        raise ValueError(f"Unknown weight_mode: {mode}")

    def _decode_coords(
        self,
        heatmap: Tensor,
        *,
        mode: str,
        fit_mode: str,
        fit_window: int,
        eps: float,
    ) -> Tensor:
        b, h, w = heatmap.shape
        if mode == "expected":
            ys = torch.linspace(0, h - 1, h, device=heatmap.device, dtype=heatmap.dtype)
            xs = torch.linspace(0, w - 1, w, device=heatmap.device, dtype=heatmap.dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            exp_x = (heatmap * xx).sum(dim=(-2, -1))
            exp_y = (heatmap * yy).sum(dim=(-2, -1))
            return torch.stack([exp_x, exp_y], dim=-1)
        if mode not in ("map", "fit"):
            raise ValueError(f"Unknown decode mode: {mode}")
        flat = heatmap.view(b, -1)
        idx = torch.argmax(flat, dim=-1)
        ys = (idx // w).to(dtype=heatmap.dtype)
        xs = (idx % w).to(dtype=heatmap.dtype)
        coords = torch.stack([xs, ys], dim=-1)
        if mode == "fit":
            offsets = _fit_peak(heatmap, fit_mode=fit_mode, fit_window=fit_window, eps=eps)
            coords = coords + offsets
        return coords

    @_torch_no_grad
    def predict(
        self,
        frames: np.ndarray,
        *,
        frame_indices: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Predict ball position for a batch of consecutive frames."""
        if frames.size == 0:
            return {
                "ball_uv": np.zeros((0, 2), dtype=np.float32),
                "ball_xy_px": np.zeros((0, 2), dtype=np.float32),
                "visibility": np.zeros((0,), dtype=bool),
                "score": np.zeros((0,), dtype=np.float32),
                "frame_indices": np.zeros((0,), dtype=np.int64),
                "heatmap": np.zeros((0, 1, 1), dtype=np.float32),
            }

        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected frames [B,H,W,3], got {frames.shape}")

        b, h0, w0, _ = frames.shape
        if frame_indices is None:
            start = self._expected_next_frame_index
            frame_indices = list(range(start, start + b))

        if len(frame_indices) != b:
            raise ValueError("frame_indices length must match number of frames")

        if frame_indices and frame_indices[0] != self._expected_next_frame_index:
            for r in self.runners:
                r.reset()
            self._forward_state = None
        self._expected_next_frame_index = frame_indices[-1] + 1

        eps = float(_cfg_get(self.fusion_cfg, "eps", 1e-6))

        per_model_probs: list[Tensor] = []
        target_hw: tuple[int, int] | None = self.output_heatmap_hw
        for mi, runner in enumerate(self.runners):
            logits_list = runner.predict_batch_heatmaps_tta(
                frames,
                device=self.device,
                tta_transforms=self.tta_transforms,
                target_hw=target_hw,
            )
            if target_hw is None:
                target_hw = self._infer_target_hw(logits_list)
            aligned: list[Tensor] = []
            for logits in logits_list:
                if logits.shape[-2:] != target_hw:
                    logits = F.interpolate(
                        logits.unsqueeze(1),
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                aligned.append(logits)
            t = self.calibration_t[mi]
            b0 = self.calibration_b[mi]
            tta_probs = []
            for logits in aligned:
                scaled = logits / t + b0
                prob = torch.sigmoid(scaled)
                prob = _normalize_heatmap(prob + eps, eps)
                tta_probs.append(prob)
            per_model_probs.append(torch.stack(tta_probs, dim=0).mean(dim=0))

        if target_hw is None:
            raise RuntimeError("Failed to infer target_hw for heatmap decoding")
        weights = self._model_weights(per_model_probs, eps)
        if weights.dim() == 1:
            weights = weights.view(1, -1).expand(b, -1)

        logp: Tensor | None = None
        for mi, prob in enumerate(per_model_probs):
            w_m = weights[:, mi].view(b, 1, 1)
            term = w_m * (prob + eps).log()
            logp = term if logp is None else (logp + term)

        assert logp is not None
        emissions = _softmax_spatial(logp)

        smoothing_enabled = bool(_cfg_get(self.smoothing_cfg, "enabled", True))
        if smoothing_enabled:
            kernel_cfg = _cfg_get(self.smoothing_cfg, "kernel", {})
            kernel = _build_kernel(kernel_cfg, device=emissions.device, dtype=emissions.dtype)
            accel_weight = float(_cfg_get(self.smoothing_cfg, "accel_weight", 0.0))
            warmup_tau = float(_cfg_get(self.smoothing_cfg, "warmup_tau", 0.0))
            smoothed, new_state = _forward_backward_smooth(
                emissions,
                kernel=kernel,
                accel_weight=accel_weight,
                warmup_tau=warmup_tau,
                forward_state=self._forward_state,
                eps=eps,
            )
            self._forward_state = new_state
        else:
            smoothed = emissions

        decode_mode = str(_cfg_get(self.decode_cfg, "mode", "fit"))
        fit_mode = str(_cfg_get(self.decode_cfg, "fit_mode", "quadratic"))
        fit_window = int(_cfg_get(self.decode_cfg, "fit_window", 3))
        if fit_window % 2 == 0:
            raise ValueError("fit_window must be odd")
        coords = self._decode_coords(
            smoothed, mode=decode_mode, fit_mode=fit_mode, fit_window=fit_window, eps=eps
        )

        denom_w = float(max(target_hw[1] - 1, 1))
        denom_h = float(max(target_hw[0] - 1, 1))
        uv = torch.stack((coords[:, 0] / denom_w, coords[:, 1] / denom_h), dim=-1).clamp(0.0, 1.0)
        ball_xy_px = torch.stack(
            (
                uv[:, 0] * float(max(w0 - 1, 1)),
                uv[:, 1] * float(max(h0 - 1, 1)),
            ),
            dim=-1,
        )

        confidence_mode = str(_cfg_get(self.decode_cfg, "confidence_mode", "peak"))
        if confidence_mode == "entropy":
            confidence = 1.0 - _entropy_score(smoothed, eps)
        else:
            confidence = _peak_score(smoothed)
        visibility_threshold = float(_cfg_get(self.decode_cfg, "visibility_threshold", 0.0))
        visibility = confidence >= visibility_threshold

        return_heatmap = bool(_cfg_get(self.decode_cfg, "return_heatmap", True))
        return {
            "ball_uv": uv.detach().cpu().numpy().astype(np.float32, copy=False),
            "ball_xy_px": ball_xy_px.detach().cpu().numpy().astype(np.float32, copy=False),
            "visibility": visibility.detach().cpu().numpy().astype(bool, copy=False),
            "score": confidence.detach().cpu().numpy().astype(np.float32, copy=False),
            "frame_indices": np.asarray(frame_indices, dtype=np.int64),
            "heatmap": smoothed.detach().cpu().numpy().astype(np.float32, copy=False)
            if return_heatmap
            else np.zeros((0, 1, 1), dtype=np.float32),
        }
