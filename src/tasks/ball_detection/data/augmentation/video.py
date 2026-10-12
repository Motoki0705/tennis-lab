"""RGB-space video augmentation with synchronized endpoint-normalized GT.

Random Erasing/Cutout and DynaAugment/CVRL inspire these task-specific
perturbations. They are not physically complete camera/occlusion simulators.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class AugmentationConfig:
    camera_probability: float
    occlusion_probability: float
    targeted_probability: float
    pan_speed: float
    roll_speed_degrees: float
    log_zoom_speed: float
    max_translation: float
    max_roll_degrees: float
    max_log_zoom: float
    jitter_amplitude: float
    occlusion_min_frames: int
    occlusion_max_frames: int
    occlusion_min_size: float
    occlusion_max_size: float
    occluder_speed: float
    max_removed_fraction: float

    @classmethod
    def load(cls, path: Path) -> AugmentationConfig:
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("Augmentation config must declare all fields exactly")
        return cls(**raw)

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid augmentation parameter: {key}")
        if any(not 0 <= v <= 1 for v in (self.camera_probability, self.occlusion_probability,
                                         self.targeted_probability, self.max_removed_fraction)):
            raise ValueError("Augmentation probabilities must be in [0,1]")
        if not 1 <= self.occlusion_min_frames <= self.occlusion_max_frames <= 8:
            raise ValueError("Occlusion must leave temporal context in a 32-frame clip")
        if not 0 < self.occlusion_min_size <= self.occlusion_max_size <= .25:
            raise ValueError("Occlusion size must be in (0,.25]")

    def profile(self, name: str) -> AugmentationConfig:
        if name == "train":
            return self
        if name not in {"clean", "camera", "occlusion", "combined"}:
            raise ValueError("Unknown augmentation evaluation profile")
        return replace(self, camera_probability=float(name in {"camera", "combined"}),
                       occlusion_probability=float(name in {"occlusion", "combined"}))


def sample_seed(seed: int, epoch: int, clip: str, start: int, step: int, profile: str) -> int:
    identity = f"{seed}:{epoch}:{clip}:{start}:{step}:{profile}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")


def transform_uv(uv: Tensor, matrix: Tensor) -> Tensor:
    return torch.einsum("btij,btj->bti", matrix[..., :2, :], torch.cat((uv, torch.ones_like(uv[..., :1])), -1))


def warp_rgb(rgb: Tensor, matrix: Tensor) -> Tensor:
    """Forward normalized-UV matrices; inverse sampling, endpoints align exactly."""
    b, t, c, h, w = rgb.shape
    y, x = torch.meshgrid(torch.linspace(0, 1, h, device=rgb.device),
                          torch.linspace(0, 1, w, device=rgb.device), indexing="ij")
    grid = torch.stack((x, y, torch.ones_like(x)), -1)
    inverse = torch.linalg.inv(matrix.float())
    source = torch.einsum("btij,hwj->bthwi", inverse, grid)[..., :2]
    image = F.grid_sample(rgb.flatten(0, 1).float(), (source * 2 - 1).flatten(0, 1),
                          mode="bilinear", padding_mode="border", align_corners=True)
    return image.round().clamp(0, 255).to(torch.uint8).reshape(b, t, c, h, w)


class VideoAugmenter:
    def __init__(self, config: AugmentationConfig, seed: int) -> None:
        self.config, self.seed = config, seed

    def __call__(self, rgb: Tensor, batch: dict[str, Any], *, epoch: int,
                 profile: str = "train") -> tuple[Tensor, dict[str, Any], dict[str, Tensor]]:
        cfg = self.config.profile(profile)
        b, t, _, h, w = rgb.shape
        if t != 32 or rgb.dtype != torch.uint8 or min(h, w) < 2:
            raise ValueError("Video augmentation expects uint8 B,32,3,H,W")
        if not cfg.camera_probability and not cfg.occlusion_probability:
            return rgb, batch, {}
        # Only small metadata/parameters are sampled on CPU; image processing stays on device.
        rngs = [random.Random(sample_seed(self.seed, epoch, str(clip), int(batch["start"][i]),
                    int(batch["frame_step"][i]), profile)) for i, clip in enumerate(batch["clip_id"])]
        camera = [r.random() < cfg.camera_probability for r in rngs]
        occlusion = [r.random() < cfg.occlusion_probability for r in rngs]
        uv = batch["uv"].to(rgb.device, dtype=torch.float32)
        valid = batch["position_valid"].to(rgb.device)
        times = batch["timestamps"].to(rgb.device, dtype=torch.float32)
        tau = times - (times[:, :1] + times[:, -1:]) / 2
        matrices = torch.eye(3, device=rgb.device).expand(b, t, 3, 3).clone()
        for i, r in enumerate(rngs):
            if not camera[i]:
                continue
            dx = (r.uniform(-cfg.pan_speed, cfg.pan_speed) * tau[i]).clamp(-cfg.max_translation, cfg.max_translation)
            dy = (r.uniform(-cfg.pan_speed, cfg.pan_speed) * tau[i]).clamp(-cfg.max_translation, cfg.max_translation)
            phase = r.uniform(-math.pi, math.pi)
            dx = dx + cfg.jitter_amplitude * torch.sin(2 * math.pi * r.uniform(.5, 2.) * tau[i] + phase)
            dy = dy + cfg.jitter_amplitude * torch.sin(2 * math.pi * r.uniform(.5, 2.) * tau[i] - phase)
            angle = (r.uniform(-cfg.roll_speed_degrees, cfg.roll_speed_degrees) * tau[i]).clamp(
                -cfg.max_roll_degrees, cfg.max_roll_degrees) * (math.pi / 180)
            scale = (r.uniform(-cfg.log_zoom_speed, cfg.log_zoom_speed) * tau[i]).clamp(-cfg.max_log_zoom, cfg.max_log_zoom).exp()
            a, sine = scale * angle.cos(), scale * angle.sin()
            cxy, cyx = -sine * (h - 1) / (w - 1), sine * (w - 1) / (h - 1)
            matrices[i, :, 0, 0], matrices[i, :, 0, 1] = a, cxy
            matrices[i, :, 1, 0], matrices[i, :, 1, 1] = cyx, a
            matrices[i, :, 0, 2] = .5 + dx - .5 * (a + cxy)
            matrices[i, :, 1, 2] = .5 + dy - .5 * (cyx + a)
        transformed = transform_uv(uv, matrices)
        kept = valid & ((transformed >= 0) & (transformed <= 1)).all(-1)
        # Explicitly reject excessive crop loss rather than training a near-empty target.
        removed_fraction = (valid & ~kept).sum(1) / valid.sum(1).clamp_min(1)
        rejected = (removed_fraction > cfg.max_removed_fraction) | (kept.sum(1) < valid.sum(1).clamp_max(8))
        matrices = torch.where(rejected[:, None, None, None], torch.eye(3, device=rgb.device), matrices)
        transformed = transform_uv(uv, matrices)
        kept = valid & ((transformed >= 0) & (transformed <= 1)).all(-1)
        output = warp_rgb(rgb, matrices) if any(camera) else rgb
        covered = torch.zeros_like(kept)
        if any(occlusion):
            yy = torch.linspace(0, 1, h, device=rgb.device)[None, :, None]
            xx = torch.linspace(0, 1, w, device=rgb.device)[None, None, :]
            images = []
            for i, r in enumerate(rngs):
                if not occlusion[i]:
                    images.append(output[i])
                    continue
                length = r.randint(cfg.occlusion_min_frames, cfg.occlusion_max_frames)
                observed = [j for j in torch.nonzero(batch["position_valid"][i].cpu(), as_tuple=False).flatten().tolist() if 0 < j < t - 1]
                if not observed:
                    raise ValueError("Occlusion augmentation requires observed temporal context")
                anchor = r.choice(observed)
                start = min(max(anchor - length // 2, 1), t - length - 1)
                width = r.uniform(cfg.occlusion_min_size, cfg.occlusion_max_size)
                height = r.uniform(cfg.occlusion_min_size, cfg.occlusion_max_size)
                if r.random() < cfg.targeted_probability:
                    center = transformed[i, anchor] + transformed.new_tensor((r.uniform(-.3, .3) * width, r.uniform(-.3, .3) * height))
                else:
                    center = transformed.new_tensor((r.random(), r.random()))
                # Independent motion, NOT a mask that follows GT on every frame.
                dt = times[i] - times[i, anchor]
                cx = center[0] + dt * r.uniform(-cfg.occluder_speed, cfg.occluder_speed)
                cy = center[1] + dt * r.uniform(-cfg.occluder_speed, cfg.occluder_speed)
                active = (torch.arange(t, device=rgb.device) >= start) & (torch.arange(t, device=rgb.device) < start + length)
                mask = active[:, None, None] & ((xx - cx[:, None, None]).abs() <= width / 2) & ((yy - cy[:, None, None]).abs() <= height / 2)
                fill = torch.tensor([r.randrange(256) for _ in range(3)], device=rgb.device, dtype=torch.uint8)
                images.append(torch.where(mask[:, None], fill[None, :, None, None], output[i]))
                covered[i] = active & kept[i] & ((transformed[i, :, 0] - cx).abs() <= width / 2) & ((transformed[i, :, 1] - cy).abs() <= height / 2)
            output = torch.stack(images)
        result = dict(batch, uv=transformed, position_valid=kept)
        audit = dict(matrix=matrices, artificially_occluded=covered, camera_rejected=rejected,
                     removed_by_camera=valid & ~kept)
        return output, result, audit
