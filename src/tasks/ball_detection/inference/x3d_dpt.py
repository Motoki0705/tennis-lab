"""Scene inference for explicitly exported X3D/CSN + DPT heatmap artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.tasks.ball_detection.model_io.contracts import (
    DEFAULT_CANDIDATE_CONFIG,
    BallCandidateConfig,
    BallModelIOError,
    BallPrediction,
)
from src.tasks.ball_detection.model_io.mdd_coordinates import validate_rgb_input
from src.tasks.ball_detection.model_io.mdd_heatmap import (
    MDDHeatmapAdapter,
    restore_heatmap_model,
)
from src.tasks.ball_detection.models.mdd import MDDDPTDetector, MDDHeatmapConfig
from src.tasks.ball_detection.training.runtime.compilation import (
    compile_coordinate_model,
    coordinate_compile_scope,
)
from src.utils.device import resolve_device
from src.utils.inference_recording import span


def scene_dpt_input_contract(*, image_size: tuple[int, int], frames: int, input_channels: int) -> dict[str, Any]:
    """Spatial metadata supplements the native artifact's RGB/MDD contract.

    Historical heatmap artifacts did not save a spatial size. Exporters must
    obtain it from the pinned training evaluation manifest, never a filename.
    """
    if len(image_size) != 2 or any(type(v) is not int or v < 8 for v in image_size):
        raise BallModelIOError("Scene DPT image_size requires positive H,W >= 8")
    return dict(schema="scene_x3d_csn_dpt_input.v1", image_size_hw=list(image_size),
                frames=frames, input_channels=input_channels, resize="opencv_inter_linear_uint8",
                rgb_normalization="divide_255_only", output_stride=4)


class X3DDPTPredictor:
    """Preserve native uint8 preprocessing and return the existing evidence type.

    Window batches use the same B,T,C,H,W tensor layout as ConvNeXt. The dtype
    is uint8; RGB+MDD is made by the model's canonical I/O transform per window.
    Precision is explicit and never changed in response to allocation failures.
    """

    def __init__(self, model: MDDDPTDetector, *, device: torch.device,
                 image_size: tuple[int, int], precision: str, subpixel_refine: bool, compile_mode: str) -> None:
        if type(model) is not MDDDPTDetector or type(model.config) is not MDDHeatmapConfig:
            raise BallModelIOError("X3D/CSN + DPT requires the heatmap-only DPT model")
        if (model.config.encoder_variant, model.config.temporal_mixing, model.config.input_channels) != ("x3d_csn", "depthwise3d", 5):
            raise BallModelIOError("Expected the five-channel X3D/CSN encoder")
        if precision not in {"fp32", "bf16"}:
            raise BallModelIOError("DPT precision must be fp32 or bf16")
        if precision == "bf16" and device.type != "cuda":
            raise BallModelIOError("Scene bf16 inference requires CUDA")
        self.model = model.to(device).eval()
        self.device, self.image_size, self.precision = device, image_size, precision
        self.subpixel_refine = subpixel_refine
        compile_coordinate_model(self.model, mode=compile_mode)

    @classmethod
    def load_from_checkpoint(cls, path: Path, *, device: str, image_size: tuple[int, int],
                             frames: int, input_channels: int, normalize_imagenet: bool,
                             precision: str, subpixel_refine: bool, compile_mode: str) -> X3DDPTPredictor:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        expected = scene_dpt_input_contract(image_size=image_size, frames=frames, input_channels=input_channels)
        if not isinstance(saved, dict) or saved.get("scene_input_contract") != expected:
            raise BallModelIOError("Checkpoint scene input contract does not match configured channels, T, size or normalization")
        if normalize_imagenet:
            raise BallModelIOError("X3D/CSN + DPT forbids ImageNet normalization")
        config = saved.get("model_config", {})
        if (config.get("frames"), config.get("input_channels")) != (frames, input_channels):
            raise BallModelIOError("Checkpoint model configuration disagrees with scene input contract")
        model = restore_heatmap_model(saved)  # validates the full RGB/MDD contract and loads strictly
        if type(model) is not MDDDPTDetector:
            raise BallModelIOError("X3D/CSN + DPT does not accept a query or hybrid artifact")
        return cls(model, device=resolve_device(device), image_size=image_size,
                   precision=precision, subpixel_refine=subpixel_refine, compile_mode=compile_mode)

    @property
    def configured_frames(self) -> int:
        return int(self.model.config.frames)

    def predict(self, images: Tensor, *, candidate_config: BallCandidateConfig = DEFAULT_CANDIDATE_CONFIG) -> BallPrediction:
        validate_rgb_input(images, frames=self.configured_frames)
        if tuple(images.shape[-2:]) != self.image_size:
            raise BallModelIOError(f"X3D/CSN + DPT requires image size {self.image_size}")
        with torch.inference_mode(), coordinate_compile_scope(self.model):
            with span("transfer", model="Ball", device=str(self.device)):
                rgb = images.to(self.device, non_blocking=True)
            with (span("inference", model="Ball", device=str(self.device), batch_shape=list(rgb.shape)),
                  torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16")):
                logits = self.model(rgb)
            expected = (len(images), self.configured_frames, (self.image_size[0] + 3) // 4, (self.image_size[1] + 3) // 4)
            if tuple(logits.shape) != expected:
                raise BallModelIOError(f"DPT must return native /4 heatmaps {expected}, got {tuple(logits.shape)}")
            return MDDHeatmapAdapter(self.model.config, subpixel=self.subpixel_refine,
                                     candidate_config=candidate_config).decode_output(logits)
