from __future__ import annotations

from typing import Any

import pytest
from omegaconf import OmegaConf

from src.wasb.models import TemporalConvGRUModel, build_model


def _build_cfg(model_cfg: dict[str, Any]):
    """Helper to wrap a model config in the expected hydra-style structure."""
    return OmegaConf.create({"model": model_cfg})


def test_build_hrnet_model_minimal():
    model_cfg = {
        "name": "hrnet",
        "frames_in": 1,
        "frames_out": 1,
        "out_scales": [0],
        "MODEL": {
            "EXTRA": {
                "STEM": {"STRIDES": [2, 2], "INPLANES": 64},
                "FINAL_CONV_KERNEL": 1,
                "DECONV": {"NUM_DECONVS": 1, "KERNEL_SIZE": [4]},
                "PRETRAINED_LAYERS": ["*"],
                "STAGE1": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 1,
                    "BLOCK": "BOTTLENECK",
                    "NUM_BLOCKS": [4],
                    "NUM_CHANNELS": [64],
                    "FUSE_METHOD": "SUM",
                },
                "STAGE2": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 2,
                    "BLOCK": "BASIC",
                    "NUM_BLOCKS": [4, 4],
                    "NUM_CHANNELS": [48, 96],
                    "FUSE_METHOD": "SUM",
                },
                "STAGE3": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 3,
                    "BLOCK": "BASIC",
                    "NUM_BLOCKS": [4, 4, 4],
                    "NUM_CHANNELS": [48, 96, 192],
                    "FUSE_METHOD": "SUM",
                },
                "STAGE4": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 4,
                    "BLOCK": "BASIC",
                    "NUM_BLOCKS": [4, 4, 4, 4],
                    "NUM_CHANNELS": [48, 96, 192, 384],
                    "FUSE_METHOD": "SUM",
                },
            }
        },
    }
    cfg = _build_cfg(model_cfg)

    model, (prepare_frames, extract_heatmaps) = build_model(cfg)

    assert callable(prepare_frames)
    assert callable(extract_heatmaps)

    import torch

    x = torch.randn(2, 3, 288, 512)
    y = model(x)
    heatmaps = extract_heatmaps(y)
    assert heatmaps.shape[0] == 2


def test_build_hrcnet_model_minimal():
    model_cfg = {
        "name": "hrcnet",
        "frames_in": 1,
        "frames_out": 1,
        "high_channels": 32,
        "low_channels": 64,
        "num_stages": 2,
        "high_block": "BASIC",
        "low_block": "BASIC",
        "num_high_blocks": 1,
        "num_low_blocks": 1,
        "upsample_mode": "nearest",
    }
    cfg = _build_cfg(model_cfg)

    model, (prepare_frames, extract_heatmaps) = build_model(cfg)

    assert callable(prepare_frames)
    assert callable(extract_heatmaps)

    import torch

    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    heatmaps = extract_heatmaps(y)
    assert heatmaps.shape[0] == 2


def test_build_hrnet_gru_with_hrnet_backbone():
    model_cfg = {
        "name": "hrnet_gru",
        "frames_in": 2,
        "frames_out": 2,
        "gru_hidden_channels": [32],
        "gru_kernel_size": 3,
        "stack_channels": False,
        "backbone": {
            "name": "hrnet",
            "frames_in": 1,
            "frames_out": 1,
            "out_scales": [0],
            "MODEL": {
                "EXTRA": {
                    "STEM": {"STRIDES": [2, 2], "INPLANES": 64},
                    "FINAL_CONV_KERNEL": 1,
                    "DECONV": {"NUM_DECONVS": 1, "KERNEL_SIZE": [4]},
                    "PRETRAINED_LAYERS": ["*"],
                    "STAGE1": {
                        "NUM_MODULES": 1,
                        "NUM_BRANCHES": 1,
                        "BLOCK": "BOTTLENECK",
                        "NUM_BLOCKS": [4],
                        "NUM_CHANNELS": [64],
                        "FUSE_METHOD": "SUM",
                    },
                    "STAGE2": {
                        "NUM_MODULES": 1,
                        "NUM_BRANCHES": 2,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [4, 4],
                        "NUM_CHANNELS": [48, 96],
                        "FUSE_METHOD": "SUM",
                    },
                    "STAGE3": {
                        "NUM_MODULES": 1,
                        "NUM_BRANCHES": 3,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [4, 4, 4],
                        "NUM_CHANNELS": [48, 96, 192],
                        "FUSE_METHOD": "SUM",
                    },
                    "STAGE4": {
                        "NUM_MODULES": 1,
                        "NUM_BRANCHES": 4,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [4, 4, 4, 4],
                        "NUM_CHANNELS": [48, 96, 192, 384],
                        "FUSE_METHOD": "SUM",
                    },
                }
            },
        },
    }
    cfg = _build_cfg(model_cfg)

    model, (prepare_frames, extract_heatmaps) = build_model(cfg)

    assert isinstance(model, TemporalConvGRUModel)

    import torch

    x = torch.randn(2, 2, 3, 288, 512)
    frames = prepare_frames(x)
    assert frames.shape == x.shape
    y = model(frames)
    heatmaps = extract_heatmaps(y)
    assert heatmaps.shape[:2] == (2, 2)


def test_build_model_invalid_name_raises():
    cfg = _build_cfg({"name": "unknown_model"})

    with pytest.raises(KeyError):
        build_model(cfg)
