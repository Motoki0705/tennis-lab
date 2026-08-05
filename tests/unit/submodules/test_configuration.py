"""Strict model-runtime and bundled-asset configuration contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.submodules.configuration import (
    BundledModelAssetPaths,
    SubmoduleRuntimeConfig,
    ViTPoseHeadConfig,
)
from src.utils.configuration import ConfigurationError, PathResolver, RuntimePathRoots


def _runtime_mapping() -> dict[str, object]:
    return {
        "device": "cpu",
        "allow_device_fallback": False,
        "tracking": {"yolo_confidence": 0.25, "bbox_enlarge": 1.2},
        "dino_detector": {
            "confidence": 0.35,
            "short_side": 800,
            "max_long_side": 1333,
        },
        "vitpose": {
            "flip_test": True,
            "batch_size": 8,
            "head": {
                "in_channels": 1280,
                "out_channels": 17,
                "num_deconv_layers": 2,
                "num_deconv_filters": [256, 256],
                "num_deconv_kernels": [4, 4],
                "final_conv_kernel": 1,
                "num_conv_layers": 0,
                "num_conv_kernels": [],
            },
        },
        "hmr2": {"batch_size": 8},
        "static_cam": True,
    }


def test_vitpose_released_head_is_complete_and_typed() -> None:
    runtime = SubmoduleRuntimeConfig.from_mapping(_runtime_mapping())

    assert runtime.vitpose.head == ViTPoseHeadConfig(
        in_channels=1280,
        out_channels=17,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        final_conv_kernel=1,
        num_conv_layers=0,
        num_conv_kernels=(),
    )


@pytest.mark.parametrize("failure", ["missing", "unknown", "wrong_type"])
def test_vitpose_head_rejects_incomplete_or_untyped_values(failure: str) -> None:
    mapping = _runtime_mapping()
    vitpose = mapping["vitpose"]
    assert isinstance(vitpose, dict)
    head = vitpose["head"]
    assert isinstance(head, dict)
    if failure == "missing":
        del head["num_conv_kernels"]
    elif failure == "unknown":
        head["typo"] = True
    else:
        head["final_conv_kernel"] = True

    with pytest.raises(ConfigurationError):
        SubmoduleRuntimeConfig.from_mapping(mapping)


def test_bundled_assets_require_explicit_project_files(tmp_path: Path) -> None:
    project = tmp_path.resolve()
    roots = RuntimePathRoots.from_mapping(
        {
            "project_root": ".",
            "data_root": "data",
            "checkpoint_root": "ckpt",
            "artifact_root": "artifacts",
            "output_root": "outputs",
            "cache_root": ".cache",
            "external_asset_root": "third_party",
        },
        repository_root=project,
    )
    asset_names = {
        "hmr2_mean_params": "assets/mean.npz",
        "smplx_to_smpl": "assets/to-smpl.pt",
        "smpl_coco17_regressor": "assets/coco.pt",
        "smplx_verts437": "assets/verts.pt",
        "smpl_neutral_joint_regressor": "assets/joints.pt",
    }
    assets = BundledModelAssetPaths.from_mapping(
        asset_names,
        resolver=PathResolver(roots),
    )
    with pytest.raises(FileNotFoundError, match="Bundled GVHMR asset"):
        assets.require_files()
    for relative in asset_names.values():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assets.require_files()


def test_bundled_assets_reject_absolute_and_unknown_paths(tmp_path: Path) -> None:
    roots = RuntimePathRoots.from_mapping(
        {
            "project_root": ".",
            "data_root": "data",
            "checkpoint_root": "ckpt",
            "artifact_root": "artifacts",
            "output_root": "outputs",
            "cache_root": ".cache",
            "external_asset_root": "third_party",
        },
        repository_root=tmp_path.resolve(),
    )
    mapping: dict[str, object] = {
        "hmr2_mean_params": "/tmp/mean.npz",
        "smplx_to_smpl": "assets/to-smpl.pt",
        "smpl_coco17_regressor": "assets/coco.pt",
        "smplx_verts437": "assets/verts.pt",
        "smpl_neutral_joint_regressor": "assets/joints.pt",
        "typo": "assets/typo.pt",
    }
    with pytest.raises(ConfigurationError):
        BundledModelAssetPaths.from_mapping(mapping, resolver=PathResolver(roots))
