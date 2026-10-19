"""Strict model-runtime and user-acquired SMPL asset configuration contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.submodules.configuration import (
    SmplAssetPaths,
    SubmoduleRuntimeConfig,
    ViTPoseHeadConfig,
)
from src.submodules.smpl_assets import SMPL_ASSET_SOURCES
from src.utils.configuration import ConfigurationError, PathResolver, RuntimePathRoots


def _runtime_mapping() -> dict[str, object]:
    return {
        "device": "cpu",
        "tracking": {"bbox_enlarge": 1.2},
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


def test_runtime_rejects_removed_device_fallback_toggle() -> None:
    mapping = _runtime_mapping()
    mapping["allow_device_fallback"] = False

    with pytest.raises(ConfigurationError, match="allow_device_fallback"):
        SubmoduleRuntimeConfig.from_mapping(mapping)


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


@pytest.mark.parametrize("missing_name", SMPL_ASSET_SOURCES)
def test_smpl_assets_require_explicit_checkpoint_files(tmp_path: Path, missing_name: str) -> None:
    project = tmp_path.resolve()
    roots = RuntimePathRoots.from_mapping(
        {
            "project_root": ".",
            "data_root": "data",
            "checkpoint_root": "published-models",
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
    assets = SmplAssetPaths.from_mapping(
        asset_names,
        resolver=PathResolver(roots),
    )
    for name, relative in asset_names.items():
        path = project / "published-models" / relative
        assert getattr(assets, name) == path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assets.require_files()
    missing = getattr(assets, missing_name)
    missing.unlink()
    with pytest.raises(FileNotFoundError) as error:
        assets.require_files()
    assert str(missing) in str(error.value)
    assert SMPL_ASSET_SOURCES[missing_name].source_url in str(error.value)
    assert "src/submodules/README.md" in str(error.value)


@pytest.mark.parametrize("failure", ["absolute", "unknown", "escape"])
def test_smpl_assets_reject_invalid_paths(tmp_path: Path, failure: str) -> None:
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
        "hmr2_mean_params": "assets/mean.npz",
        "smplx_to_smpl": "assets/to-smpl.pt",
        "smpl_coco17_regressor": "assets/coco.pt",
        "smplx_verts437": "assets/verts.pt",
        "smpl_neutral_joint_regressor": "assets/joints.pt",
    }
    if failure == "absolute":
        mapping["hmr2_mean_params"] = str(tmp_path / "mean.npz")
    elif failure == "unknown":
        mapping["typo"] = "assets/typo.pt"
    else:
        mapping["hmr2_mean_params"] = "../mean.npz"
    with pytest.raises(ConfigurationError):
        SmplAssetPaths.from_mapping(mapping, resolver=PathResolver(roots))
