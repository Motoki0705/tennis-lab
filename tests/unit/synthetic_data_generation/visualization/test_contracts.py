"""Tests for synthetic dataset visualization request contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.synthetic_data_generation.visualization.configuration import (
    build_visualization_request,
)
from src.synthetic_data_generation.visualization.contracts import (
    CourtOverlayConfiguration,
    CourtOverlayMode,
    DatasetVisualizationDomain,
    DatasetVisualizationRequest,
)
from src.utils.configuration import PathContractError
from src.utils.configuration.catalog import BOUNDARY_CONTRACTS


def _dataset_root(tmp_path: Path, domain: str) -> Path:
    root = tmp_path / "scenes" / "scene-0" / "datasets" / domain
    root.mkdir(parents=True)
    return root


def _court_overlay_config(*, mode: str = "semantic") -> dict[str, object]:
    return {
        "mode": mode,
        "color_rgb": [255, 96, 32],
        "background_color_rgb": [0, 0, 0],
        "opacity": 0.55,
        "depth_epsilon_m": 0.02,
        "near_plane_m": 0.05,
        "maximum_cells": 1_000_000,
        "maximum_surface_faces": 4_000_000,
        "maximum_projected_pixels": 100_000_000,
    }


def _court_overlay_contract(
    *,
    mode: CourtOverlayMode = CourtOverlayMode.SEMANTIC,
) -> CourtOverlayConfiguration:
    return CourtOverlayConfiguration(
        mode=mode,
        color_rgb=(255, 96, 32),
        background_color_rgb=(0, 0, 0),
        opacity=0.55,
        depth_epsilon_m=0.02,
        near_plane_m=0.05,
        maximum_cells=1_000_000,
        maximum_surface_faces=4_000_000,
        maximum_projected_pixels=100_000_000,
    )


def test_court_request_requires_only_explicit_trajectory_id(tmp_path: Path) -> None:
    request = DatasetVisualizationRequest(
        domain=DatasetVisualizationDomain.COURT,
        dataset_root=_dataset_root(tmp_path, "court"),
        output_video=tmp_path / "court.mp4",
        trajectory_id="orbit-0",
        logical_scene_id=None,
        camera_id=None,
        fps=30.0,
        crf=17,
        history_frames=12,
    )

    assert request.metadata_path == tmp_path / "court.json"
    assert request.court_overlay.mode is CourtOverlayMode.SEMANTIC

    with pytest.raises(ValueError, match="does not accept"):
        DatasetVisualizationRequest(
            domain=DatasetVisualizationDomain.COURT,
            dataset_root=request.dataset_root,
            output_video=tmp_path / "invalid.mp4",
            trajectory_id="orbit-0",
            logical_scene_id="scene-0",
            camera_id=None,
            fps=30.0,
            crf=17,
            history_frames=12,
            court_overlay=_court_overlay_contract(),
        )


@pytest.mark.parametrize(
    "domain",
    [DatasetVisualizationDomain.BLCS, DatasetVisualizationDomain.PLCS],
)
def test_compact_domains_require_scene_and_camera(
    tmp_path: Path,
    domain: DatasetVisualizationDomain,
) -> None:
    root = _dataset_root(tmp_path, domain.value)

    with pytest.raises(ValueError, match="requires logical_scene_id and camera_id"):
        DatasetVisualizationRequest(
            domain=domain,
            dataset_root=root,
            output_video=tmp_path / f"{domain.value}.mp4",
            trajectory_id=None,
            logical_scene_id="logical-0",
            camera_id=None,
            fps=30.0,
            crf=17,
            history_frames=12,
            court_overlay=_court_overlay_contract(),
        )


def test_request_rejects_output_inside_dataset_or_existing_output(
    tmp_path: Path,
) -> None:
    root = _dataset_root(tmp_path, "court")
    with pytest.raises(ValueError, match="outside the dataset"):
        DatasetVisualizationRequest(
            domain=DatasetVisualizationDomain.COURT,
            dataset_root=root,
            output_video=root / "preview.mp4",
            trajectory_id="orbit-0",
            logical_scene_id=None,
            camera_id=None,
            fps=30.0,
            crf=17,
            history_frames=12,
            court_overlay=_court_overlay_contract(),
        )

    output = tmp_path / "existing.mp4"
    output.touch()
    with pytest.raises(FileExistsError, match="already exists"):
        DatasetVisualizationRequest(
            domain=DatasetVisualizationDomain.COURT,
            dataset_root=root,
            output_video=output,
            trajectory_id="orbit-0",
            logical_scene_id=None,
            camera_id=None,
            fps=30.0,
            crf=17,
            history_frames=12,
            court_overlay=_court_overlay_contract(),
        )


def test_hydra_boundary_builds_one_strict_domain_selection(tmp_path: Path) -> None:
    data_root = tmp_path / "runtime-data"
    output_root = tmp_path / "runtime-output"
    root = _dataset_root(data_root, "plcs")
    config = OmegaConf.create(
        {
            "roots": {
                "project_root": str(tmp_path),
                "data_root": str(data_root),
                "checkpoint_root": "ckpt",
                "artifact_root": "artifacts",
                "output_root": str(output_root),
                "cache_root": ".cache",
                "external_asset_root": "third_party",
            },
            "visualization": {
                "domain": "plcs",
                "dataset_root": "scenes/scene-0/datasets/plcs",
                "output_video": "previews/plcs.mp4",
                "trajectory_id": None,
                "logical_scene_id": "logical-0",
                "camera_id": "camera-0",
                "fps": 24.0,
                "crf": 19,
                "history_frames": 0,
            },
        }
    )

    request = build_visualization_request(config)

    assert request.domain is DatasetVisualizationDomain.PLCS
    assert request.logical_scene_id == "logical-0"
    assert request.camera_id == "camera-0"
    assert request.fps == 24.0
    assert request.dataset_root == root
    assert request.output_video == output_root / "previews/plcs.mp4"
    assert request.court_overlay.mode is CourtOverlayMode.SEMANTIC

    config.visualization["unknown_overlay_alias"] = "semantic"
    with pytest.raises(ValueError, match="unknown=.*unknown_overlay_alias"):
        build_visualization_request(config)


@pytest.mark.parametrize("field", ["dataset_root", "output_video"])
def test_hydra_boundary_rejects_absolute_path_bypass_of_configured_roots(
    tmp_path: Path,
    field: str,
) -> None:
    _dataset_root(tmp_path / "runtime-data", "court")
    visualization: dict[str, object] = {
        "domain": "court",
        "dataset_root": "scenes/scene-0/datasets/court",
        "output_video": "previews/court.mp4",
        "trajectory_id": "orbit-0",
        "logical_scene_id": None,
        "camera_id": None,
        "fps": 24.0,
        "crf": 19,
        "history_frames": 0,
        "court_overlay": _court_overlay_config(),
    }
    visualization[field] = str(tmp_path / f"outside-{field}")
    config = OmegaConf.create(
        {
            "roots": {
                "project_root": str(tmp_path),
                "data_root": "runtime-data",
                "checkpoint_root": "ckpt",
                "artifact_root": "artifacts",
                "output_root": "runtime-output",
                "cache_root": ".cache",
                "external_asset_root": "third_party",
            },
            "visualization": visualization,
        }
    )

    with pytest.raises(PathContractError, match="must be relative"):
        build_visualization_request(config)


def test_hydra_boundary_selects_aabb_mode_and_rejects_it_for_other_domains(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "runtime-data"
    output_root = tmp_path / "runtime-output"
    _dataset_root(data_root, "court")
    roots = {
        "project_root": str(tmp_path),
        "data_root": str(data_root),
        "checkpoint_root": "ckpt",
        "artifact_root": "artifacts",
        "output_root": str(output_root),
        "cache_root": ".cache",
        "external_asset_root": "third_party",
    }
    visualization = {
        "domain": "court",
        "dataset_root": "scenes/scene-0/datasets/court",
        "output_video": "previews/court.mp4",
        "trajectory_id": "orbit-0",
        "logical_scene_id": None,
        "camera_id": None,
        "fps": 24.0,
        "crf": 19,
        "history_frames": 0,
        "court_overlay": _court_overlay_config(mode="trajectory_support_aabb"),
    }

    request = build_visualization_request(
        OmegaConf.create({"roots": roots, "visualization": visualization})
    )

    assert request.court_overlay == _court_overlay_contract(
        mode=CourtOverlayMode.TRAJECTORY_SUPPORT_AABB
    )

    plcs_root = _dataset_root(data_root, "plcs")
    assert plcs_root.is_dir()
    visualization.update(
        {
            "domain": "plcs",
            "dataset_root": "scenes/scene-0/datasets/plcs",
            "trajectory_id": None,
            "logical_scene_id": "logical-0",
            "camera_id": "camera-0",
        }
    )
    with pytest.raises(ValueError, match="accepted only for Court"):
        build_visualization_request(
            OmegaConf.create({"roots": roots, "visualization": visualization})
        )


def test_catalog_exposes_visualization_fields_and_path_roles() -> None:
    boundary = next(
        contract
        for contract in BOUNDARY_CONTRACTS
        if contract.boundary_id
        == "src.synthetic_data_generation.scripts.visualize_dataset:main"
    )

    assert (
        "src.synthetic_data_generation.visualization.contracts."
        "DatasetVisualizationConfiguration" in boundary.authority_symbols
    )
    assert any(
        "DatasetVisualizationConfiguration.dataset_root" in path
        for path in boundary.field_paths
    )
    assert any("path-role:data" in value for value in boundary.path_role_authorities)
    assert any("path-role:output" in value for value in boundary.path_role_authorities)
