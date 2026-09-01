"""Tests for synthetic dataset visualization request contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.synthetic_data_generation.visualization.configuration import (
    build_visualization_request,
)
from src.synthetic_data_generation.visualization.contracts import (
    CourtAABBRenderStyle,
    CourtAABBTrajectoryFilterRadiusMode,
    CourtAABBTrajectoryFilterScope,
    CourtAABBWireframeTopology,
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
        "render_style": "wireframe",
        "wireframe_topology": "boundary",
        "trajectory_filter_scope": "local_swept_segments",
        "trajectory_filter_radius_mode": "explicit_radius",
        "trajectory_filter_radius_m": 1.5,
        "color_rgb": [255, 96, 32],
        "background_color_rgb": [0, 0, 0],
        "opacity": 0.55,
        "edge_opacity": 0.40,
        "edge_width_px": 1,
        "depth_epsilon_m": 0.02,
        "near_plane_m": 0.05,
        "maximum_cells": 1_000_000,
        "maximum_surface_faces": 4_000_000,
        "maximum_edge_segments": 8_000_000,
        "maximum_projected_pixels": 100_000_000,
    }


def _court_overlay_contract(
    *,
    mode: CourtOverlayMode = CourtOverlayMode.SEMANTIC,
    render_style: CourtAABBRenderStyle = CourtAABBRenderStyle.WIREFRAME,
    wireframe_topology: CourtAABBWireframeTopology = (
        CourtAABBWireframeTopology.BOUNDARY
    ),
    trajectory_filter_scope: CourtAABBTrajectoryFilterScope = (
        CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS
    ),
    trajectory_filter_radius_mode: CourtAABBTrajectoryFilterRadiusMode | None = (
        CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS
    ),
    trajectory_filter_radius_m: float | None = 1.5,
) -> CourtOverlayConfiguration:
    return CourtOverlayConfiguration(
        mode=mode,
        render_style=render_style,
        wireframe_topology=wireframe_topology,
        trajectory_filter_scope=trajectory_filter_scope,
        trajectory_filter_radius_mode=trajectory_filter_radius_mode,
        trajectory_filter_radius_m=trajectory_filter_radius_m,
        color_rgb=(255, 96, 32),
        background_color_rgb=(0, 0, 0),
        opacity=0.55,
        edge_opacity=0.40,
        edge_width_px=1,
        depth_epsilon_m=0.02,
        near_plane_m=0.05,
        maximum_cells=1_000_000,
        maximum_surface_faces=4_000_000,
        maximum_edge_segments=8_000_000,
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
    assert request.court_overlay.render_style is CourtAABBRenderStyle.WIREFRAME
    assert (
        request.court_overlay.wireframe_topology
        is CourtAABBWireframeTopology.BOUNDARY
    )
    assert (
        request.court_overlay.trajectory_filter_scope
        is CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS
    )
    assert (
        request.court_overlay.trajectory_filter_radius_mode
        is CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS
    )
    assert request.court_overlay.trajectory_filter_radius_m == 1.5

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

    solid_overlay = _court_overlay_config(mode="trajectory_support_aabb")
    solid_overlay["render_style"] = "solid"
    solid_request = build_visualization_request(
        OmegaConf.create(
            {
                "roots": roots,
                "visualization": {**visualization, "court_overlay": solid_overlay},
            }
        )
    )
    assert solid_request.court_overlay.render_style is CourtAABBRenderStyle.SOLID

    all_edges_overlay = _court_overlay_config(mode="trajectory_support_aabb")
    all_edges_overlay["wireframe_topology"] = "all_edges"
    all_edges_request = build_visualization_request(
        OmegaConf.create(
            {
                "roots": roots,
                "visualization": {
                    **visualization,
                    "court_overlay": all_edges_overlay,
                },
            }
        )
    )
    assert (
        all_edges_request.court_overlay.wireframe_topology
        is CourtAABBWireframeTopology.ALL_EDGES
    )

    explicit_filter_overlay = _court_overlay_config(
        mode="trajectory_support_aabb"
    )
    explicit_filter_overlay["trajectory_filter_radius_mode"] = "explicit_radius"
    explicit_filter_overlay["trajectory_filter_radius_m"] = 1.25
    explicit_filter_request = build_visualization_request(
        OmegaConf.create(
            {
                "roots": roots,
                "visualization": {
                    **visualization,
                    "court_overlay": explicit_filter_overlay,
                },
            }
        )
    )
    assert (
        explicit_filter_request.court_overlay.trajectory_filter_radius_mode
        is CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS
    )
    assert explicit_filter_request.court_overlay.trajectory_filter_radius_m == 1.25

    all_filter_overlay = _court_overlay_config(mode="trajectory_support_aabb")
    all_filter_overlay["trajectory_filter_scope"] = "all"
    all_filter_overlay["trajectory_filter_radius_mode"] = None
    all_filter_overlay["trajectory_filter_radius_m"] = None
    all_filter_request = build_visualization_request(
        OmegaConf.create(
            {
                "roots": roots,
                "visualization": {
                    **visualization,
                    "court_overlay": all_filter_overlay,
                },
            }
        )
    )
    assert (
        all_filter_request.court_overlay.trajectory_filter_scope
        is CourtAABBTrajectoryFilterScope.ALL
    )
    assert all_filter_request.court_overlay.trajectory_filter_radius_mode is None

    invalid_overlay = dict(solid_overlay)
    invalid_overlay["render_style"] = "filled"
    with pytest.raises(ValueError, match="render_style must be wireframe or solid"):
        build_visualization_request(
            OmegaConf.create(
                {
                    "roots": roots,
                    "visualization": {
                        **visualization,
                        "court_overlay": invalid_overlay,
                    },
                }
            )
        )

    invalid_topology_overlay = dict(solid_overlay)
    invalid_topology_overlay["wireframe_topology"] = "seams"
    with pytest.raises(
        ValueError,
        match="wireframe_topology must be boundary or all_edges",
    ):
        build_visualization_request(
            OmegaConf.create(
                {
                    "roots": roots,
                    "visualization": {
                        **visualization,
                        "court_overlay": invalid_topology_overlay,
                    },
                }
            )
        )

    invalid_filter_overlay = dict(solid_overlay)
    invalid_filter_overlay["trajectory_filter_scope"] = "nearest_camera"
    with pytest.raises(ValueError, match="trajectory_filter_scope must be"):
        build_visualization_request(
            OmegaConf.create(
                {
                    "roots": roots,
                    "visualization": {
                        **visualization,
                        "court_overlay": invalid_filter_overlay,
                    },
                }
            )
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


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"edge_opacity": 0.0}, "edge_opacity"),
        ({"edge_width_px": 0}, "edge_width_px"),
        ({"edge_width_px": 65}, "edge_width_px"),
        ({"maximum_edge_segments": 0}, "maximum_edge_segments"),
        (
            {
                "trajectory_filter_radius_mode": (
                    CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS
                ),
                "trajectory_filter_radius_m": None,
            },
            "explicit_radius",
        ),
        (
            {
                "trajectory_filter_scope": CourtAABBTrajectoryFilterScope.ALL,
                "trajectory_filter_radius_mode": None,
                "trajectory_filter_radius_m": 1.0,
            },
            "require.*None",
        ),
        (
            {
                "trajectory_filter_scope": CourtAABBTrajectoryFilterScope.ALL,
                "trajectory_filter_radius_mode": (
                    CourtAABBTrajectoryFilterRadiusMode.SUPPORT_RADIUS
                ),
            },
            "require.*None",
        ),
        (
            {"trajectory_filter_radius_mode": None},
            "requires trajectory_filter_radius_mode",
        ),
        (
            {
                "trajectory_filter_radius_mode": (
                    CourtAABBTrajectoryFilterRadiusMode.SUPPORT_RADIUS
                ),
                "trajectory_filter_radius_m": 1.0,
            },
            "support_radius.*None",
        ),
        (
            {
                "trajectory_filter_radius_mode": "support_radius",
            },
            "CourtAABBTrajectoryFilterRadiusMode",
        ),
        (
            {"trajectory_filter_scope": "local_swept_segments"},
            "CourtAABBTrajectoryFilterScope",
        ),
        ({"render_style": "wireframe"}, "CourtAABBRenderStyle"),
        (
            {"wireframe_topology": "boundary"},
            "CourtAABBWireframeTopology",
        ),
    ),
)
def test_court_overlay_contract_rejects_invalid_wireframe_policy(
    changes: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {
        "mode": CourtOverlayMode.TRAJECTORY_SUPPORT_AABB,
        "render_style": CourtAABBRenderStyle.WIREFRAME,
        "wireframe_topology": CourtAABBWireframeTopology.BOUNDARY,
        "trajectory_filter_scope": (
            CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS
        ),
        "trajectory_filter_radius_mode": (
            CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS
        ),
        "trajectory_filter_radius_m": 1.5,
        "color_rgb": (255, 96, 32),
        "background_color_rgb": (0, 0, 0),
        "opacity": 0.55,
        "edge_opacity": 0.40,
        "edge_width_px": 1,
        "depth_epsilon_m": 0.02,
        "near_plane_m": 0.05,
        "maximum_cells": 1_000_000,
        "maximum_surface_faces": 4_000_000,
        "maximum_edge_segments": 8_000_000,
        "maximum_projected_pixels": 100_000_000,
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError), match=match):
        CourtOverlayConfiguration(**values)  # type: ignore[arg-type]


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
