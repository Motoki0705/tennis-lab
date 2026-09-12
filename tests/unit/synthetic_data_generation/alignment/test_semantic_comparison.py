"""Exercise paired artifact publication and fit/holdout isolation on tiny inputs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from src.synthetic_data_generation.alignment import semantic_comparison as comparison
from src.synthetic_data_generation.alignment.contracts import (
    GroundPlaneFrame,
    MetricSceneAdapter,
)
from src.synthetic_data_generation.alignment.heatmaps import (
    AlignmentLineHeatmaps,
    AlignmentLineHeatmapView,
    validate_line_heatmaps,
)
from src.synthetic_data_generation.alignment.semantic import SemanticRasterObjective
from src.synthetic_data_generation.alignment.settings import LineProjectionSettings
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.tasks.court_detection.target_schemas import SEMANTIC_LINE_CHANNEL_NAMES


def test_manual_source_rescales_plane_and_evidence_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.synthetic_data_generation.alignment import comparison_source

    (tmp_path / "manual-confirmation.json").write_text("{}")
    frame = GroundPlaneFrame(
        (2.0, 4.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-4.0, 4.0, -4.0, 4.0),
    )
    heatmaps = AlignmentLineHeatmaps(
        frame.bounds_uv_metres,
        1.0,
        2.0,
        2.0,
        (
            AlignmentLineHeatmapView(
                "fit",
                np.ones((2, 2), dtype=np.float32),
                np.asarray([[2.0, 2.0]]),
                np.ones(1, dtype=np.float32),
                np.ones(1),
                True,
            ),
        ),
    )
    original = MetricSceneAdapter.from_nht_scene_from_metric_scene(np.eye(4))
    changed = MetricSceneAdapter.from_nht_scene_from_metric_scene(
        np.diag([2.0, 2.0, 2.0, 1.0])
    )
    monkeypatch.setattr(
        comparison_source,
        "validate_alignment_outputs",
        lambda path: SimpleNamespace(metric_adapter=changed),
    )
    monkeypatch.setattr(
        comparison_source, "validate_line_heatmaps", lambda path: heatmaps
    )
    monkeypatch.setattr(
        comparison_source,
        "load_manual_source",
        lambda path: SimpleNamespace(
            plane=frame, initial=SimpleNamespace(metric_adapter=original)
        ),
    )
    result = comparison_source.load_comparison_baseline(tmp_path)
    assert result.heatmaps.raster_shape == heatmaps.raster_shape
    np.testing.assert_allclose(result.plane.origin_metric_scene, (1, 2, 0))
    np.testing.assert_allclose(result.heatmaps.views[0].points_uv, [[1, 1]])
    np.testing.assert_allclose(
        changed.nht_from_metric_points(
            result.plane.from_uv(result.heatmaps.views[0].points_uv)
        ),
        original.nht_from_metric_points(frame.from_uv(heatmaps.views[0].points_uv)),
    )


@pytest.mark.parametrize("profile", ["b00", "b01", "b02", "b03"])
def test_comparison_configuration_ignores_unrequested_dataset_generation(
    profile: str,
) -> None:
    from hydra import compose, initialize_config_dir

    from src.synthetic_data_generation.alignment.comparison_configuration import (
        ComparisonRuntime,
    )
    from src.utils.paths import PROJECT_ROOT

    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/synthetic_data_generation/configs"),
        version_base="1.3",
    ):
        config = compose(
            config_name="compare_semantic_alignment", overrides=[f"profile={profile}"]
        )
    runtime = ComparisonRuntime.from_config(config)
    assert runtime.workspace.scene_id == profile.upper()


def test_comparison_writes_both_methods_and_never_fits_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)
    camera_pose = np.diag([1.0, -1.0, -1.0, 1.0])
    camera_pose[2, 3] = 20
    cameras = tuple(
        SceneCamera(
            camera_id=name,
            source_frame_index=i,
            width=32,
            height=32,
            intrinsics=(20.0, 0.0, 16.0, 0.0, 20.0, 16.0, 0.0, 0.0, 1.0),
            camera_to_scene=RigidTransform.from_matrix(camera_pose),
            image_path=str(image_path),
        )
        for i, name in enumerate(("fit", "holdout"))
    )
    frame = GroundPlaneFrame(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-16.0, 16.0, -16.0, 16.0),
    )
    baseline_heatmaps = AlignmentLineHeatmaps(
        frame.bounds_uv_metres,
        1.0,
        20.0,
        2.0,
        tuple(
            AlignmentLineHeatmapView(
                camera_id=name,
                probability=np.ones((2, 2), dtype=np.float32),
                points_uv=np.zeros((1, 2)),
                projected_probabilities=np.ones(1, dtype=np.float32),
                proximity_weights=np.ones(1),
                included_in_aggregate=name == "fit",
            )
            for name in ("fit", "holdout")
        ),
    )
    baseline = SimpleNamespace(
        plane=frame,
        heatmaps=baseline_heatmaps,
        result=SimpleNamespace(
            metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
                np.eye(4)
            ),
            partitions=SimpleNamespace(
                fit_camera_ids=("fit",), holdout_camera_ids=("holdout",)
            ),
            layout=SimpleNamespace(
                courts=(
                    SimpleNamespace(
                        court_instance_id="court-1",
                        scene_from_court=RigidTransform.from_matrix(np.eye(4)),
                    ),
                )
            ),
        ),
    )
    monkeypatch.setattr(comparison, "load_comparison_baseline", lambda path: baseline)
    monkeypatch.setattr(
        comparison,
        "validate_standard_scene_export",
        lambda path: SimpleNamespace(scene_id="test", cameras=cameras),
    )

    class SemanticPredictor:
        model_io = None
        device = torch.device("cpu")
        adapter = SimpleNamespace(
            spec=SimpleNamespace(
                target_bundle=SimpleNamespace(
                    targets={
                        "semantic_line": SimpleNamespace(
                            channel_names=SEMANTIC_LINE_CHANNEL_NAMES
                        )
                    }
                )
            )
        )

        @classmethod
        def load_from_checkpoint(cls, *args: Any, **kwargs: Any) -> SemanticPredictor:
            return cls()

        def predict(self, rgb: object) -> SimpleNamespace:
            logits = torch.full((12, 32, 32), -10.0)
            logits[1] = 10.0
            return SimpleNamespace(logits=logits)

    class BinaryPredictor:
        def __init__(self, *args: Any) -> None:
            self.calls = 0

        def predict(self, rgb: object) -> SimpleNamespace:
            self.calls += 1
            # Holdout is deliberately different from the fit view.
            return SimpleNamespace(
                probability=torch.full((32, 32), 0.6 if self.calls == 1 else 1.0)
            )

    monkeypatch.setattr(comparison, "CourtSemanticLinePredictor", SemanticPredictor)
    monkeypatch.setattr(comparison, "CourtLinePredictor", BinaryPredictor)
    scores = []

    def capture_fit(
        objective: SemanticRasterObjective, initial: Any, **kwargs: Any
    ) -> Any:
        scores.append(float(objective.rasters.max()))
        if len(objective.rasters) == 1:
            assert objective.rasters.max() <= 0.6
        return initial

    monkeypatch.setattr(comparison, "refine_placement", capture_fit)
    runtime: Any = SimpleNamespace(
        workspace=SimpleNamespace(root=tmp_path),
        resolver=None,
        alignment=SimpleNamespace(
            evidence=SimpleNamespace(
                seed=880,
                line_model=SimpleNamespace(
                    device="cpu",
                    probability_threshold=0.5,
                    maximum_selected_pixels_per_camera=2000,
                ),
                projection=LineProjectionSettings(0.05, 100.0, 1.0, 20.0, 2.0, 1.0, 1),
            )
        ),
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"test-checkpoint")
    output = tmp_path / "comparison"
    comparison.compare_alignment(
        runtime,
        checkpoint=checkpoint,
        output=output,
        grid_spacing_metres=1.0,
        translation_radius_metres=1.0,
        yaw_radius_radians=0.1,
        smoothing_metres=0.1,
        maximum_iterations=5,
    )
    assert len(scores) == 2
    payload = json.loads((output / "comparison.json").read_text())
    assert payload["acceptance"] == "experimental_not_production_validated"
    assert payload["fit_camera_ids"] == ["fit"]
    assert payload["holdout_camera_ids"] == ["holdout"]
    assert {record["method"] for record in payload["records"]} == {"binary", "semantic"}
    for name in ("binary", "semantic", "legacy"):
        with Image.open(output / f"{name}-projection.png") as image:
            image.verify()
    with Image.open(output / "overlay-holdout.jpg") as image:
        assert image.size == (96, 32)
    restored = validate_line_heatmaps(output / "projections/baseline")
    assert restored.aggregate_camera_ids == ("fit",)
    assert len(restored.views) == 2
