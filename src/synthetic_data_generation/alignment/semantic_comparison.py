"""Reproducible paired binary/semantic comparison, without publishing acceptance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.alignment.evidence_source import (
    _GroundPlane,
    _project_probability_to_ground,
)
from src.synthetic_data_generation.alignment.heatmaps import (
    AlignmentLineHeatmaps,
    AlignmentLineHeatmapView,
    aggregate_line_heatmaps,
    write_line_heatmaps,
)
from src.synthetic_data_generation.alignment.semantic import (
    SemanticRasterObjective,
    refine_placement,
    transform_segments,
)
from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.reconstruction.scene_export import (
    validate_standard_scene_export,
)
from src.synthetic_data_generation.visualization.publication.alignment import (
    load_alignment_publication_data,
)
from src.tasks.court_detection.inference.mask_predictor import (
    CourtLinePredictor,
    CourtSemanticLinePredictor,
)
from src.tasks.court_detection.inference.semantic_lines import (
    INVARIANT_LINE_NAMES,
    INVARIANT_LINE_SCHEMA,
    merge_camera_line_probabilities,
)
from src.utils.seeding import seed_everything


def compare_alignment(
    runtime: ScenePipelineConfiguration,
    *,
    checkpoint: Path,
    output: Path,
    grid_spacing_metres: float,
    translation_radius_metres: float,
    yaw_radius_radians: float,
    smoothing_metres: float,
    maximum_iterations: int,
) -> None:
    """Use one model, fixed camera partitions, identical bounds and paired searches."""
    if output.exists():
        raise FileExistsError(f"Comparison output already exists: {output}")
    scene_root = runtime.workspace.root
    baseline = load_alignment_publication_data(scene_root / "alignment")
    scene = validate_standard_scene_export(
        scene_root / "reconstruction/export/scene.json"
    )
    settings = runtime.alignment.evidence
    seed_everything(settings.seed)
    semantic = CourtSemanticLinePredictor.load_from_checkpoint(
        checkpoint,
        resolver=runtime.resolver,
        device=settings.line_model.device,
    )
    binary = CourtLinePredictor(semantic.model_io, semantic.device)
    spec = semantic.adapter.spec.target_bundle.targets["semantic_line"]
    frame = baseline.evidence.ground_plane_frame
    scale = baseline.result.metric_adapter.nht_scene_units_per_metre
    normal = np.asarray(frame.normal_metric_scene)
    origin = np.asarray(frame.origin_metric_scene)
    plane = _GroundPlane(
        normal=normal,
        offset=-float(normal @ origin),
        origin=origin,
        basis_u=np.asarray(frame.basis_u_metric_scene),
        basis_v=np.asarray(frame.basis_v_metric_scene),
        support_uv_bounds=frame.bounds_uv_metres,
    )
    projection = replace(
        settings.projection,
        bounds_margin=settings.projection.bounds_margin / scale,
        maximum_ray_distance=settings.projection.maximum_ray_distance / scale,
        proximity_scale=settings.projection.proximity_scale / scale,
        grid_spacing=grid_spacing_metres,
    )
    # Preserve the baseline projection rectangle in metric coordinates.
    margin = settings.projection.bounds_margin / scale
    u0, u1, v0, v1 = frame.bounds_uv_metres
    plane = replace(
        plane, support_uv_bounds=(u0 + margin, u1 - margin, v0 + margin, v1 - margin)
    )
    cameras = {camera.camera_id: camera for camera in scene.cameras}
    fit_ids = set(baseline.result.partitions.fit_camera_ids)
    holdout_ids = set(baseline.result.partitions.holdout_camera_ids)
    names = ("binary", *INVARIANT_LINE_NAMES[1:])
    views: dict[str, list[AlignmentLineHeatmapView]] = {name: [] for name in names}
    for camera_id in baseline.heatmaps.camera_ids:
        camera = cameras[camera_id]
        metric_camera = replace(
            camera,
            camera_to_scene=baseline.result.metric_adapter.metric_from_nht_camera(
                camera.camera_to_scene
            ),
        )
        with Image.open(camera.image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        probabilities = merge_camera_line_probabilities(
            semantic.predict(rgb).logits.float().softmax(dim=0).numpy(),
            channel_names=spec.channel_names,
        )
        channel_maps = (binary.predict(rgb).probability.numpy(), *probabilities[1:])
        for name, probability in zip(names, channel_maps, strict=True):
            projected = _project_probability_to_ground(
                probability,
                camera=metric_camera,
                plane=plane,
                model_settings=settings.line_model,
                projection_settings=projection,
            )
            views[name].append(
                AlignmentLineHeatmapView(
                    camera_id=camera_id,
                    probability=probability,
                    points_uv=projected.points_uv,
                    projected_probabilities=projected.probabilities,
                    proximity_weights=projected.proximity_weights,
                    included_in_aggregate=camera_id in fit_ids
                    and len(projected.points_uv) > 0,
                )
            )
        print(f"Projected {scene.scene_id}/{camera_id}", flush=True)
    heatmaps = {
        name: AlignmentLineHeatmaps(
            bounds_uv=frame.bounds_uv_metres,
            grid_spacing=grid_spacing_metres,
            proximity_scale=projection.proximity_scale,
            proximity_power=projection.proximity_power,
            views=tuple(items),
        )
        for name, items in views.items()
    }
    fit_rasters = {
        name: aggregate_line_heatmaps(value).evidence_sum
        for name, value in heatmaps.items()
    }
    holdout_rasters = {
        name: aggregate_line_heatmaps(
            replace(
                value,
                views=tuple(
                    replace(
                        view,
                        included_in_aggregate=view.camera_id in holdout_ids
                        and len(view.points_uv) > 0,
                    )
                    for view in value.views
                ),
            )
        ).evidence_sum
        for name, value in heatmaps.items()
    }
    objectives = {
        mode: SemanticRasterObjective(
            np.stack([fit_rasters[name] for name in selected]),
            frame.bounds_uv_metres,
            grid_spacing_metres,
        )
        for mode, selected in (("binary", ("binary",)), ("semantic", names[1:]))
    }
    holdout = {
        mode: SemanticRasterObjective(
            np.stack([holdout_rasters[name] for name in selected]),
            frame.bounds_uv_metres,
            grid_spacing_metres,
        )
        for mode, selected in (("binary", ("binary",)), ("semantic", names[1:]))
    }
    placements: dict[str, list[NDArray[np.float64]]] = {
        "legacy": [],
        "binary": [],
        "semantic": [],
    }
    records = []
    for court in baseline.result.layout.courts:
        transform = court.scene_from_court.matrix()
        center = frame.to_uv(transform[:3, 3][None])[0]
        direction = transform[:3, 0]
        yaw = np.arctan2(
            direction @ np.asarray(frame.basis_v_metric_scene),
            direction @ np.asarray(frame.basis_u_metric_scene),
        )
        initial = np.asarray([*center, yaw])
        placements["legacy"].append(initial)
        for mode, objective in objectives.items():
            refined = refine_placement(
                objective,
                initial,
                translation_radius_metres=translation_radius_metres,
                yaw_radius_radians=yaw_radius_radians,
                smoothing_metres=smoothing_metres,
                seed=settings.seed,
                maximum_iterations=maximum_iterations,
            )
            placements[mode].append(refined)
            records.append(
                {
                    "court_id": court.court_instance_id,
                    "method": mode,
                    "initial_uv_yaw": initial.tolist(),
                    "refined_uv_yaw": refined.tolist(),
                    "fit_support_before": objective.score(initial),
                    "fit_support_after": objective.score(refined),
                    "holdout_support_before": holdout[mode].score(initial),
                    "holdout_support_after": holdout[mode].score(refined),
                }
            )
    output.mkdir(parents=True)
    (output / "projections").mkdir()
    for name, value in heatmaps.items():
        write_line_heatmaps(output / "projections" / name, heatmaps=value)
    projection_rasters = {
        mode: objective.rasters.sum(axis=0) for mode, objective in objectives.items()
    }
    projection_rasters["legacy"] = aggregate_line_heatmaps(
        baseline.heatmaps
    ).evidence_sum
    for mode, raster in projection_rasters.items():
        figure = Figure(figsize=(14, 6), layout="constrained")
        FigureCanvasAgg(figure)
        for ax, show_lines in zip(figure.subplots(1, 2), (False, True), strict=True):
            picture = ax.imshow(
                raster,
                origin="lower",
                extent=frame.bounds_uv_metres,
                cmap="magma",
                vmin=0,
            )
            if show_lines:
                for parameters in placements[mode]:
                    for segment in transform_segments(parameters):
                        ax.plot(
                            segment[:, 0], segment[:, 1], color="cyan", linewidth=0.8
                        )
            ax.set(
                title=f"{mode}: {'alignment overlay' if show_lines else 'weighted projection'}",
                xlabel="u (m)",
                ylabel="v (m)",
            )
            figure.colorbar(picture, ax=ax, label="weighted probability sum")
        figure.savefig(output / f"{mode}-projection.png", dpi=130)
    for camera_id in baseline.heatmaps.camera_ids:
        camera = cameras[camera_id]
        with Image.open(camera.image_path) as image:
            original = np.asarray(image.convert("RGB"))
        panels = []
        for mode in placements:
            canvas = original.copy()
            for parameters in placements[mode]:
                for segment in transform_segments(parameters):
                    metric = frame.from_uv(segment)
                    points = baseline.result.metric_adapter.nht_from_metric_points(
                        metric
                    )
                    camera_points = camera.camera_to_scene.inverse().apply(points)
                    if np.any(camera_points[:, 2] <= 0):
                        continue
                    uv, depth = camera.project_scene_points(points)
                    if (
                        np.all(depth > 0)
                        and np.isfinite(uv).all()
                        and np.max(np.abs(uv)) < 1e7
                    ):
                        cv2.polylines(
                            canvas,
                            [np.round(uv).astype(np.int32)],
                            False,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
            cv2.putText(
                canvas,
                f"{mode} | {camera_id} | {'FIT' if camera_id in fit_ids else 'HOLDOUT' if camera_id in holdout_ids else 'EXCLUDED'}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            panels.append(canvas)
        Image.fromarray(np.concatenate(panels, axis=1)).save(
            output / f"overlay-{camera_id}.jpg", quality=90
        )
    with checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "comparison.json").write_text(
        json.dumps(
            {
                "schema": INVARIANT_LINE_SCHEMA,
                "scene_id": scene.scene_id,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "channel_names": list(INVARIANT_LINE_NAMES),
                "baseline_owner": str(scene_root / "alignment"),
                "camera_ids": list(baseline.heatmaps.camera_ids),
                "fit_camera_ids": sorted(fit_ids),
                "holdout_camera_ids": sorted(holdout_ids),
                "ground_plane": frame.to_dict(),
                "metric_adapter": baseline.result.metric_adapter.to_dict(),
                "seed": settings.seed,
                "grid_spacing_metres": grid_spacing_metres,
                "translation_radius_metres": translation_radius_metres,
                "yaw_radius_radians": yaw_radius_radians,
                "smoothing_metres": smoothing_metres,
                "maximum_iterations": maximum_iterations,
                "acceptance": "experimental_not_production_validated",
                "records": records,
                "comparison_note": "Fixed baseline court count and scale. Support values across binary/semantic objectives are not directly comparable. Visual review required.",
            },
            indent=2,
        )
        + "\n"
    )
