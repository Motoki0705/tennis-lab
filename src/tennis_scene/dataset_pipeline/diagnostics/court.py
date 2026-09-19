"""Compare configured Court checkpoints/ball-guided crops on a labelled clip.

Manual court annotations are evaluation targets only. Crop proposals use only
outsourced observed ball centres, never court labels. Run through training queue.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.court_detection.geometry.pose import (
    canonical_semantic_court_points_batched,
    project_predicted_canonical_points,
)
from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor
from src.tasks.court_detection.model_io import CourtDecodedOutput
from src.tasks.court_detection.model_io.images import prepare_court_input
from src.tasks.court_detection.models.pose_head import CourtModelOutput
from src.tennis_scene.dataset_pipeline.court import fit_static_court
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.reference_pipeline.observations import court_homographies
from src.utils.configuration import PathResolver

from .configuration import CourtProbeConfig
from .media import sample_frames


def probe(spec: CourtProbeConfig) -> None:
    runtime, cfg = spec.runtime, spec.build
    runtime.output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, runtime.output / "build_config.yaml", resolve=True)
    clip_dir = (
        runtime.source / "videos" / runtime.clip_ids[0].replace("/", "/clips/", 1)
    )
    clip = ClipManifest.load(clip_dir)
    targets, _ = court_homographies(clip_dir)
    pixels = np.asarray([clip.width, clip.height])
    frames: dict[str, list[np.ndarray]] = {}
    rois = {}
    indices = np.unique(
        np.linspace(0, clip.num_frames - 1, int(spec.samples)).round().astype(int)
    )
    for camera in clip.camera_ids:
        frames[camera] = sample_frames(clip, camera, indices)
        source = json.loads(
            (clip_dir / "outsource" / f"{camera}_annotations.json").read_text()
        )
        points = np.asarray(
            [
                [item["center_px"]["x"], item["center_px"]["y"]]
                for item in source["frames"]
                if item["status"] == "observed"
            ]
        )
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise ValueError(
                f"{camera}: crop proposals require finite observed ball points"
            )
        low, high = np.percentile(points, [1, 99], axis=0)
        span = high - low
        rois[camera] = {"full": (0, 0, clip.width, clip.height)}
        for margin in spec.margins:
            start = np.maximum(0, np.floor(low - float(margin) * span)).astype(int)
            end = np.minimum(pixels, np.ceil(high + float(margin) * span)).astype(int)
            if np.any(end <= start):
                raise ValueError(f"{camera}: empty ball-guided crop")
            rois[camera][f"ball_context_{margin}"] = (*start, *end)
    rows = []
    resolver = PathResolver(
        replace(
            runtime.resolver.roots, checkpoint_root=runtime.resolver.roots.output_root
        )
    )
    for checkpoint_index, checkpoint in enumerate(spec.checkpoints):
        predictor = CourtKeypointPredictor.load_from_checkpoint(
            checkpoint,
            resolver=resolver,
            device=str(cfg.device),
            subpixel_refine=True,
            peak_threshold=runtime.court.min_score,
        )
        for view, camera in enumerate(clip.camera_ids):
            for variant, (x0, y0, x1, y1) in rois[camera].items():
                raw_frames, score_frames, projected_poses = [], [], []
                for image in frames[camera]:
                    crop = image[y0:y1, x0:x1]
                    prediction = predictor.predict(crop)
                    raw_frames.append(
                        prediction.keypoints[:, 0].cpu().numpy() + [x0, y0]
                    )
                    score_frames.append(prediction.scores[:, 0].cpu().numpy())
                    prepared = prepare_court_input(
                        crop, spec=predictor.adapter.spec, device=predictor.device
                    )
                    images = prepared.images
                    with torch.no_grad():
                        call = predictor.adapter.prepare_images(images)
                        output = predictor.model(*call.model_args)
                        if (
                            isinstance(output, CourtModelOutput)
                            and output.pose is not None
                        ):
                            decoded = predictor.adapter.decode_output(output)
                            if not isinstance(decoded, CourtDecodedOutput):
                                raise TypeError(
                                    "Pose model output must decode to CourtDecodedOutput"
                                )
                            canonical_points = canonical_semantic_court_points_batched(
                                torch.arange(14, device=predictor.device)[None]
                            )
                            ih, iw = images.shape[-2:]
                            projection = project_predicted_canonical_points(
                                decoded.pose,
                                canonical_points,
                                canonical_points.new_tensor([[iw / 2, ih / 2]]),
                            )
                            projected_poses.append(
                                projection.points_xy[0].cpu().numpy()
                                * prepared.source_from_model_xy
                                + [x0, y0]
                            )
                raw, scores = np.asarray(raw_frames), np.asarray(score_frames)
                row: dict[str, object] = dict(
                    checkpoint=str(checkpoint),
                    camera=camera,
                    variant=variant,
                    roi=[int(x) for x in (x0, y0, x1, y1)],
                )
                errors = np.linalg.norm(raw - targets[view, :1] * pixels, axis=-1)
                supported = scores >= runtime.court.min_score
                row.update(
                    raw_error_px_median=float(np.median(errors[supported]))
                    if supported.any()
                    else None,
                    detected_fraction=float(supported.mean()),
                )
                if projected_poses:
                    row["pose_error_px_median"] = float(
                        np.median(
                            np.linalg.norm(
                                np.asarray(projected_poses)
                                - targets[view, :1] * pixels,
                                axis=-1,
                            )
                        )
                    )
                try:
                    fitted, _, diagnostic = fit_static_court(
                        raw,
                        scores,
                        size=(clip.width, clip.height),
                        settings=runtime.court,
                    )
                    fitted_errors = np.linalg.norm(
                        (fitted - targets[view, 0]) * pixels, axis=-1
                    )
                    row.update(
                        fit_error_to_manual_px_median=float(np.median(fitted_errors)),
                        fit_error_to_manual_px_max=float(np.max(fitted_errors)),
                        **diagnostic,
                    )
                except ValueError as error:
                    row["rejected"] = str(error)
                prefix = f"{checkpoint_index}_{camera}_{variant}"
                np.savez_compressed(
                    runtime.output / f"{prefix}.npz",
                    keypoints_px=raw,
                    scores=scores,
                    pose_px=projected_poses,
                    target_px=targets[view, 0] * pixels,
                    indices=indices,
                )
                rows.append(row)
                (runtime.output / "metrics.json").write_text(json.dumps(rows, indent=2))
                print(json.dumps(row), flush=True)
        del predictor
        torch.cuda.empty_cache()
