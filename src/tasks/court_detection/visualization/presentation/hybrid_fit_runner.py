"""Run pinned real-image candidate search or export one hybrid fit story."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from src.tasks.base.visualization.presentation.export import render_presentation
from src.tasks.court_detection.data.contracts import CourtRawSample, CourtSampleRecord
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    TennisCourtDetectorInput,
)
from src.tasks.court_detection.inference.contracts import CourtPrediction
from src.tasks.court_detection.inference.predictor import CourtPredictor
from src.tasks.court_detection.visualization.presentation.hybrid_fit import (
    HYBRID_COLOR,
    KP_ONLY_COLOR,
    STAGES,
    build_story,
    draw_court,
    fit_frame,
    render_story_frame,
    story_clock,
    story_stills,
)
from src.tasks.court_detection.visualization.presentation.hybrid_fit_config import (
    HybridFitConfig,
)
from src.tasks.court_detection.visualization.presentation.hybrid_fit_search import (
    FitErrors,
    measure_errors,
)
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": dual_sha256(path)}


def _sample_identity(
    source: TennisCourtDetectorInput,
    record: CourtSampleRecord,
    image: NDArray[np.uint8],
) -> dict[str, object]:
    row = cast(int, record.payload["image_index"])
    return {
        "sample_id": record.sample_id,
        "shard_path": str(record.image_path),
        "image_index": row,
        "jpeg_sha256": bytes(source.image_store.arrays["sha256"][row]).decode("ascii"),
        "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "annotation_sha256": record.payload["source_target_sha256"],
    }


def _dataset_identity(source: TennisCourtDetectorInput) -> dict[str, object]:
    return {
        "kind": "tennis_court_detector",
        "split": "val",
        "descriptor": _file_identity(source.root / "dataset.json"),
        "annotations": _file_identity(source.root / "index.npz"),
    }


def _errors(prediction: CourtPrediction, sample: CourtRawSample) -> FitErrors:
    channels = sample.keypoint_channels
    if channels is None:
        raise ValueError("TennisCourtDetector sample has no GT KP14")
    return measure_errors(
        prediction,
        channels.points_xy[:, 0].numpy(),
        channels.point_visible[:, 0].numpy(),
    )


def _comparison(
    image: NDArray[np.uint8], prediction: CourtPrediction
) -> NDArray[np.uint8]:
    result = prediction.homography
    if result is None or result.best is None or result.kp_only.matrix is None:
        raise ValueError(
            "A comparison requires both accepted hybrid and existing KP-only fits"
        )
    frame = image.copy()
    draw_court(frame, result.kp_only.projected, KP_ONLY_COLOR)
    draw_court(frame, result.best.projected, HYBRID_COLOR)
    return fit_frame(frame, width=480, height=270)


def _contact_sheet(
    ranked: list[tuple[dict[str, Any], NDArray[np.uint8]]], path: Path
) -> None:
    columns = min(4, max(1, len(ranked)))
    sheet: NDArray[np.uint8] = np.full(
        (max(1, math.ceil(len(ranked) / columns)) * 310, columns * 480, 3),
        18,
        dtype=np.uint8,
    )
    if not ranked:
        cv2.putText(
            sheet,
            "No samples meet the configured criteria",
            (12, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (241, 243, 245),
            1,
            cv2.LINE_AA,
        )
    for index, (candidate, frame) in enumerate(ranked):
        left, top = (index % columns) * 480, (index // columns) * 310
        sheet[top : top + 270, left : left + 480] = frame
        label = f"{index + 1}. {candidate['sample_id']}"
        errors = f"KP {candidate['kp_only_error']:.2f}px   hybrid {candidate['hybrid_error']:.2f}px"
        for offset, text in ((285, label), (302, errors)):
            cv2.putText(
                sheet,
                text,
                (left + 8, top + offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (241, 243, 245),
                1,
                cv2.LINE_AA,
            )
    Image.fromarray(sheet).save(path)


def search_candidates(
    runtime: HybridFitConfig, predictor: CourtPredictor, manifest: dict[str, Any]
) -> None:
    """Keep only the explicit thresholds; record empty searches before failing."""
    source = TennisCourtDetectorInput(runtime.source)
    records = source.records("val")[: runtime.search.max_samples]
    manifest["input"] = _dataset_identity(source)
    ranked: list[tuple[dict[str, Any], NDArray[np.uint8]]] = []
    statuses: Counter[str] = Counter()
    scanned: list[dict[str, object]] = []
    qualified = 0
    for index, record in enumerate(records):
        sample = source.load(record)
        image = np.asarray(sample.image.convert("RGB"), dtype=np.uint8)
        prediction = predictor.predict(
            image, postprocess="hybrid", heads=("kp", "line")
        )
        errors = _errors(prediction, sample)
        statuses[errors.status] += 1
        identity = _sample_identity(source, record, image)
        scanned.append({**identity, **asdict(errors)})
        if runtime.search.accepts(errors):
            qualified += 1
            candidate = {
                **identity,
                **asdict(errors),
                "improvement_px": errors.improvement_px,
            }
            ranked.append((candidate, _comparison(image, prediction)))
            ranked.sort(
                key=lambda item: (
                    -item[0]["improvement_px"],
                    item[0]["hybrid_error"],
                    item[0]["sample_id"],
                )
            )
            del ranked[runtime.search.top_k :]
        LOGGER.info(
            "Search %d/%d %s status=%s kp_only=%s hybrid=%s qualified=%d",
            index + 1,
            len(records),
            record.sample_id,
            errors.status,
            errors.kp_only_error,
            errors.hybrid_error,
            qualified,
        )
    runtime.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        runtime.output_dir / "ranking.json",
        {
            "schema": "court_hybrid_fit_search.v1",
            "criteria": asdict(runtime.search),
            "sort": [
                "improvement_px descending",
                "hybrid_error ascending",
                "sample_id ascending",
            ],
            "scanned_count": len(records),
            "qualified_count": qualified,
            "statuses": dict(statuses),
            "ranking": [candidate for candidate, _ in ranked],
            "scanned": scanned,
            "checkpoint": manifest["checkpoint"],
        },
    )
    _contact_sheet(ranked, runtime.output_dir / "contact_sheet.png")
    _write_json(runtime.output_dir / "manifest.json", manifest)
    if not ranked:
        raise ValueError(
            f"No samples met the configured criteria; see {runtime.output_dir / 'ranking.json'}"
        )


def render_sample(
    runtime: HybridFitConfig, predictor: CourtPredictor, manifest: dict[str, Any]
) -> None:
    sample: CourtRawSample | None = None
    if runtime.input_kind == "image":
        assert runtime.image_path is not None
        with Image.open(runtime.image_path) as source_image:
            image = np.array(source_image.convert("RGB"), dtype=np.uint8)
        manifest["input"] = {"kind": "image", **_file_identity(runtime.image_path)}
    else:
        source = TennisCourtDetectorInput(runtime.source)
        records = [
            record
            for record in source.records("val")
            if record.sample_id == runtime.sample_id
        ]
        if len(records) != 1:
            raise ValueError(
                f"Expected one TennisCourtDetector val record for sample_id={runtime.sample_id!r}, got {len(records)}"
            )
        record = records[0]
        sample = source.load(record)
        image = np.asarray(sample.image.convert("RGB"), dtype=np.uint8)
        manifest["input"] = {
            **_dataset_identity(source),
            **_sample_identity(source, record, image),
        }
    prediction = predictor.predict(image, postprocess="hybrid", heads=("kp", "line"))
    story = build_story(prediction, image)
    manifest["errors"] = (
        asdict(_errors(prediction, sample)) if sample is not None else None
    )
    manifest["geometry"] = prediction.geometry_diagnostics()
    manifest["seed_scores"] = [seed.score for seed in story.seeds]
    manifest["kp_only_fit_inliers"] = np.flatnonzero(story.kp_only_fit_inliers).tolist()
    clock = story_clock(runtime.fps)
    start = 0.0
    stages = []
    for name, seconds in STAGES:
        stages.append(
            {
                "name": name,
                "start_seconds": start,
                "stop_seconds": start + seconds,
                "provenance": "measured",
            }
        )
        start += seconds
    manifest["stages"] = stages
    render_presentation(
        lambda frame: fit_frame(
            render_story_frame(story, frame, clock),
            width=runtime.width,
            height=runtime.height,
        ),
        frame_count=clock.frame_count(),
        fps=runtime.fps,
        output_dir=runtime.output_dir,
        name="hybrid_fit",
        gif_width=runtime.gif_width,
        gif_fps=runtime.gif_fps,
        stills=story_stills(clock),
        manifest=manifest,
    )


def run_hybrid_fit(config: DictConfig) -> None:
    runtime = HybridFitConfig.from_config(config)
    if runtime.output_dir.exists() or runtime.output_dir.is_symlink():
        raise FileExistsError(f"output_dir already exists: {runtime.output_dir}")
    manifest: dict[str, Any] = {
        "schema": "court_hybrid_fit_presentation.v1",
        "mode": runtime.mode,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "config": OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        "checkpoint": _file_identity(runtime.checkpoint),
        "provenance": {
            "image": "measured",
            "kp": "measured",
            "line": "measured",
            "kp_only": "measured",
            "seeds": "measured",
            "hybrid": "measured",
            "timing_and_colors": "schematic",
        },
        "selection": "candidate_for_user_review",
    }
    predictor = CourtPredictor.load_from_checkpoint(
        runtime.checkpoint, device=runtime.device, resolver=runtime.resolver
    )
    manifest["checkpoint_identity"] = predictor.checkpoint_identity
    if runtime.mode == "search":
        search_candidates(runtime, predictor, manifest)
    else:
        render_sample(runtime, predictor, manifest)
    LOGGER.info("Hybrid fit %s output: %s", runtime.mode, runtime.output_dir)
