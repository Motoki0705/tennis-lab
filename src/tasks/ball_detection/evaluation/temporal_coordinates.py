"""Direct coordinates on the same immutable owner frames as temporal heatmaps."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.checkpoints.temporal import json_bytes, save_json
from src.tasks.ball_detection.data.windows.all_frames import EDGE_BINS, LABEL_KINDS
from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.evaluation.temporal_heatmap import prediction_identities
from src.tasks.ball_detection.evaluation.temporal_heatmap_metrics import (
    location_metrics,
)
from src.tasks.ball_detection.model_io.mdd_temporal import temporal_forward
from src.tasks.ball_detection.models.mdd import (
    DeepMDDQueryDetector,
    MDDTemporalQueryConfig,
)
from src.tasks.ball_detection.training.runtime.compilation import (
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.runtime.images import (
    coordinate_batches,
    image_input,
    jpeg_decoder_contract,
)
from src.utils.checksum import dual_sha256

EVALUATION_SCHEMA = "ball_temporal_coordinates_evaluation.v1"


def evaluate_temporal_coordinates(
    model: DeepMDDQueryDetector,
    manifest: Path,
    *,
    split: str,
    runtime: CoordinateRuntime,
    device: torch.device,
    artifact_sha256: str,
    distance_px: float,
    calibration_report: dict[str, Any] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    if type(model.config) is not MDDTemporalQueryConfig or split not in {"val", "test"}:
        raise ValueError(
            "Explicit temporal query config and validation/test split required"
        )
    if (
        not math.isfinite(distance_px)
        or distance_px <= 0
        or calibration_report is not None
    ):
        raise ValueError("Direct coordinates have no presence threshold or calibration")
    started = time.perf_counter()
    dataset = FrozenBallWindowDataset(
        manifest, split=split, requires_pose=False, jpeg_decoder=runtime.jpeg_decoder
    )
    if (
        not dataset.all_frames
        or dataset.manifest["mode"] != "evaluation"
        or dataset.window_length != model.config.frames
    ):
        raise ValueError("Coordinates require the frozen owner manifest at the saved T")
    records = []
    model.eval()
    with torch.inference_mode(), coordinate_compile_scope(model):
        for batch in coordinate_batches(
            runtime.loader(dataset, batch_size=1),
            device,
            prefetch=runtime.image_prefetch,
        ):
            rgb = image_input(batch, device)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=runtime.precision == "bf16"
            ):
                uv = (
                    temporal_forward(model, rgb, batch["timestamps"].to(device))
                    .float()
                    .cpu()
                )
            if uv.shape != batch["uv"].shape or not bool(torch.isfinite(uv).all()):
                raise ValueError("Invalid direct-coordinate prediction")
            errors = ((uv - batch["uv"]) * (batch["source_size"][:, None] - 1)).norm(
                dim=-1
            )
            clip_id = batch["clip_id"][0]
            clip = dataset.store.clip_by_id(clip_id)
            for offset in (
                torch.nonzero(batch["owner_mask"][0], as_tuple=False).flatten().tolist()
            ):
                frame = int(batch["frame_indices"][0, offset])
                kind = LABEL_KINDS[int(batch["label_kind"][0, offset])]
                row = dataset.store.row_of(clip, frame)
                natural = (
                    bool(
                        dataset.store.instances["occluded"][
                            dataset.store.frames["inst_start"][row]
                        ]
                    )
                    if kind == "observed"
                    else None
                )
                records.append(
                    dict(
                        clip_id=clip_id,
                        frame_index=frame,
                        frame_step=1,
                        native_fps=batch["native_fps"][0],
                        kind=kind,
                        source=clip.source,
                        source_group=f"{clip.source}/{clip.group_id}",
                        natural_occlusion=natural,
                        pred_uv=uv[0, offset].tolist(),
                        error_px=float(errors[0, offset])
                        if kind == "observed"
                        else None,
                        pts=int(batch["pts"][0, offset]),
                        time_base=batch["time_base"][0],
                        owner_start=batch["start"][0],
                        owner_offset=offset,
                        context_left=offset,
                        context_right=model.config.frames - 1 - offset,
                        mdd_start=offset == 0,
                        edge_distance=int(batch["edge_distance"][0, offset]),
                        edge_bin=int(batch["edge_bin"][0, offset]),
                        edge_distance_seconds=float(
                            batch["edge_distance_seconds"][0, offset]
                        ),
                    )
                )
    keys = [(r["clip_id"], r["frame_index"]) for r in records]
    if keys != sorted(set(keys)):
        raise ValueError("Coordinate owner frames must be ordered and unique")
    identities = prediction_identities(records)
    expected = dataset.manifest["summaries"][split]
    if (
        any(expected[key] != value for key, value in identities.items())
        or len(records) != expected["counts"]["frames"]
    ):
        raise ValueError(
            "Actual coordinate owner/teacher membership differs from the manifest"
        )
    metrics = location_metrics(records, distance_px=distance_px)
    for field, group in (
        ("source", "by_source"),
        ("native_fps", "by_fps"),
        ("source_group", "by_source_group"),
        ("natural_occlusion", "by_natural_occlusion"),
    ):
        metrics[group] = {
            str(v): location_metrics(
                [r for r in records if r[field] == v], distance_px=distance_px
            )
            for v in sorted({r[field] for r in records}, key=str)
        }
    metrics["by_edge_distance"] = {
        name: location_metrics(
            [r for r in records if r["edge_bin"] == i], distance_px=distance_px
        )
        for i, name in enumerate(EDGE_BINS)
    }
    report = dict(
        schema=EVALUATION_SCHEMA,
        split=split,
        artifact_sha256=artifact_sha256,
        manifest_sha256=dual_sha256(manifest),
        snapshot=dataset.manifest["ball_store"],
        model_config=asdict(model.config),
        model_kind="mdd_temporal_query",
        runtime=asdict(runtime),
        image_decode=jpeg_decoder_contract(runtime.jpeg_decoder),
        identities=identities,
        metrics=metrics,
        absence_detection=dict(
            available=False,
            reason="direct coordinates have no presence output; negative frames retained in coverage",
        ),
        seconds=time.perf_counter() - started,
    )
    if output is not None:
        predictions = output.with_suffix(".frames.jsonl")
        temporary = predictions.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            for record in records:
                stream.write(json_bytes(record))
        temporary.replace(predictions)
        report["predictions"] = dict(
            file=predictions.name, sha256=dual_sha256(predictions), frames=len(records)
        )
        save_json(output, report)
    return report
