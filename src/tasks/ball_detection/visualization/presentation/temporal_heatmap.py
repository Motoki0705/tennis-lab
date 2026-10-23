"""Stream matching owner-frame predictions into a labelled comparison video."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from src.tasks.ball_detection.checkpoints.temporal_heatmap import save_json
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.evaluation.temporal_heatmap import EVALUATION_SCHEMA
from src.tasks.ball_detection.evaluation.temporal_heatmap_metrics import present
from src.tasks.ball_detection.visualization.rendering.clip_renderer import (
    DrawStyle,
    render_overlay_frames,
)
from src.tasks.base.visualization.layout import compose_row, label_panel
from src.utils.checksum import dual_sha256
from src.utils.video.writer import VideoWriter


def render_temporal_heatmap_comparison(evaluations: list[Path], *, clip_id: str, start: int, stop: int,
                                       output: Path, panel_width: int = 480) -> dict[str, Any]:
    if len(evaluations) < 2 or not 0 <= start < stop or panel_width < 2 or panel_width % 2:
        raise ValueError("Specify at least two evaluations, a nonempty clip range and an even panel width")
    if output.suffix != ".mp4" or output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Use a new .mp4 output path")
    reports = [json.loads(path.read_text()) for path in evaluations]
    first = reports[0]
    for report in reports:
        if (report["schema"] != EVALUATION_SCHEMA or report["split"] != first["split"]
                or report["identities"] != first["identities"] or report["snapshot"] != first["snapshot"]
                or report["image_decode"] != first["image_decode"] or report["runtime"]["precision"] != first["runtime"]["precision"]):
            raise ValueError("Compare evaluations on identical frames, teachers, snapshot, decoder and precision")
    store_root = Path(first["snapshot"]["directory"])
    if any(dual_sha256(store_root / name) != digest for name, digest in first["snapshot"]["hashes"].items()):
        raise ValueError("Comparison snapshot identity changed")
    store = BallFrameStore(store_root)
    clip = store.clip_by_id(clip_id)
    if clip.split != first["split"] or stop > clip.frame_count:
        raise ValueError("Comparison range leaves the evaluated split/clip")
    selected = []
    for path, report in zip(evaluations, reports, strict=True):
        name = report["predictions"]["file"]
        predictions = path.parent / name
        if Path(name).name != name or dual_sha256(predictions) != report["predictions"]["sha256"]:
            raise ValueError("Comparison predictions path/hash mismatch")
        frames = {}
        with predictions.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["clip_id"] == clip_id and start <= record["frame_index"] < stop:
                    if record["frame_index"] in frames:
                        raise ValueError("Duplicate comparison owner frame")
                    frames[record["frame_index"]] = record
        if set(frames) != set(range(start, stop)):
            raise ValueError("Comparison range is not completely covered by each evaluation")
        selected.append(frames)
    height = max(2, round(clip.height * panel_width / clip.width / 2) * 2)
    draw = DrawStyle(gt_radius=4, pred_radius=4, thickness=1, gt_color_rgb=(0, 255, 0),
                     pred_color_rgb=(255, 140, 0), text_color_rgb=(245, 245, 245), muted_text_color_rgb=(170, 170, 170))
    with VideoWriter(output, fps=float(clip.fps)) as writer:
        for frame in range(start, stop):
            rgb = store.read_bgr(store.row_of(clip, frame))[..., ::-1].copy()
            rgb = cv2.resize(rgb, (panel_width, height))
            panels = []
            for report, records in zip(reports, selected, strict=True):
                record = records[frame]
                # Show the raw coordinate even when below the validation presence threshold.
                panel = render_overlay_frames(frames_rgb=[rgb], pred_coords_normalized=[tuple(record["pred_uv"])],
                                               pred_visibility=[True], draw=draw)[0]
                mode = "present" if present(record["score"], report["calibration"]) else "absent"
                label = f"{report['model_kind']} T={report['model_config']['frames']} | {frame} | {mode} {record['score']:.3f}"
                panels.append(label_panel(panel, text=label, label_height=32, background_rgb=(18, 18, 18),
                                          text_color_rgb=(245, 245, 245), text_scale=.4, text_thickness=1))
            writer.write_frame(compose_row(panels=panels, tile_gap=8, background_rgb=(18, 18, 18)))
    result = dict(schema="ball_temporal_heatmap_comparison.v1", clip_id=clip_id, start=start, stop=stop,
                  frames=stop - start, fps=clip.fps, evaluations=[dict(file=str(path), sha256=dual_sha256(path)) for path in evaluations],
                  identities=first["identities"], output_sha256=dual_sha256(output),
                  decoding="raw_heatmap_coordinates_orange; presence_from_saved_validation_threshold")
    save_json(output.with_suffix(".json"), result)
    return result
