"""CPU-only, unpublished ball-evidence diagnostics for video_000/clip_000.

The production phase opens no labels. Counterfactuals use only this dev clip's
observed ball labels, never person labels or annotation-derived production sides.
No threshold search, model inference or component-store writes are supported.
"""
from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf
from pipeline_stop_collection import load_completed  # type: ignore[import-not-found]

from src.tasks.ball_detection.inference.trajectory_gate import apply_trajectory_gate
from src.tasks.court_side.hypothesis import (
    CourtSideConfig,
    CourtSideUndecided,
    collect_side_evidence,
    distinct_observation_frames,
    judge_side_evidence,
)
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.definition import file_identity
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
from src.tennis_scene.pipeline.storage.codec import restore_type
from src.utils.geometry.multiview_consistency import score_multiview_points


def accept(uv: NDArray[np.float32], scores: NDArray[np.float32], valid: NDArray[np.bool_],
           settings: dict[str, Any]) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """The existing point gate, without producing an artifact or changing settings."""
    observed = valid & (scores >= settings['score_threshold'])
    gate = settings['trajectory_gate']
    if not gate['enabled']:
        raise ValueError('This diagnosis requires the existing enabled trajectory gate')
    kept, detail = apply_trajectory_gate(np.where(observed[:, None], uv, 0).astype(np.float32),
        observed, np.where(observed, scores, 0).astype(np.float32),
        **{k: v for k, v in gate.items() if k != 'enabled'})
    return kept, {'valid_top1': int(valid.sum()), 'above_score_threshold': int(observed.sum()),
                  'after_trajectory_gate': int(kept.sum()), 'gate': asdict(detail)}


def score_stream(name: str, uv: NDArray[np.float32], visible: NDArray[np.bool_], calibration: CourtCalibrationOutput,
                 source: ClipSource, config: CourtSideConfig, max_frames: int,
                 output: Path) -> dict[str, Any]:
    """Use the unchanged decision and decompose its frame mean without reweighting."""
    sample = sampled_frame_indices(source.num_frames, source.fps, max_frames=max_frames)
    config = replace(config, reprojection_px=config.reprojection_px * source.pixel_threshold_scale,
                     min_motion_px=config.min_motion_px * source.pixel_threshold_scale)
    cameras = tuple(v.camera for v in calibration.calibration.views)
    if tuple(c.camera_id for c in cameras) != source.camera_ids:
        raise ValueError('This diagnosis requires all three calibrated cameras in source order')
    sampled_uv, sampled_visible = uv[:, sample], visible[:, sample]
    evidence = collect_side_evidence(cameras, calibration.reference_camera, sampled_uv, sampled_visible, config)
    result: dict[str, Any] = {'name': name, 'frames': evidence.frames, 'pair_frames': evidence.pair_record(),
        'hypotheses': [asdict(h) for h in evidence.hypotheses],
        'margin': evidence.hypotheses[1].cost - evidence.hypotheses[0].cost,
        'observed_per_camera': visible.sum(1).tolist(), 'sampled_per_camera': sampled_visible.sum(1).tolist(),
        'sampled_frames': len(sample), 'thresholds': asdict(config)}
    try:
        decision = judge_side_evidence(evidence, config)
        result.update(decided=True, view_half_turns=decision.view_half_turns, reason=None)
    except CourtSideUndecided as error:
        result.update(decided=False, view_half_turns=None, reason=error.reason)
    keep = distinct_observation_frames(sampled_uv, sampled_visible, config.min_motion_px)
    chosen = keep & (sampled_visible.sum(0) >= 2)
    u, vis = sampled_uv[:, chosen], sampled_visible[:, chosen]
    point_scores = []
    for hypothesis in evidence.hypotheses:
        turned = tuple(c.half_turned(t) for c, t in zip(cameras, hypothesis.view_half_turns, strict=True))
        values = score_multiview_points(u, vis, turned, threshold_px=config.reprojection_px, bounds=config.bounds)
        np.testing.assert_allclose(values.cost.mean(), hypothesis.cost, atol=1e-14, rtol=0)
        np.testing.assert_allclose(values.support.mean(), hypothesis.support, atol=1e-14, rtol=0)
        point_scores.append(values)
    masks = (vis * np.array([1, 2, 4])[:, None]).sum(0)
    groups = {}
    for mask in sorted(np.unique(masks)):
        selected = masks == mask
        groups[str(int(mask))] = {'cameras': [c for i, c in enumerate(source.camera_ids) if int(mask) & (1 << i)],
            'frames': int(selected.sum()), 'costs': [float(s.cost[selected].mean()) for s in point_scores],
            'supports': [float(s.support[selected].mean()) for s in point_scores],
            'margin_contribution': float((point_scores[1].cost[selected] - point_scores[0].cost[selected]).sum() / evidence.frames)}
    result['view_mask_breakdown'] = groups
    with (output / f'{name}-frames.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame', 'view_mask', *[f'cost_{i}' for i in range(len(point_scores))],
                         *[f'support_{i}' for i in range(len(point_scores))]])
        for i, frame in enumerate(sample[chosen]):
            writer.writerow([int(frame), int(masks[i]), *[float(s.cost[i]) for s in point_scores],
                             *[int(s.support[i]) for s in point_scores]])
    np.savez_compressed(output / f'{name}-observations.npz', uv_px=uv, visible=visible,
                        sampled_frame_indices=sample, distinct_mask=keep, scored_mask=chosen)
    return result


def production(qualification: Path, output: Path) -> None:
    started = time.monotonic()
    index, loaded, descriptors = load_completed(qualification / 'store')
    source: ClipSource = restore_type(index['source'], ClipSource)
    if source.clip_id != 'video_000/clip_000' or source.camera_ids != ('cam0', 'cam1', 'cam2'):
        raise ValueError('Only the explicitly authorized dev clip may be opened')
    preflight = json.loads((qualification / 'preflight.json').read_text())
    config_path = Path(preflight['config']['path'])
    if file_identity(config_path) != preflight['config']:
        raise ValueError('Qualification config changed')
    settings = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(settings, dict):
        raise TypeError('Expected config mapping')
    side_config = CourtSideConfig(**settings['court_side'])
    balls = [loaded[f'ball_detection/{c}'] for c in source.camera_ids]
    gate_records = {}
    for camera, ball in zip(source.camera_ids, balls, strict=True):
        if ball.evidence is None:
            raise ValueError('Missing detector evidence')
        raw = ball.evidence
        valid, detail = accept(raw.candidate_uv_px[:, 0], raw.candidate_scores[:, 0], raw.candidate_valid[:, 0],
                               descriptors[f'ball_detection/{camera}']['identity']['settings']['config'])
        np.testing.assert_array_equal(valid, ball.observed)
        np.testing.assert_array_equal(np.where(valid[:, None], raw.candidate_uv_px[:, 0], 0), ball.uv_px)
        np.testing.assert_array_equal(np.where(valid, raw.candidate_scores[:, 0], 0), ball.confidence)
        gate_records[camera] = detail
    output.mkdir(parents=True, exist_ok=False)
    record = score_stream('production', np.stack([b.uv_px for b in balls]), np.stack([b.observed for b in balls]),
        loaded['court_calibration'], source, side_config, settings['frame_sampling']['max_frames'], output)
    original = json.loads((qualification / 'execute.json').read_text())['run']['error_diagnostics']
    for current, old in zip(record['hypotheses'], original['hypotheses'], strict=True):
        if list(current['view_half_turns']) != old['view_half_turns']:
            raise ValueError('Hypothesis order changed')
        np.testing.assert_array_equal([current[k] for k in ('cost', 'support', 'frames')],
                                      [old[k] for k in ('cost', 'support', 'frames')])
    if record['frames'] != original['frames'] or record['pair_frames'] != original['pair_frames']:
        raise ValueError('Side evidence differs from the queue failure')
    record.update(gates=gate_records, labels_opened=False, imported_nodes=[], production_receipt_exact=True,
                  inputs=[file_identity(config_path), file_identity(qualification / 'execute.json')],
                  wall_seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    write_json_atomic(output / 'production.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    production(args.qualification.resolve(), args.output.resolve())
