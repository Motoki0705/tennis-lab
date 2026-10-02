"""Diagnostic-only ball substitution on the authorized clip_000 dev labels.

Reads exactly three e9 cache NPZs and three ball annotation files. No component
publication, model inference, parameter selection, person labels or other clips.
"""
from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from court_side_clip000_diagnosis import (  # type: ignore[import-not-found]
    accept,
    score_stream,
)
from numpy.typing import NDArray
from omegaconf import OmegaConf
from pipeline_stop_collection import load_completed  # type: ignore[import-not-found]

from src.tasks.court_side.hypothesis import CourtSideConfig
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.definition import file_identity
from src.tennis_scene.pipeline.imports.ball_annotations import convert_ball_annotation
from src.tennis_scene.pipeline.storage.codec import restore_type
from src.utils.checksum import dual_sha256


def source_top1(arrays: dict[str, Any], width: int, height: int,
                frames: int) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    """Cache v1 coordinates are source pixels divided by (W-1,H-1), not W,H."""
    np.testing.assert_array_equal(arrays['frame_index'], np.arange(frames))
    coords, scores, valid = (arrays[k][:, 0] for k in ('candidate_coords', 'candidate_scores', 'candidate_valid'))
    if coords.shape != (frames, 2) or scores.shape != (frames,) or valid.dtype != np.bool_ \
            or not np.isfinite(coords).all() or not np.isfinite(scores).all() \
            or (coords < 0).any() or (coords > 1).any():
        raise ValueError('Invalid complete cache top-1 timeline')
    np.testing.assert_array_equal(coords, arrays['argmax_uv'])
    np.testing.assert_array_equal(scores, arrays['argmax_score'])
    return (coords * np.array([width - 1, height - 1], np.float32)).astype(np.float32), scores, valid


def agreement(uv: NDArray[np.float32], visible: NDArray[np.bool_], label_uv: NDArray[np.float32],
              label_visible: NDArray[np.bool_], radius: float) -> dict[str, Any]:
    """Missing/estimated labels are excluded, not converted to negative examples."""
    both = visible & label_visible
    error = np.linalg.norm(uv[both] - label_uv[both], axis=1)
    matched = int((error <= radius).sum())
    return {'predicted': int(visible.sum()), 'observed_labels': int(label_visible.sum()),
            'both': int(both.sum()), 'matched': matched, 'far': int(both.sum()) - matched,
            'predicted_without_observed_label': int((visible & ~label_visible).sum()),
            'recall_at_20px': matched / int(label_visible.sum()) if label_visible.any() else None,
            'precision_on_observed_labels_at_20px': matched / len(error) if len(error) else None,
            'median_error_px': float(np.median(error)) if len(error) else None,
            'p90_error_px': float(np.quantile(error, .9)) if len(error) else None}


def compare(qualification: Path, cache: Path, output: Path) -> None:
    started = time.monotonic()
    index, loaded, descriptors = load_completed(qualification / 'store')
    source: ClipSource = restore_type(index['source'], ClipSource)
    if source.clip_id != 'video_000/clip_000' or source.camera_ids != ('cam0', 'cam1', 'cam2'):
        raise ValueError('Only video_000/clip_000 is authorized for label diagnosis')
    preflight = json.loads((qualification / 'preflight.json').read_text())
    config_path = Path(preflight['config']['path'])
    if file_identity(config_path) != preflight['config']:
        raise ValueError('Qualification config changed')
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise TypeError('Expected config mapping')
    side_config = CourtSideConfig(**config['court_side'])
    manifest = json.loads((cache / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['schema'] != 'ball_refiner_detector_evidence.v1' \
            or manifest['coordinate_system'] != 'source_xy_div_size_minus_one':
        raise ValueError('Expected complete e9 evidence cache v1')
    if manifest['detector']['sha256'] != '37f4c59aead00062829280ad591b3874886978891104a290f704a2c33c3c1b36':
        raise ValueError('This comparison requires the pinned #935 epoch9 detector')
    labels, e9_uv, e9_scores, e9_valid, e9_gate = [], [], [], [], []
    entries, gate_records, annotations = [], {}, []
    inputs = [file_identity(config_path), file_identity(cache / 'manifest.json'), file_identity(qualification / 'store/scene.json')]
    for video in source.videos:
        camera = video.camera_id
        path = video.path.parent.parent / 'outsource' / f'{camera}_annotations.json'
        # Conversion only: never call import_ball_annotations/publish_import.
        label, provenance = convert_ball_annotation(path, video)
        labels.append(label)
        annotations.append(provenance)
        inputs.append(file_identity(path))
        matching = [e for e in manifest['clips'] if e['clip']['clip_id'] == f'meiji/{source.clip_id}/{camera}']
        if len(matching) != 1:
            raise ValueError('Expected one exact clip/camera cache entry')
        entry = matching[0]
        clip = entry['clip']
        if (clip['media_sha256'], clip['frame_count'], clip['source_width'], clip['source_height'], clip['camera_id']) \
                != (video.sha256, video.num_frames, video.width, video.height, camera) \
                or abs(float(Fraction(clip['fps'])) - video.fps) > 1e-5:
            raise ValueError('Cache does not identify the qualification source video')
        npz = (cache / entry['file']).resolve()
        if not npz.is_relative_to(cache) or dual_sha256(npz) != entry['sha256']:
            raise ValueError('Cache NPZ path/checksum mismatch')
        inputs.append(file_identity(npz))
        with np.load(npz, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        uv, score, valid = source_top1(arrays, video.width, video.height, video.num_frames)
        seconds = arrays['pts'] * float(Fraction(clip['time_base']))
        np.testing.assert_allclose(seconds, np.arange(video.num_frames) / video.fps, atol=1e-6, rtol=0)
        np.testing.assert_allclose(arrays['timestamps_seconds'], seconds, atol=2e-6, rtol=0)
        np.testing.assert_array_equal(arrays['window_start'] + arrays['time_index'], arrays['frame_index'])
        kept, gate = accept(uv, score, valid, descriptors[f'ball_detection/{camera}']['identity']['settings']['config'])
        e9_uv.append(uv)
        e9_scores.append(score)
        e9_valid.append(valid)
        e9_gate.append(kept)
        entries.append(entry)
        gate_records[camera] = gate
    production = [loaded[f'ball_detection/{c}'] for c in source.camera_ids]
    label_uv, label_visible = np.stack([b.uv_px for b in labels]), np.stack([b.observed for b in labels])
    conditions = {
        'annotation_observed': (label_uv, label_visible),
        'e9_top1_ungated': (np.stack(e9_uv), np.stack(e9_valid)),
        'e9_top1_production_gate': (np.stack(e9_uv), np.stack(e9_gate)),
        'production': (np.stack([b.uv_px for b in production]), np.stack([b.observed for b in production])),
    }
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    with (output / 'quality-all-frames.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['condition', 'camera', 'frame', 'observed', 'label_point_kind', 'error_px_observed_label_only'])
        for name, (uv, visible) in conditions.items():
            record = score_stream(name, uv, visible, loaded['court_calibration'], source,
                                  side_config, config['frame_sampling']['max_frames'], output)
            record['quality'] = {c: agreement(uv[v], visible[v], label_uv[v], label_visible[v], side_config.reprojection_px)
                                 for v, c in enumerate(source.camera_ids)}
            errors = np.linalg.norm(uv - label_uv, axis=2)
            with np.load(output / f'{name}-observations.npz') as observation:
                frames = observation['sampled_frame_indices'][observation['scored_mask']]
            comparable = (~visible[:, frames] | label_visible[:, frames]).all(0)
            correct = (~visible[:, frames] | (errors[:, frames] <= side_config.reprojection_px)).all(0)
            record['scored_frame_quality'] = {'frames': len(frames), 'all_observing_views_labeled': int(comparable.sum()),
                'all_observing_views_within20px': int((comparable & correct).sum()),
                'at_least_one_far_view': int((comparable & ~correct).sum()),
                'far_by_camera': {c: int((visible[v, frames] & label_visible[v, frames] & (errors[v, frames] > 20)).sum())
                                  for v, c in enumerate(source.camera_ids)}}
            for v, camera in enumerate(source.camera_ids):
                for frame in range(source.num_frames):
                    err = float(errors[v, frame]) if visible[v, frame] and label_visible[v, frame] else None
                    writer.writerow([name, camera, frame, int(visible[v, frame]), int(labels[v].point_kind[frame]), err])
            records[name] = record
    write_json_atomic(output / 'counterfactuals.json', {'conditions': records, 'e9_gates': gate_records,
        'e9_manifest_excerpt': {'detector': manifest['detector'], 'coordinate_system': manifest['coordinate_system'], 'clips': entries},
        'annotation_provenance': annotations, 'inputs': inputs,
        'labels_opened': 'video_000/clip_000 observed ball only for diagnosis; estimated points excluded',
        'person_labels_opened': False, 'imported_nodes': [], 'defaults_changed': False,
        'wall_seconds': time.monotonic() - started, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--e9-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.qualification.resolve(), args.e9_cache.resolve(), args.output.resolve())
