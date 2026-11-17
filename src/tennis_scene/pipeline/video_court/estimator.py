"""CPU multi-start RGB fit; the same algorithm is used by every caller."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from src.tasks.court_detection.geometry.homography import court_template_xy
from src.tasks.court_detection.geometry.hybrid_homography import project_candidate
from src.tasks.court_detection.geometry.video_shared.candidates import (
    CourtSeed,
    read_frozen_candidates,
)
from src.tasks.court_detection.geometry.video_shared.drift import inspect_drift
from src.tasks.court_detection.geometry.video_shared.fit import (
    SharedPaintConfig,
    choose_common_solution,
    fit_common_court,
)
from src.tasks.court_detection.geometry.video_shared.observations import (
    FramePaint,
    clustered_initializers,
    measure_frame,
)
from src.utils.video import read_video_frame

PROFILE = 'temporal_rgb_candidate_consensus_v1'
LOGGER = logging.getLogger(__name__)


def sample_indices(count: int, numerators: list[int], denominator: int) -> tuple[int, ...]:
    if (type(denominator) is not int or denominator <= 0 or not numerators
            or any(type(n) is not int or not 0 <= n <= denominator for n in numerators)):
        raise ValueError('Invalid shared-court frame selection')
    indices = tuple(n * (count - 1) // denominator for n in numerators)
    if len(set(indices)) != len(indices):
        raise ValueError('Shared-court frame selection repeats a source frame')
    return indices


def estimate_camera(*, camera: str, size_wh: tuple[int, int], sources: dict[str, dict[str, Any]],
                    paths: dict[str, Path], seeds_path: Path, seeds_sha256: str, seed_metadata: dict[str, Any],
                    dataset_id: str, video_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """No model, reference, manual points, masks, or alternate camera are read."""
    config = SharedPaintConfig()
    clip_ids = tuple(sorted(sources))
    seed_clips = tuple(request['seed_clip_ids'] or clip_ids)
    fit_clips = tuple(request['fit_clip_ids'] or clip_ids)
    seeds = read_frozen_candidates(seeds_path, sha256=seeds_sha256, dataset_id=dataset_id, video_id=video_id,
                                   camera_id=camera, size_wh=size_wh, clip_ids=seed_clips)
    initializers, initial_diagnostics = clustered_initializers(seeds, config)
    LOGGER.info('Shared court %s: %d frozen candidates, %d initializers', camera, len(seeds), len(initializers))
    decoded_hashes: dict[str, str] = {}

    @lru_cache(maxsize=12)
    def image(clip: str, frame: int) -> np.ndarray:
        packet = read_video_frame(paths[clip], frame)
        if tuple(packet.original_size) != size_wh:
            raise ValueError('Shared court RGB size disagrees with source')
        rgb = cast(np.ndarray, cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB))
        key = f'{clip}/{frame}'
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        if key in decoded_hashes and decoded_hashes[key] != digest:
            raise ValueError('Shared court source RGB changed between reads')
        decoded_hashes[key] = digest
        return rgb

    # Bind every frozen neural initializer to the actual source bytes/RGB.
    frame_provenance = seed_metadata.get('source_frames')
    if not isinstance(frame_provenance, dict):
        raise ValueError('Frozen seeds require source_frames provenance')
    for clip, frame in sorted({(seed.clip_id, seed.frame_index) for seed in seeds}):
        record = frame_provenance.get(f'{clip}/{frame}')
        if (record is None or record['media_sha256'] != sources[clip]['sha256']
                or record['rgb_sha256'] != hashlib.sha256(image(clip, frame).tobytes()).hexdigest()):
            raise ValueError('Frozen model initializer disagrees with source RGB')
    bootstrap_clips = request['bootstrap_clip_ids']
    if bootstrap_clips is None:
        bootstrap_clips = sorted({fit_clips[i] for i in (0, len(fit_clips) // 2, len(fit_clips) - 1)})
    fit_frames = [(clip, frame) for clip in fit_clips for frame in sample_indices(sources[clip]['num_frames'], request['fit_numerators'], request['denominator'])]
    bootstrap = [(clip, next(frame for c, frame in fit_frames if c == clip)) for clip in bootstrap_clips]

    def measure(task: tuple[CourtSeed, str, int]) -> FramePaint:
        seed, clip, frame = task
        return measure_frame(image(clip, frame), seed.matrix, clip_id=clip, frame_index=frame, config=config)

    proposals = []
    bootstrap_reports = []
    with ThreadPoolExecutor(max_workers=request['workers']) as executor:
        tasks = [(seed, clip, frame) for seed in initializers for clip, frame in bootstrap]
        for task, measured in zip(tasks, executor.map(measure, tasks), strict=True):
            seed, clip, frame = task
            local = fit_common_court(seed.matrix, measured.groups, size_wh, replace(config, min_clips_per_line=1))
            bootstrap_reports.append({'seed': seed.candidate_id, 'clip_id': clip, 'frame_index': frame,
                                      'measurement_reason': measured.diagnostic['reason'], 'reasons': list(local.reasons)})
            if local.matrix is not None:
                proposals.append(CourtSeed(f'{seed.candidate_id}/measured/{clip}/{frame}', clip, frame, local.matrix, seed.model_accepted))
    refined, refinement_diagnostics = clustered_initializers(tuple(proposals), config)
    LOGGER.info('Shared court %s: %d RGB initializers, %d fit frames each', camera, len(refined), len(fit_frames))
    solutions = []
    solution_reports = []
    with ThreadPoolExecutor(max_workers=request['workers']) as executor:
        for seed in refined:
            tasks = [(seed, clip, frame) for clip, frame in fit_frames]
            measured_frames = list(executor.map(measure, tasks))
            groups = tuple(group for measured in measured_frames for group in measured.groups)
            fitted = fit_common_court(seed.matrix, groups, size_wh, config)
            solutions.append(fitted)
            LOGGER.info('Shared court %s: candidate %d/%d reasons=%s', camera, len(solutions), len(refined), fitted.reasons)
            solution_reports.append({'initializer': seed.candidate_id, 'reasons': list(fitted.reasons),
                                     'matrix': None if fitted.matrix is None else fitted.matrix.tolist(),
                                     'fitted_matrix': None if fitted.fitted_matrix is None else fitted.fitted_matrix.tolist(),
                                     'diagnostics': fitted.diagnostic,
                                     'frames': [{'clip_id': clip, 'frame_index': frame, 'reason': measured.diagnostic['reason']}
                                                for (_, clip, frame), measured in zip(tasks, measured_frames, strict=True)]})
    selected = choose_common_solution(tuple(solutions), config)
    selected_index = next((i for i, solution in enumerate(solutions) if selected.matrix is solution.matrix and selected.matrix is not None), None)
    used_frames = sorted({(row['clip_id'], row['frame_index']) for row in selected.diagnostic.get('group_residuals', []) if row['used']})
    diagnostics: dict[str, Any] = {'profile': PROFILE, 'settings': asdict(config), 'reasons': list(selected.reasons),
                                   'initializers': initial_diagnostics, 'bootstrap': bootstrap_reports,
                                   'refined_initializers': refinement_diagnostics, 'solutions': solution_reports,
                                   'selected_initializer': None if selected_index is None else refined[selected_index].candidate_id,
                                   'quality': selected.diagnostic, 'fit_frames': [list(f) for f in used_frames],
                                   'requested_fit_frames': [list(f) for f in fit_frames],
                                   'seed_model_accepted_clip_count': len({s.clip_id for s in seeds if s.model_accepted})}
    clips = {}
    for clip in clip_ids:
        frames = []
        for frame in sample_indices(sources[clip]['num_frames'], request['diagnostic_numerators'], request['denominator']):
            if selected.matrix is None:
                report = {'status': 'failed', 'reason': 'shared_court_unavailable', 'mean_displacement_px': None, 'max_displacement_px': None}
            else:
                measured = measure_frame(image(clip, frame), selected.matrix, clip_id=clip, frame_index=frame, config=config)
                report = inspect_drift(selected.matrix, measured.groups, size_wh, config)
                report['measurement'] = measured.diagnostic
            frames.append({'frame_index': frame, **report})
        status = 'failed' if any(f['status'] == 'failed' for f in frames) else 'warning' if any(f['status'] == 'warning' for f in frames) else 'ok'
        differences = []
        if selected.matrix is not None:
            shared_points = project_candidate(selected.matrix, court_template_xy().astype(float))
            assert shared_points is not None
            visible = ((shared_points >= 0) & (shared_points <= np.asarray(size_wh) - 1)).all(axis=1)
            for seed in seeds:
                if seed.clip_id != clip:
                    continue
                points = project_candidate(seed.matrix, court_template_xy().astype(float))
                if points is not None:
                    distance = np.linalg.norm(points - shared_points, axis=1)
                    differences.append({'candidate_id': seed.candidate_id, 'model_accepted': seed.model_accepted,
                                        'mean_all_px': float(distance.mean()), 'max_all_px': float(distance.max()),
                                        'mean_visible_px': float(distance[visible].mean()) if visible.any() else None,
                                        'max_visible_px': float(distance[visible].max()) if visible.any() else None})
        clips[clip] = {'status': status, 'frames': frames, 'model_disagreement': differences,
                       'model_disagreement_status': 'measured' if differences else 'no_initializer_for_clip_or_no_common_h'}
    diagnostics['clip_status_counts'] = dict(Counter(value['status'] for value in clips.values()))
    diagnostics['decoded_rgb_sha256'] = dict(sorted(decoded_hashes.items()))
    LOGGER.info('Shared court %s: common reasons=%s, clips=%s', camera, selected.reasons, diagnostics['clip_status_counts'])
    return {'matrix': None if selected.matrix is None else selected.matrix.tolist(),
            'status': 'failed' if selected.matrix is None else 'accepted', 'diagnostics': diagnostics, 'clips': clips}
