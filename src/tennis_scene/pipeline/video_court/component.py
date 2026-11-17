"""Adapt a pinned video artifact to the ordinary clip execution graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.tasks.court_detection.geometry.homography import court_template_xy
from src.tasks.court_detection.geometry.hybrid_homography import project_candidate
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    ComponentIO,
    SourceVideo,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.utils.geometry.keypoints import normalize_grid_keypoints

from .diagnostics import clip_report, drift_identity
from .store import VideoCourtArtifact


@dataclass(frozen=True)
class SharedCourtInput:
    video: SourceVideo
    clip_id: str
    camera: dict[str, Any]
    artifact_digest: str
    drift_report: dict[str, Any] | None = None
    drift_digest: str | None = None


@dataclass(frozen=True)
class SharedCourtAssembler:
    artifact: VideoCourtArtifact
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> SharedCourtInput:
        self.artifact.bind(context.source)
        if context.camera_id is None:
            raise ValueError('Shared court requires an explicit camera')
        camera = context.camera_id
        return SharedCourtInput(context.source.video(camera), context.source.clip_id,
                                self.artifact.cameras[camera], self.artifact.camera_digest(camera),
                                clip_report(self.artifact, camera, context.source.clip_id),
                                drift_identity(self.artifact).get('camera_digests', {}).get(camera))


class SharedCourtModule:
    io = ComponentIO('court_detection', SharedCourtInput, CourtKPResult, {}, 'court_observations', 3)

    def process(self, inputs: SharedCourtInput) -> CourtKPResult:
        camera, video = inputs.camera, inputs.video
        report = camera['clips'][inputs.clip_id] if inputs.drift_report is None else inputs.drift_report
        if camera['status'] != 'accepted' or report['status'] == 'failed':
            raise ReconstructionUnavailable('video_shared_court_failed', f'Shared court unavailable for {inputs.clip_id}/{video.camera_id}',
                                            diagnostics={'camera_id': video.camera_id, 'camera_status': camera['status'],
                                                         'artifact_digest': inputs.artifact_digest,
                                                         'drift_digest': inputs.drift_digest,
                                                         'camera_reasons': camera['diagnostics'].get('reasons', []), 'clip': report})
        matrix = np.asarray(camera['matrix'], float)
        points = project_candidate(matrix, court_template_xy().astype(float))
        if points is None:
            raise ValueError('Invalid accepted video court projection')
        visible = ((points >= 0) & (points <= np.array([video.width - 1, video.height - 1]))).all(axis=1)
        shared = {'dataset_id': camera['dataset_id'], 'video_id': camera['video_id'], 'camera_id': video.camera_id,
                  'artifact_digest': inputs.artifact_digest,
                  'drift_digest': inputs.drift_digest,
                  'selected_initializer': camera['diagnostics'].get('selected_initializer'),
                  'fit_frames': camera['diagnostics'].get('fit_frames', [])}
        diagnostic = {'schema': 'court_kp_quality_v3', 'output_keypoint_contract': 'camera_view_v2',
                      'temporal_policy': 'static_video_shared', 'observed_frame_indices': [],
                      'source_frame_count': video.num_frames, 'shared_court': shared, 'drift': report,
                      'cameras': [{'camera_id': video.camera_id, 'frames': [{'frame_index': 0, 'status': 'ok',
                                   'fit_source': 'video_shared_rgb', 'homography_court_metres_to_image_pixels': matrix.tolist(),
                                   'quality': {'accepted': True, 'source': inputs.artifact_digest}}]}]}
        output = CourtKPResult(normalize_grid_keypoints(points, video.width, video.height)[None, None].astype(np.float32),
                               visible[None, None].astype(np.float32), np.array([0], np.int32), diagnostic)
        valid, errors = output.validate()
        if not valid:
            raise ValueError(f'Invalid shared court output: {errors}')
        return output
