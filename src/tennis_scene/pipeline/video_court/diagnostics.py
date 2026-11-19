"""Versioned drift evidence attached to an immutable shared-H artifact.

The two content addresses are separate: changing diagnostics invalidates clip
consumers without rewriting or re-estimating the shared camera geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.court_detection.geometry.video_drift import DriftConfig, summarize_clip
from src.tennis_scene.pipeline.video_artifact import (
    load_document,
    publish_document,
    validate_digest,
)

from .store import VideoCourtArtifact, VideoCourtStore


@dataclass(frozen=True)
class DriftReference:
    directory: Path
    digest: str

    def __post_init__(self) -> None:
        validate_digest(self.digest)


def _validate_camera(document: dict[str, Any], base: VideoCourtArtifact, camera: str) -> None:
    if (set(document) != {'schema', 'base_camera_digest', 'camera_id', 'protocol', 'clips'}
            or document['schema'] != 'video_court_drift_camera_v1' or document['camera_id'] != camera
            or document['base_camera_digest'] != base.manifest['cameras'][camera]):
        raise ValueError('Drift camera identity/schema does not match shared H')
    profile = DriftConfig(**document['protocol']['settings'])
    sources = base.cameras[camera]['sources']
    if set(document['clips']) != set(sources):
        raise ValueError('Drift artifact must inspect every shared-court clip')
    for clip, report in document['clips'].items():
        if summarize_clip(report['frames']) != report:
            raise ValueError('Drift clip does not preserve worst-frame evidence')
        for frame in report['frames']:
            if (type(frame['frame_index']) is not int or not 0 <= frame['frame_index'] < sources[clip]['num_frames']
                    or frame['schema'] != profile.version or frame['temporal_pooling'] is not False
                    or frame['status'] not in {'ok', 'warning', 'failed'}
                    or (frame['reason'] == 'drift_unobservable' and frame['status'] != 'failed')):
                raise ValueError('Invalid drift frame evidence')
            validate_digest(frame['rgb_sha256'])
            reasons: dict[str, set[str | None]] = {'ok': {None}, 'warning': {'camera_moved_possible'},
                                                  'failed': {'drift_unobservable', 'camera_drift_exceeds_tolerance'}}
            if frame['reason'] not in reasons[frame['status']]:
                raise ValueError('Drift status disagrees with failure/warning reason')
            if frame['reason'] != 'drift_unobservable' and any(
                    not isinstance(frame.get(key), (int, float)) or not np.isfinite(frame[key]) or frame[key] < 0
                    for key in ('mean_displacement_px', 'max_displacement_px', 'max_visible_projection_std_px')):
                raise ValueError('Observable drift requires finite displacement and uncertainty')


@dataclass(frozen=True)
class DiagnosedVideoCourt(VideoCourtArtifact):
    drift: DriftReference
    drift_manifest: dict[str, Any]
    drift_cameras: dict[str, dict[str, Any]]

    def verify(self) -> None:
        super().verify()
        if load_document(self.drift.directory, self.drift.digest) != self.drift_manifest:
            raise ValueError('Drift manifest changed in memory')
        if (set(self.drift_manifest) != {'schema', 'base_bundle_digest', 'cameras'}
                or self.drift_manifest['schema'] != 'video_court_drift_bundle_v1'
                or self.drift_manifest['base_bundle_digest'] != self.digest
                or set(self.drift_manifest['cameras']) != set(self.manifest['cameras'])):
            raise ValueError('Drift bundle does not match shared H')
        for camera, ref in self.drift_manifest['cameras'].items():
            document = load_document(self.drift.directory, ref)
            if document != self.drift_cameras[camera]:
                raise ValueError('Drift camera changed in memory')
            _validate_camera(document, self, camera)


def load_video_drift(base: VideoCourtArtifact, reference: DriftReference) -> DiagnosedVideoCourt:
    manifest = load_document(reference.directory, reference.digest)
    cameras = {camera: load_document(reference.directory, ref) for camera, ref in manifest['cameras'].items()}
    result = DiagnosedVideoCourt(base.root, base.digest, base.manifest, base.cameras, reference, manifest, cameras)
    result.verify()
    return result


def publish_video_drift(base: VideoCourtArtifact, directory: Path, *, protocol: dict[str, Any],
                        reports: dict[str, dict[str, dict[str, Any]]]) -> DiagnosedVideoCourt:
    base.verify()
    if set(reports) != set(base.cameras):
        raise ValueError('Drift publication requires all shared cameras')
    if directory.resolve() == base.root.resolve():
        raise ValueError('Drift evidence must use a separate store from common H')
    references = {}
    with VideoCourtStore(directory).lock():
        for camera, clips in sorted(reports.items()):
            document = {'schema': 'video_court_drift_camera_v1', 'base_camera_digest': base.manifest['cameras'][camera],
                        'camera_id': camera, 'protocol': protocol, 'clips': clips}
            _validate_camera(document, base, camera)
            references[camera] = publish_document(directory, document)
        base.verify()
        digest = publish_document(directory, {'schema': 'video_court_drift_bundle_v1', 'base_bundle_digest': base.digest, 'cameras': references})
    return load_video_drift(base, DriftReference(directory, digest))


def clip_report(artifact: VideoCourtArtifact, camera: str, clip: str) -> dict[str, Any]:
    # An explicitly loaded v1 artifact keeps its original embedded diagnostic.
    # When a separate reference was requested, load/verify errors propagate.
    documents = artifact.drift_cameras if isinstance(artifact, DiagnosedVideoCourt) else artifact.cameras
    return dict(documents[camera]['clips'][clip])


def drift_identity(artifact: VideoCourtArtifact) -> dict[str, Any]:
    if isinstance(artifact, DiagnosedVideoCourt):
        return {'directory': str(artifact.drift.directory), 'artifact_digest': artifact.drift.digest,
                'camera_digests': artifact.drift_manifest['cameras']}
    return {}
