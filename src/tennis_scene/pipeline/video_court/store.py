"""Content-addressed video/camera court documents with atomic publication."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.video_artifact import (
    load_document,
    publish_document,
    store_lock,
    validate_digest,
)

from .estimator import PROFILE


def validate_camera(document: dict[str, Any]) -> None:
    if set(document) != {'schema', 'dataset_id', 'video_id', 'camera_id', 'size_wh', 'identity',
                         'sources', 'matrix', 'status', 'diagnostics', 'clips'} or document['schema'] != 'video_court_camera_v1':
        raise ValueError('Invalid video court camera schema')
    if document['status'] not in {'accepted', 'failed'}:
        raise ValueError('Invalid shared court status')
    if document['diagnostics'].get('profile') != PROFILE or document['identity'].get('profile') != PROFILE:
        raise ValueError('Unknown shared court estimator profile')
    if any(document['identity'].get(key) != document[key] for key in ('dataset_id', 'video_id', 'camera_id', 'size_wh', 'sources')):
        raise ValueError('Shared court producer identity disagrees with payload')
    matrix = None if document['matrix'] is None else np.asarray(document['matrix'], float)
    if ((document['status'] == 'accepted') != (matrix is not None)
            or (matrix is not None and (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12))):
        raise ValueError('Shared court status and homography disagree')
    if set(document['sources']) != set(document['clips']) or not document['sources']:
        raise ValueError('Shared court must diagnose every declared clip')
    size = document['size_wh']
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 2 for v in size):
        raise ValueError('Invalid shared court image size')
    for clip, source in document['sources'].items():
        if clip.split('/')[0] != document['video_id']:
            raise ValueError('Foreign video in shared court source')
        validate_digest(source['sha256'])
        if type(source['num_frames']) is not int or source['num_frames'] < 1 or not np.isfinite(source['fps']) or source['fps'] <= 0:
            raise ValueError('Invalid shared court source timeline')
        report = document['clips'][clip]
        if report['status'] not in {'ok', 'warning', 'failed'} or not report['frames']:
            raise ValueError('Every clip requires explicit drift inspection')
        for frame in report['frames']:
            if not 0 <= frame['frame_index'] < source['num_frames'] or frame['status'] not in {'ok', 'warning', 'failed'}:
                raise ValueError('Invalid shared court drift frame')
        expected = 'failed' if any(f['status'] == 'failed' for f in report['frames']) else 'warning' if any(f['status'] == 'warning' for f in report['frames']) else 'ok'
        if report['status'] != expected or (document['status'] == 'failed' and expected != 'failed'):
            raise ValueError('Clip drift status disagrees with its evidence')


@dataclass(frozen=True)
class VideoCourtArtifact:
    root: Path
    digest: str
    manifest: dict[str, Any]
    cameras: dict[str, dict[str, Any]]

    def verify(self) -> None:
        if load_document(self.root, self.digest) != self.manifest:
            raise ValueError('Video court manifest changed in memory')
        for camera, digest in self.manifest['cameras'].items():
            value = load_document(self.root, digest)
            if value != self.cameras[camera]:
                raise ValueError('Video court camera changed in memory')
            validate_camera(value)

    def bind(self, source: ClipSource) -> None:
        self.verify()
        self.check_source(source)

    def check_source(self, source: ClipSource) -> None:
        """Check membership against this snapshot after the caller has verified its objects."""
        if source.clip_id.split('/')[0] != self.manifest['video_id']:
            raise ValueError('Clip belongs to another shared-court video')
        for video in source.videos:
            if video.camera_id not in self.cameras:
                raise ValueError('Shared court camera is missing')
            camera = self.cameras[video.camera_id]
            record = camera['sources'].get(source.clip_id)
            if (record is None or record['sha256'] != video.sha256 or record['num_frames'] != video.num_frames
                    or abs(record['fps'] - video.fps) > 1e-6 or camera['size_wh'] != [video.width, video.height]):
                raise ValueError('Shared court source membership/content/timeline mismatch')

    def camera_digest(self, camera: str) -> str:
        self.verify()
        return str(self.manifest['cameras'][camera])


class VideoCourtStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def lock(self) -> AbstractContextManager[None]:
        return store_lock(self.root, '.video-court.lock')

    def load(self, digest: str) -> VideoCourtArtifact:
        manifest = load_document(self.root, digest)
        if set(manifest) != {'schema', 'dataset_id', 'video_id', 'cameras'} or manifest['schema'] != 'video_court_bundle_v1' or not manifest['cameras']:
            raise ValueError('Invalid video court manifest')
        cameras = {camera: load_document(self.root, ref) for camera, ref in manifest['cameras'].items()}
        for camera, value in cameras.items():
            validate_camera(value)
            if (value['camera_id'], value['dataset_id'], value['video_id']) != (camera, manifest['dataset_id'], manifest['video_id']):
                raise ValueError('Shared court camera identity mismatch')
        return VideoCourtArtifact(self.root, digest, manifest, cameras)

    def publish(self, cameras: dict[str, dict[str, Any]], *, repair: bool = False) -> VideoCourtArtifact:
        if not cameras:
            raise ValueError('Cannot publish an empty shared court')
        for value in cameras.values():
            validate_camera(value)
        first = next(iter(cameras.values()))
        references = {camera: publish_document(self.root, document, repair=repair) for camera, document in sorted(cameras.items())}
        manifest = {'schema': 'video_court_bundle_v1', 'dataset_id': first['dataset_id'], 'video_id': first['video_id'], 'cameras': references}
        return self.load(publish_document(self.root, manifest, repair=repair))

    def cached_camera(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        path = self.root / 'requests' / f'{document_digest(identity)}.json'
        if not path.exists():
            return None
        ref = json.loads(path.read_text())
        value = load_document(self.root, ref['camera_digest'])
        validate_camera(value)
        if value['identity'] != identity:
            raise ValueError('Video court request identity mismatch')
        return value

    def remember(self, document: dict[str, Any], *, repair: bool = False) -> None:
        validate_camera(document)
        digest = publish_document(self.root, document, repair=repair)
        path = self.root / 'requests' / f'{document_digest(document["identity"])}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, {'camera_digest': digest})
