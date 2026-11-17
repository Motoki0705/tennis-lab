"""One explicit build/load boundary for all clip entry points."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tennis_scene.generate_dataset.manifest import (
    load_dataset_manifest,
    validate_id_component,
)
from src.tennis_scene.pipeline.clip_inputs import (
    InputSnapshot,
    prepare_clip_source,
    resolve_dataset_clip,
    select_dataset_clips,
)
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathResolver, PathRole

from .estimator import PROFILE, estimate_camera, sample_indices
from .store import VideoCourtArtifact, VideoCourtStore, _digest


@dataclass(frozen=True)
class VideoCourtConfig:
    action: str
    directory: Path
    dataset_id: str
    video_id: str
    request: Path | None
    request_sha256: str | None
    artifact_digest: str | None
    rebuild: bool

    def __post_init__(self) -> None:
        if self.action not in {'build', 'load'} or not self.dataset_id or not self.video_id:
            raise ValueError('video_shared requires explicit build/load and dataset/video IDs')
        validate_id_component(self.dataset_id, field_name='dataset_id')
        validate_id_component(self.video_id, field_name='video_id')
        for value in (self.request_sha256, self.artifact_digest):
            if value is not None:
                _digest(value)
        if self.action == 'build':
            if self.request is None or self.request_sha256 is None or self.artifact_digest is not None:
                raise ValueError('Shared court build requires a pinned request, without a load digest')
        elif self.artifact_digest is None or self.request is not None or self.request_sha256 is not None or self.rebuild:
            raise ValueError('Shared court load requires only a pinned artifact, and forbids rebuild')

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], resolver: PathResolver) -> VideoCourtConfig:
        if any(value[k] is None for k in ('action', 'directory', 'dataset_id', 'video_id')):
            raise ValueError('video_shared acquisition and scope must be explicit')
        return cls(value['action'], resolver.resolve(PathRole.ARTIFACT, value['directory']), value['dataset_id'], value['video_id'],
                   None if value['request'] is None else resolver.resolve(PathRole.ARTIFACT, value['request']),
                   value['request_sha256'], value['artifact_digest'], value['rebuild'])


def _selection(value: Any, available: tuple[str, ...], name: str) -> tuple[str, ...]:
    selected = available if value is None else tuple(value)
    if not selected or len(set(selected)) != len(selected) or any(c not in available for c in selected):
        raise ValueError(f'Invalid shared court {name}: empty/duplicate/foreign/unknown clip')
    return tuple(sorted(selected))


def prepare_video_court(config: VideoCourtConfig, resolver: PathResolver, *, code_identity: str) -> VideoCourtArtifact:
    """Load never builds. Build uses frozen model candidates plus actual RGB.

    Locks and identity checks cover publication; a camera with no solution is
    persisted as failed, not replaced by another camera or a clip detector.
    """
    store = VideoCourtStore(config.directory)
    if config.action == 'load':
        assert config.artifact_digest is not None
        artifact = store.load(config.artifact_digest)
        if (artifact.manifest['dataset_id'], artifact.manifest['video_id']) != (config.dataset_id, config.video_id):
            raise ValueError('Loaded shared court belongs to another dataset/video')
        return artifact
    assert config.request is not None and config.request_sha256 is not None
    if dual_sha256(config.request) != config.request_sha256:
        raise ValueError('Shared court build request changed')
    request = json.loads(config.request.read_text())
    required = {'schema', 'profile', 'dataset_directory', 'video_id', 'candidate_manifests', 'seed_clip_ids', 'fit_clip_ids',
                'bootstrap_clip_ids', 'fit_numerators', 'diagnostic_numerators', 'denominator', 'workers'}
    if (set(request) != required or request['schema'] != 'video_court_build_v1' or request['profile'] != PROFILE
            or request['video_id'] != config.video_id or type(request['workers']) is not int or request['workers'] < 1):
        raise ValueError('Invalid shared court build request/profile')
    dataset = resolver.resolve(PathRole.DATA, request['dataset_directory'])
    manifest = load_dataset_manifest(dataset)
    if manifest.dataset_id != config.dataset_id:
        raise ValueError('Shared court dataset identity mismatch')
    records = tuple(r for r in select_dataset_clips(dataset) if r.video_id == config.video_id)
    clip_ids = tuple(r.clip_id for r in records)
    if not records:
        raise ValueError('Shared court video contains no clips')
    for key in ('seed_clip_ids', 'fit_clip_ids'):
        request[key] = list(_selection(request[key], clip_ids, key))
    if request['bootstrap_clip_ids'] is not None:
        request['bootstrap_clip_ids'] = list(_selection(request['bootstrap_clip_ids'], tuple(request['fit_clip_ids']), 'bootstrap_clip_ids'))
    source_by_camera: dict[str, dict[str, Any]] = {}
    paths_by_camera: dict[str, dict[str, Path]] = {}
    snapshots = [InputSnapshot(((config.request, config.request_sha256), (dataset / 'dataset.json', dual_sha256(dataset / 'dataset.json'))))]
    size = None
    camera_ids = None
    for record in records:
        bound = resolve_dataset_clip(dataset, record)
        media = prepare_clip_source(bound.video_paths, bound.camera_ids, clip_id=record.clip_id)
        if camera_ids is None:
            camera_ids, size = media.source.camera_ids, media.source.size
        if media.source.camera_ids != camera_ids or media.source.size != size:
            raise ValueError('Shared camera IDs or source resolution changed across clips')
        for key in ('fit_numerators', 'diagnostic_numerators'):
            sample_indices(record.num_frames, request[key], request['denominator'])
        snapshots.extend((bound.snapshot, media.snapshot))
        for video in media.source.videos:
            source_by_camera.setdefault(video.camera_id, {})[record.clip_id] = {'sha256': video.sha256, 'num_frames': video.num_frames, 'fps': video.fps}
            paths_by_camera.setdefault(video.camera_id, {})[record.clip_id] = video.path
    assert camera_ids is not None and size is not None
    if set(request['candidate_manifests']) != set(camera_ids):
        raise ValueError('Frozen court candidate cameras must match the whole video')
    cameras = {}
    with store.lock():
        for camera in camera_ids:
            ref = request['candidate_manifests'][camera]
            if set(ref) != {'path', 'sha256'}:
                raise ValueError('Candidate manifest requires path and digest')
            seeds_path = resolver.resolve(PathRole.ARTIFACT, ref['path'])
            if dual_sha256(seeds_path) != ref['sha256']:
                raise ValueError('Frozen candidate manifest changed')
            snapshots.append(InputSnapshot(((seeds_path, ref['sha256']),)))
            frozen = json.loads(seeds_path.read_text())
            identity = {'implementation': code_identity, 'profile': PROFILE, 'dataset_id': config.dataset_id, 'video_id': config.video_id,
                        'camera_id': camera, 'size_wh': list(size), 'sources': source_by_camera[camera], 'candidate_sha256': ref['sha256'],
                        'selection': {key: request[key] for key in ('seed_clip_ids', 'fit_clip_ids', 'bootstrap_clip_ids', 'fit_numerators', 'diagnostic_numerators', 'denominator')}}
            cached = None if config.rebuild else store.cached_camera(identity)
            if cached is not None:
                cameras[camera] = cached
                continue
            estimated = estimate_camera(camera=camera, size_wh=size, sources=source_by_camera[camera], paths=paths_by_camera[camera],
                                        seeds_path=seeds_path, seeds_sha256=ref['sha256'], seed_metadata=frozen['provenance'],
                                        dataset_id=config.dataset_id, video_id=config.video_id, request=request)
            camera_result = {'schema': 'video_court_camera_v1', 'dataset_id': config.dataset_id, 'video_id': config.video_id,
                             'camera_id': camera, 'size_wh': list(size), 'identity': identity, 'sources': source_by_camera[camera], **estimated}
            used_paths = set(paths_by_camera[camera].values())
            current_files = {path: digest for snapshot in snapshots for path, digest in snapshot.files
                             if path in used_paths or path.suffix == '.json'}
            InputSnapshot(tuple(current_files.items())).assert_unchanged()
            store.remember(camera_result, repair=config.rebuild)
            cameras[camera] = camera_result
        InputSnapshot(tuple({path: digest for snapshot in snapshots for path, digest in snapshot.files}.items())).assert_unchanged()
        return store.publish(cameras, repair=config.rebuild)
