"""Ball-specific statistics HTTP extension, with one bounded CPU job per server."""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from src.tasks.ball_detection.dataset_statistics.configuration import StatisticsConfig
from src.tasks.ball_detection.dataset_statistics.pipeline import (
    compute_dataset_statistics,
)
from src.tasks.ball_detection.visualization.review.datasets import (
    BallDatasetCatalog,
    split_scene_id,
)


class StatisticsRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dataset: str
    settings: dict[str, Any]


class StatisticsService:
    def __init__(self, project_root: Path, data_root: Path, play_poses: Path | None) -> None:
        self.project_root, self.data_root, self.play_poses = project_root, data_root, play_poses
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ball-statistics')
        self._job: dict[str, Any] | None = None

    def start(self, dataset: str, settings: StatisticsConfig) -> dict[str, Any]:
        # Validate configuration and catalog membership before scheduling work.
        settings.validate()
        catalog = BallDatasetCatalog(self.data_root, play_poses=self.play_poses, project_root=self.project_root)
        refs = catalog.refs(dataset)
        if not refs:
            raise ValueError('No available clips in the selected dataset')
        store, _ = catalog.store_clip(dataset, refs[0].local_id)
        poses = self.play_poses if catalog.spec(dataset).pose_approved else None
        with self._lock:
            if self._job is not None and self._job['state'] in ('pending', 'running'):
                raise HTTPException(409, 'A statistics calculation is already running')
            job: dict[str, Any] = dict(id=uuid.uuid4().hex, dataset=dataset, state='pending', completed=0,
                                       total=len(refs), error=None, report=None)
            self._job = job
        self._executor.submit(self._run, job, store, [ref.clip_id for ref in refs], poses, settings)
        return self.status(job['id'])

    def _run(self, job: dict[str, Any], store: Any, clips: list[str], poses: Path | None, settings: StatisticsConfig) -> None:
        try:
            with self._lock:
                job['state'] = 'running'
            def progress(completed: int, total: int) -> None:
                with self._lock:
                    job['completed'], job['total'] = completed, total
            report = compute_dataset_statistics(store, clips, settings, project_root=self.project_root,
                                                pose_directory=poses, progress=progress)
            with self._lock:
                job['report'], job['state'] = report, 'complete'
        except Exception as error:
            with self._lock:
                job['state'], job['error'] = 'failed', f'{type(error).__name__}: {error}'

    def _require(self, identifier: str) -> dict[str, Any]:
        if self._job is None or identifier != self._job['id']:
            raise HTTPException(404, 'Statistics job is unavailable (only the latest job is retained)')
        return self._job

    def status(self, identifier: str) -> dict[str, Any]:
        with self._lock:
            job = self._require(identifier)
            return {k: v for k, v in job.items() if k != 'report'}

    def result(self, identifier: str) -> dict[str, Any]:
        with self._lock:
            job = self._require(identifier)
            if job['state'] != 'complete':
                raise HTTPException(409, job['error'] or 'Statistics are not complete')
            return dict(job['report'])

    def overview(self, identifier: str) -> dict[str, Any]:
        report = self.result(identifier)
        rows = []
        for clip_id, clip in report['clips'].items():
            scopes = {}
            for scope in ('clip', 'play', 'selected', 'excluded'):
                metrics = clip['scopes'][scope]
                scopes[scope] = dict(
                    frames=metrics['counts']['frames'],
                    observed=metrics['rates']['frames/observed']['value'],
                    interpolated=metrics['rates']['frames/interpolated']['value'],
                    gap_p95=metrics['distributions']['gaps/coordinate_gap/bounded/frames']['p95'],
                    speed_p95=metrics['distributions']['motion/observed/speed_px']['p95'],
                    pose_flags=metrics['counts'].get('pose/flagged_joint_frames'),
                )
            rows.append(dict(clip_id=clip_id, source=clip['clip']['source'], split=clip['clip']['split'],
                             group_id=clip['clip']['group_id'], scopes=scopes,
                             annotation=clip['original_annotation']['availability'], pose=clip['pose_availability']))
        return {**{k: v for k, v in report.items() if k != 'clips'}, 'clips': rows}

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def statistics_router(service: StatisticsService) -> APIRouter:
    router = APIRouter()
    static = Path(__file__).parent.parent / 'static' / 'statistics'

    def asset(name: str) -> FileResponse:
        if name not in {'panel.mjs', 'charts.mjs', 'style.css'}:
            raise HTTPException(404)
        return FileResponse(static / name, media_type='text/css' if name.endswith('.css') else 'text/javascript')

    def settings() -> dict[str, Any]:
        return StatisticsConfig.shipped().to_dict()

    def start(request: StatisticsRequest) -> dict[str, Any]:
        return service.start(request.dataset, StatisticsConfig.from_mapping(request.settings))

    def status(identifier: str) -> dict[str, Any]:
        return service.status(identifier)

    def overview(identifier: str) -> dict[str, Any]:
        return service.overview(identifier)

    def clip(identifier: str, scene: str) -> dict[str, Any]:
        dataset, clip_id = split_scene_id(scene)
        if service.status(identifier)['dataset'] != dataset:
            raise ValueError('Statistics job and selected scene use different datasets')
        report = service.result(identifier)
        if clip_id not in report['clips']:
            raise HTTPException(404, 'Clip is not part of this statistics snapshot')
        return dict(report['clips'][clip_id])

    router.add_api_route('/statistics-static/{name}', asset, methods=['GET'])
    router.add_api_route('/api/statistics/config', settings, methods=['GET'])
    router.add_api_route('/api/statistics/jobs', start, methods=['POST'], status_code=202)
    router.add_api_route('/api/statistics/jobs/{identifier}', status, methods=['GET'])
    router.add_api_route('/api/statistics/jobs/{identifier}/result', overview, methods=['GET'])
    router.add_api_route('/api/statistics/jobs/{identifier}/clip', clip, methods=['GET'])
    return router
