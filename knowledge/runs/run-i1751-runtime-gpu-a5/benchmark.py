"""Pinned main/runtime replay; run GPU mode only inside the shared queue.

Instrumentation observes Python loader calls with sys.setprofile. It does not
replace model loaders, unload methods, component methods or scheduling.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import faulthandler
import gc
import hashlib
import inspect
import json
import logging
import os
import platform
import random
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
from health_guard import HEARTBEAT_TIMEOUT_SECONDS, write_anomaly

STACK_HANDLE = open(os.environ['TENNIS_STACK_FILE'], 'a') if os.environ.get('TENNIS_STACK_FILE') else None
if STACK_HANDLE is not None:
    faulthandler.enable(file=STACK_HANDLE, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=STACK_HANDLE, all_threads=True)
    faulthandler.dump_traceback_later(HEARTBEAT_TIMEOUT_SECONDS, file=STACK_HANDLE)

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.archive import save_scene_result
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.clip_inputs import prepare_clip_source
from src.tennis_scene.pipeline.definition import enabled_model_assets
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole

MAIN_ROOT = Path('/home/kamimura/projects/tennis-lab')
CLIP = 'tennis_multiview/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000'
CAMERAS = ('cam0', 'cam1', 'cam2')
SEED = 42


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def configuration():
    with initialize_config_dir(config_dir=str(ROOT / 'src/tennis_scene/configs'), version_base='1.3'):
        cfg = compose(config_name='pipeline_x3d_dpt', overrides=[
            'video_paths=[' + ','.join(f'{CLIP}/media/{c}.mp4' for c in CAMERAS) + ']',
            'camera_ids=[' + ','.join(CAMERAS) + ']',
            'clip_id=video_000/clip_000',
            'frame_sampling.max_frames=1024',
            'output_directory=tennis_scene/evaluate/runtime-i1751-20261118-a5',
            'output_name=scene',
            f'paths.external_asset_root={MAIN_ROOT}/third_party',
            'cache.overwrite=false',
        ])
    return PipelineRuntimeConfig.from_config(cfg), cfg


def pinned_inputs(runtime):
    source = prepare_clip_source(runtime.video_paths, runtime.camera_ids,
                                 max_frames=None, clip_id=runtime.clip_id).source
    sampled = sampled_frame_indices(source.num_frames, source.fps, max_frames=runtime.sampling_max_frames)
    assets = enabled_model_assets(runtime)
    result = {'seed': SEED, 'clip': CLIP, 'source': json_value(source),
              'sampled_frames': len(sampled), 'max_frames': runtime.max_frames,
              'settings': json_value(runtime.processing_settings),
              'models': {name: {'path': str(path.resolve()), 'size': path.stat().st_size,
                                'sha256': dual_sha256(path)} for name, path in assets.items()},
              'extension': {'path': os.environ['TENNIS_LAB_DINO_EXTENSION'],
                            'sha256': dual_sha256(os.environ['TENNIS_LAB_DINO_EXTENSION'])}}
    return result


class Observer:
    TARGETS = {
        ('src.tasks.court_detection.inference.predictor', 'CourtPredictor.load_from_checkpoint'): 'court',
        ('src.tasks.ball_detection.inference.predictor', 'BallDetectionPredictor.load_from_checkpoint'): 'ball',
        ('src.tasks.ball_detection.inference.x3d_dpt', 'X3DDPTPredictor.load_from_checkpoint'): 'ball',
        ('src.tasks.player_association.appearance.encoders', 'ClipReIDEncoder.__post_init__'): 'appearance',
        ('src.submodules.models.dino.person_detector', 'DinoPersonDetector._load_impl'): 'dino',
        ('src.submodules.models.vitpose.pose2d', 'ViTPosePose2D._load_impl'): 'vitpose',
        ('src.submodules.models.hmr2.feature_extractor', 'Hmr2FeatureExtractor._load_impl'): 'hmr2',
        ('src.submodules.models.gvhmr.mesh_recovery', 'GvhmrMeshRecovery._load_impl'): 'gvhmr',
    }

    def __init__(self, directory):
        self.directory = directory
        self.calls = []
        self.pending = {}
        self.runtime_events = []
        self.started = time.perf_counter()
        self.last_heartbeat = 0.0
        self.phase = "preflight"
        self.current_model = None
        self.cuda_error_seen = False

    def emit(self, value):
        value['elapsed_seconds'] = time.perf_counter() - self.started
        value['at_utc'] = datetime.now(UTC).isoformat()
        with (self.directory / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(value, sort_keys=True) + '\n')
        print(json.dumps(value, sort_keys=True), flush=True)

    @staticmethod
    def target(frame):
        module = frame.f_globals.get('__name__', '')
        name = frame.f_code.co_qualname
        target = Observer.TARGETS.get((module, name))
        if target:
            return target
        obj = frame.f_locals.get('self')
        if module == 'src.submodules.models.gvhmr.mesh_recovery' and frame.f_code.co_name == '_ensure_loaded':
            if name.startswith('SmplVertexReconstructor.') and obj._smplx is None:
                return 'body_vertices'
            if name.startswith('SmplCoco17Reconstructor.') and obj._model is None:
                return 'body_coco17'
        if (module, name) == ('src.tennis_scene.pipeline.components.body_placement', 'SmplGeometry.root_regressor') and obj._root is None:
            return 'body_root_regressor'
        return None

    def heartbeat(self, frame=None, event=None):
        now = time.monotonic()
        if now - self.last_heartbeat >= 1.0:
            self.last_heartbeat = now
            if STACK_HANDLE is not None:
                faulthandler.dump_traceback_later(HEARTBEAT_TIMEOUT_SECONDS, file=STACK_HANDLE)
            write(self.directory / 'heartbeat.json', {
                'at_utc': datetime.now(UTC).isoformat(), 'monotonic': now,
                'phase': self.phase, 'model': self.current_model,
                'operation': None if frame is None else f'{frame.f_globals.get("__name__", "")}.{frame.f_code.co_qualname}',
                'file': None if frame is None else frame.f_code.co_filename,
                'line': None if frame is None else frame.f_lineno, 'event': event,
            })

    @torch.compiler.disable
    def profile(self, frame, event, arg):
        self.heartbeat(frame, event)
        if event == 'call':
            target = self.target(frame)
            if target:
                self.current_model = target
                self.pending[id(frame)] = (target, time.perf_counter())
                self.emit({'event': 'backend_load_start', 'model': target})
        elif event == 'return':
            pending = self.pending.pop(id(frame), None)
            if pending:
                target, started = pending
                item = {'event': 'backend_load_return', 'model': target,
                        'seconds': time.perf_counter() - started}
                self.calls.append(item)
                self.emit(item)

    @contextmanager
    def model_event(self, action, model, *, weights, device, dtype, instance):
        started = time.perf_counter()
        self.current_model = model
        item = {'event': 'runtime_model', 'action': action, 'model': model,
                'weights': {name: str(path) for name, path in weights.items()},
                'device': device, 'dtype': dtype, 'instance': instance}
        try:
            yield
        except BaseException as exc:
            item.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            if cuda_failure(traceback.format_exc()):
                self.cuda_error_seen = True
                record_anomaly(self.directory, 'cuda_error', traceback.format_exc(), self)
            raise
        else:
            item['status'] = 'complete'
        finally:
            item['seconds'] = time.perf_counter() - started
            self.runtime_events.append(item)
            self.emit(item)


def cuda_failure(message):
    lower = message.lower()
    return any(word in lower for word in ('cuda', 'torch/cuda',
        'cublas', 'cudnn', 'device-side assert', 'illegal memory access', 'nvml', 'cuda out of memory'))


def record_anomaly(directory, kind, detail, observer):
    write_anomaly(directory.parent, {'kind': kind, 'variant': directory.name,
        'phase': observer.phase, 'last_model': observer.current_model, 'detail': detail,
        'heartbeat': json.loads((directory / 'heartbeat.json').read_text())
            if (directory / 'heartbeat.json').exists() else None})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--pin', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    if args.prepare or args.check_inputs:
        runtime, _ = configuration()
        inputs = pinned_inputs(runtime)
        if args.prepare:
            if args.pin.exists():
                raise FileExistsError(args.pin)
            write(args.pin, inputs)
        elif inputs != json.loads(args.pin.read_text()):
            raise ValueError('Pinned inputs/settings/weights differ')
        print('Pinned input/settings/weight verification passed', flush=True)
        return
    if not os.environ.get('TENNIS_RUN_ID'):
        raise RuntimeError('GPU replay requires the shared training queue')
    directory = args.directory.resolve()
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    observer = Observer(directory)
    observer.heartbeat()
    manifest = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'cwd': str(ROOT), 'queue_job': os.environ['TENNIS_RUN_ID'],
                'python': platform.python_version(), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'started_at_utc': datetime.now(UTC).isoformat(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'threads': torch.get_num_threads(), 'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
                'status': 'running'}
    write(directory / 'measurement.json', manifest)
    orchestrator = None
    started = time.perf_counter()
    sys.setprofile(observer.profile)
    try:
        runtime, cfg = configuration()
        inputs = pinned_inputs(runtime)
        if inputs != json.loads(args.pin.read_text()):
            raise ValueError('Inputs/settings/weights/extension differ from the fixed pin')
        write(directory / 'config.json', OmegaConf.to_container(cfg, resolve=True))
        observer.phase = 'cuda_initialization'
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        hook_supported = 'model_event' in inspect.signature(TennisSceneOrchestrator).parameters
        options = {'model_event': observer.model_event} if hook_supported else {}
        orchestrator = TennisSceneOrchestrator.from_runtime_config(runtime, **options)
        manifest.update(gpu=torch.cuda.get_device_name(),
                        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                        runtime_hook=hook_supported)
        observer.phase = 'pipeline_run'
        observer.emit({'event': 'pipeline_start', 'commit': manifest['commit']})
        result = orchestrator.run(runtime.video_paths, video_role=PathRole.DATA,
                                  camera_ids=runtime.camera_ids, store_root=directory / 'store',
                                  max_frames=runtime.max_frames, clip_id=runtime.clip_id)
        observer.phase = 'save_scene'
        save_scene_result(result, directory / 'scene.npz')
        manifest.update(status='complete', frames=result.num_frames, scene_status=result.metadata['status'])
    except BaseException as exc:
        sys.setprofile(None)
        detail = traceback.format_exc()
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                        error_traceback=detail, error_at_utc=datetime.now(UTC).isoformat(),
                        failed_phase=observer.phase, last_model=observer.current_model)
        if cuda_failure(detail):
            observer.cuda_error_seen = True
            record_anomaly(directory, 'cuda_error', detail, observer)
        raise
    finally:
        sys.setprofile(None)
        manifest.update(wall_seconds=time.perf_counter() - started,
                        finished_at_utc=datetime.now(UTC).isoformat(),
                        backend_load_counts=dict(Counter(item['model'] for item in observer.calls)),
                        backend_loads=observer.calls, runtime_events=observer.runtime_events,
                        receipt=None if orchestrator is None else orchestrator.last_receipt)
        # No further CUDA query or synchronization after a CUDA error.
        if not observer.cuda_error_seen and torch.cuda.is_initialized():
            manifest.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            allocated_end_bytes=torch.cuda.memory_allocated(),
                            reserved_end_bytes=torch.cuda.memory_reserved())
            gc.collect()
            torch.cuda.empty_cache()
            manifest.update(allocated_after_gc_bytes=torch.cuda.memory_allocated(),
                            reserved_after_gc_bytes=torch.cuda.memory_reserved())
        write(directory / 'measurement.json', manifest)
        observer.emit({'event': 'pipeline_terminal', 'status': manifest['status']})
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
