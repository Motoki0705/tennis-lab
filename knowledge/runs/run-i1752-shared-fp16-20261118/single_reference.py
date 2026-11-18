"""One single-entry fp16 reference; same frozen media and source as the three-clip run."""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'knowledge/runs/run-i1751-runtime-gpu-a5'))
sys.path.insert(0, str(REPO / 'knowledge/runs/run-video-batch-runtime-20261117'))
from health_guard import capture_diagnostics, run_variant, utc, write, write_anomaly
from doubles_acceptance import common_root, verify

CLIPS = ("video_000/clip_027",)

SETTINGS = Path(__file__).with_name('settings.json')


def overrides():
    return json.loads(SETTINGS.read_text())['overrides']


def arguments(root):
    return [f'paths.data_root={root.parent}', f'paths.output_root={root}',
        f'paths.artifact_root={common_root() / ".worktree-outputs"}',
        'dataset_directory=datasets/single',
        'overwrite=true', 'continue_on_error=true',
        'clip_ids=' + json.dumps(CLIPS, separators=(',', ':')),
        'pipeline_overrides=' + json.dumps(overrides(), separators=(',', ':')),
        f'hydra.run.dir={root / "batch/hydra"}']


def worker(root):
    # Register before heavy imports so a blocked import also has all-thread stacks.
    stack = open(os.environ['TENNIS_STACK_FILE'], 'a')
    faulthandler.enable(file=stack, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=stack, all_threads=True)
    faulthandler.dump_traceback_later(180, file=stack)
    directory = root / 'batch'
    directory.mkdir(exist_ok=True)
    owner = threading.get_ident()
    last = 0.0

    def heartbeat(operation):
        nonlocal last
        now = time.monotonic()
        if threading.get_ident() == owner and now - last >= 1:
            last = now
            faulthandler.dump_traceback_later(180, file=stack)
            write(directory / 'heartbeat.json', {'at_utc': utc(), 'monotonic': now, 'operation': operation})

    heartbeat('imports')
    import src.utils.inference_recording as recording
    original_span = recording.span

    @contextmanager
    def span(stage, **attributes):
        heartbeat({'stage': stage, **attributes})
        try:
            with original_span(stage, **attributes) as result:
                yield result
        finally:
            heartbeat({'finished': stage})

    # Patch before consumers bind their local aliases; forward and scheduler unchanged.
    recording.span = span
    import numpy as np
    import torch
    from src.tennis_scene.generate_dataset import pseudo_annotation
    import src.tennis_scene.scripts.generate_dataset as single
    from src.tennis_scene.pipeline.artifacts import json_value
    from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
    from src.tennis_scene.pipeline.runner import PlanExecutor
    from src.utils.inference_recording import model_event, recording_context

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    state = {'status': 'running', 'started_at_utc': utc(), 'clips': CLIPS,
             'pipeline_overrides': overrides(), 'global_python_profiler': False,
             'runtime_events': [], 'seed': 42, 'torch': torch.__version__, 'cuda': torch.version.cuda,
             'threads': torch.get_num_threads()}

    @contextmanager
    def lifecycle(action, model, **attributes):
        heartbeat({'action': action, 'model': model})
        started = time.perf_counter()
        event = {'action': action, 'model': model, **json_value(attributes)}
        try:
            with model_event(action, model, **attributes):
                yield
        except BaseException as error:
            event.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        else:
            event['status'] = 'complete'
        finally:
            event.update(seconds=time.perf_counter() - started, at_utc=utc())
            state['runtime_events'].append(event)
            print(json.dumps(event), flush=True)
            write(directory / 'runtime-events.json', state['runtime_events'])
            heartbeat({'finished': action, 'model': model})

    def construct(runtime):
        if (runtime.player_association.players_per_side != 2 or runtime.video_court is None
                or runtime.video_court_side is None or runtime.people.runtime.vitpose.precision != 'fp16'
                or runtime.people.runtime.dino_detector.precision != 'fp16'):
            raise ValueError('The parent-pinned doubles/shared/fp16 settings changed')
        state['processing_settings'] = json_value(runtime.processing_settings)
        return TennisSceneOrchestrator(runtime, model_event=lifecycle)

    original_publish = pseudo_annotation.publish_annotation
    original_finalize = PlanExecutor.finalize_clip

    def publication(inputs, *args, **kwargs):
        with recording_context(clip_id=inputs.record.clip_id, node='clip_publication'):
            return original_publish(inputs, *args, **kwargs)

    def finalize(executor, clip_id):
        with recording_context(clip_id=clip_id, node='scene_export'):
            return original_finalize(executor, clip_id)

    started = time.perf_counter()
    try:
        with (patch.object(TennisSceneOrchestrator, 'from_runtime_config', side_effect=construct),
              patch.object(pseudo_annotation, 'publish_annotation', side_effect=publication),
              patch.object(PlanExecutor, 'finalize_clip', finalize)):
            sys.argv = ['tennis_scene.single', *arguments(root)]
            original_compose = single._compose_pipeline_config
            recording_path = directory / 'recording.json'
            state['recording'] = str(recording_path)
            with (patch.object(single, '_compose_pipeline_config',
                               side_effect=lambda cfg: original_compose(cfg, config_name='pipeline_x3d_dpt')),
                  recording.InferenceRecorder(recording_path, devices=('cuda:0',),
                                              metadata={'phase': 'single', 'clips': CLIPS})):
                single.main()
        state['status'] = 'complete'
    except BaseException as error:
        detail = traceback.format_exc()
        sys.stdout.flush()
        print(detail, file=sys.stderr, flush=True)
        state.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=detail)
        if any(word in detail.lower() for word in ('cuda', 'cublas', 'cudnn', 'nvml', 'illegal memory access')):
            write_anomaly(root, {'kind': 'cuda_error', 'detail': detail, 'pid': os.getpid()})
        faulthandler.dump_traceback(file=stack, all_threads=True)
        raise
    finally:
        state.update(finished_at_utc=utc(), entry_wall_seconds=time.perf_counter() - started)
        write(directory / 'entry.json', state)
        faulthandler.cancel_dump_traceback_later()
        stack.close()


def run(root, expected):
    if not os.environ.get('TENNIS_RUN_ID'):
        raise RuntimeError('GPU qualification must use the shared queue')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if actual != expected:
        raise ValueError(f'Expected {expected}, got {actual}')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'src', str(Path(__file__).parent.relative_to(REPO))], check=True)
    verify(root)
    if (root / 'terminal.json').exists():
        raise FileExistsError('One run per directory, no automatic retry')
    batch_commit = json.loads((root.parent / 'terminal.json').read_text())['commit']
    subprocess.run(['git', 'diff', '--exit-code', batch_commit, 'HEAD', '--', 'src'], check=True)
    terminal = {'commit': actual, 'batch_commit': batch_commit, 'source_tree_unchanged': True, 'clips': CLIPS, 'queue_job': os.environ['TENNIS_RUN_ID'], 'started_at_utc': utc()}
    result = run_variant(root, 'batch', [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                         'worker', '--root', str(root)], REPO)
    terminal['batch'] = result
    if (root / 'anomaly.json').exists() and 'diagnostics' not in result:
        class Exited:
            pid = result.get('pid')
            def poll(self):
                return result.get('returncode', 1)
        result['diagnostics'] = capture_diagnostics(root, 'batch', Exited())
    terminal.update(status=result['status'], finished_at_utc=utc())
    write(root / 'terminal.json', terminal)
    verify(root)
    return 0 if terminal['status'] == 'complete' else 2


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expect-commit')
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to((REPO / 'outputs').resolve()):
        raise ValueError('Qualification output must stay in this worktree')
    if args.action == 'prepare':
        verify(root.parent)
        terminal = json.loads((root.parent / 'terminal.json').read_text())
        if terminal['status'] != 'complete' or (root.parent / 'anomaly.json').exists():
            raise ValueError('The batch must succeed without CUDA anomaly before the reference')
        root.mkdir(exist_ok=False)
        (root / 'input').symlink_to(root.parent / 'input', target_is_directory=True)
        shutil.copyfile(root.parent / 'input-manifest.json', root / 'input-manifest.json')
        verify(root)
        return 0
    if args.action == 'worker':
        worker(root)
        return 0
    return run(root, args.expect_commit)


if __name__ == '__main__':
    raise SystemExit(main())
