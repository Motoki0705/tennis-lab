"""Bounded 720p performance pilot using the committed temporal-heatmap runner."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from session_guard import elapsed, write

CONDITIONS = [(architecture, frames) for frames in (32, 64, 128) for architecture in ('dpt', 'hybrid')]
IDENTITIES = ('common_intervals_sha256', 'target_identity_sha256', 'teacher_identity_sha256',
              'scored_frame_sha256', 'training_packs_sha256')
PRIOR = Path('knowledge/runs/run-ball-colab-checkpoint-resume-20261024')
PRIOR_INPUTS = Path('/content/tennis-lab/outputs/ball_detection/precompute/temporal_heatmap/resume-proof-20261024')
FULL_STORE = Path('/content/tennis-lab/data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1')


def commit() -> str:
    subprocess.run(['git', 'diff', '--quiet'], check=True)
    subprocess.run(['git', 'diff', '--cached', '--quiet'], check=True)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def store(root: Path, remote_root: str) -> Any:
    from src.utils.artifact_store import RcloneSnapshotStore
    remote, fragment = remote_root.split(':', 1)
    return RcloneSnapshotStore(root, remote=remote, remote_root=fragment,
                               config_path=Path(os.environ['RCLONE_CONFIG']))


def gpu() -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('The pilot requires exactly one assigned CUDA GPU')
    name = torch.cuda.get_device_name()
    if name != 'NVIDIA L4':
        raise RuntimeError(f'Requested L4; observed {name}')
    return dict(name=name, total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)


def remaining(created_at: str) -> float:
    return 12600 - elapsed(created_at)


def prepare(args: argparse.Namespace) -> None:
    import shutil

    import yaml

    from src.tasks.ball_detection.data.distribution import verify_distribution
    from src.tasks.ball_detection.data.store import BallFrameStore
    from src.tasks.ball_detection.data.windows.all_frame_preparation import (
        prepare_all_frame_training,
    )
    from src.tasks.ball_detection.data.windows.all_frames import (
        AllFrameWindowConfig,
        window_starts,
    )
    from src.utils.checksum import dual_sha256

    started = time.monotonic()
    args.receipts.mkdir(parents=True, exist_ok=False)
    reference = json.loads((PRIOR / 'receipts/restore-references.json').read_text())['inputs']
    parent_root = reference['remote'].rsplit('/', 1)[0]
    restoration = store(PRIOR_INPUTS, parent_root).restore(
        reference['generation'], manifest_sha256=reference['manifest_sha256'])
    write(args.receipts / 'restoration.json', restoration)
    full = json.loads((PRIOR_INPUTS / 'full_windows/train_T32.json').read_text())
    if (Path(full['ball_store']['directory']) != FULL_STORE or
            full['ball_store']['hashes'] != {name: dual_sha256(FULL_STORE / name) for name in ('index.npz', 'metadata.json')}):
        raise ValueError('Restored full manifest does not identify the downloaded distribution')
    full_workloads = {}
    for length in (32, 64, 128):
        windows = sum(len(window_starts(a, b, length, length // 2)) for record in full['clips']
                      if record['clip']['split'] == 'val' for a, b in record['common_intervals'])
        full_workloads[str(length)] = dict(windows=windows, input_frames=windows * length)
    full_evaluation = json.loads((PRIOR_INPUTS / 'full_windows/evaluation_T32.json').read_text())
    if full_workloads['32']['windows'] != full_evaluation['summaries']['val']['counts']['windows']:
        raise ValueError('Counted evaluation workload differs from the saved owner manifest')
    diagnostic = PRIOR_INPUTS / 'diagnostic_store'
    diagnostic_verification = verify_distribution(diagnostic, workers=2)
    source = BallFrameStore(diagnostic)
    if any((clip.width, clip.height) != (1280, 720) for clip in source.clips):
        raise ValueError('Every pilot clip must be native 1280x720')
    args.inputs.mkdir(parents=True, exist_ok=False)
    plan = prepare_all_frame_training(diagnostic, args.inputs / 'windows',
                                     configs=[AllFrameWindowConfig(t) for t in (32, 64, 128)])
    previous = json.loads((PRIOR_INPUTS / 'diagnostic_windows/train_T32.json').read_text())
    for split in ('train', 'val', 'test'):
        if any(plan['datasets']['32']['train']['summaries'][split][k] != previous['summaries'][split][k]
               for k in IDENTITIES):
            raise ValueError('Pilot common set differs from the fixed diagnostic input')
    for architecture, frames in CONDITIONS:
        filename = 'mdd_dpt_heatmap.yaml' if architecture == 'dpt' else 'mdd_heatmap_hybrid.yaml'
        model = yaml.safe_load((Path('src/tasks/ball_detection/configs/model') / filename).read_text())
        model.update(frames=frames, frame_chunk_size=4, activation_checkpointing=True)
        (args.inputs / f'{architecture}-T{frames}.yaml').write_text(yaml.safe_dump(model))
    recipe = dict(schema_version=1, code_commit=commit(), uv_lock_sha256=dual_sha256(Path('uv.lock')),
                  updates=10, input_frames=128, microbatch_windows=1, learning_rate=.0001,
                  warmup_updates=2, minimum_lr_ratio=.1, weight_decay=.01, gradient_clip=1.,
                  evaluation_interval=10, checkpoint_interval=5, seed=42, sigma_ratio=.012, gamma=2.,
                  conditions=CONDITIONS, max_condition_seconds=1200, stop_work_seconds=12600,
                  native_image_size=[1280, 720], augmentation='none', diagnostic_input_reference=reference,
                  purpose='Performance only; diagnostic subset; no scientific pretraining or accuracy selection')
    write(args.inputs / 'pilot.json', recipe)
    publication = store(args.inputs, args.remote_root).publish('inputs-' + uuid.uuid4().hex)
    write(args.receipts / 'preparation.json', dict(
        schema_version=1, seconds=time.monotonic() - started, gpu=gpu(), restoration=restoration,
        diagnostic_verification=diagnostic_verification, source_clip_ids=[c.clip_id for c in source.clips],
        diagnostic_summaries=previous['summaries'], full_summaries=full['summaries'],
        diagnostic_evaluation_counts={t: plan['datasets'][t]['evaluation']['summaries']['val']['counts'] for t in ('32', '64', '128')},
        full_validation_workloads=full_workloads,
        full_manifest_sha256=dual_sha256(PRIOR_INPUTS / 'full_windows/train_T32.json'),
        input_publication=publication, recipe=recipe, disk=shutil.disk_usage(args.inputs)._asdict()))


def condition(args: argparse.Namespace) -> int:
    import torch

    import src.tasks.ball_detection.training.temporal_heatmap.runner as runner
    from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
    from src.tasks.ball_detection.training.temporal_heatmap.configuration import (
        UpdateBudget,
        load_model_config,
    )
    from src.utils.artifact_store import ArtifactStoreError
    from src.utils.checksum import dual_sha256

    started = time.monotonic()
    args.receipts.mkdir(parents=True, exist_ok=False)
    recipe = json.loads((args.inputs / 'pilot.json').read_text())
    if commit() != recipe['code_commit'] or dual_sha256(Path('uv.lock')) != recipe['uv_lock_sha256']:
        raise ValueError('Use the committed pilot and its exact lockfile')
    observed = gpu()
    budget = UpdateBudget(**{k: recipe[k] for k in UpdateBudget.__dataclass_fields__})
    runtime = CoordinateRuntime(precision='bf16', num_workers=0, pin_memory=True, prefetch_factor=1,
                                cpu_threads=2, compile_mode='off', compile_recompile_limit=8,
                                jpeg_decoder='nvjpeg', input_verification='upfront', image_prefetch=False)
    model = load_model_config(args.inputs / f'{args.architecture}-T{args.frames}.yaml', args.architecture)
    publisher = store(args.root, args.remote_root)
    publications: list[dict[str, Any]] = []
    timings: dict[str, list[dict[str, Any]]] = dict(evaluation=[], local_serialization=[], local_hash=[])
    originals: dict[str, Callable[..., Any]] = {}

    def timed(name: str, kind: str) -> None:
        original = getattr(runner, name)
        originals[name] = original

        def wrapped(*positional: Any, **keywords: Any) -> Any:
            if kind == 'local_hash' and not Path(positional[0]).is_relative_to(args.root):
                return original(*positional, **keywords)
            torch.cuda.synchronize()
            if kind == 'evaluation':
                torch.cuda.reset_peak_memory_stats()
            before = time.perf_counter()
            success = False
            try:
                result = original(*positional, **keywords)
                torch.cuda.synchronize()
                success = True
                return result
            finally:
                measurement = dict(seconds=time.perf_counter() - before, completed=success)
                if kind == 'evaluation':
                    measurement.update(peak_vram_bytes=torch.cuda.max_memory_allocated(),
                                       peak_reserved_vram_bytes=torch.cuda.max_memory_reserved())
                timings[kind].append(measurement)
                write(args.receipts / 'timings.json', timings)
        setattr(runner, name, wrapped)

    def persist(output: Path, stage: str) -> None:
        latest = json.loads((output / 'latest.json').read_text()) if (output / 'latest.json').exists() else None
        update = latest['update'] if latest else 0
        current = {f'checkpoint-update-{update:08d}.pt', f'weights-update-{update:08d}.pt'}
        include = [p.relative_to(output).as_posix() for p in sorted(output.rglob('*')) if p.is_file()
                   and (not p.name.startswith(('checkpoint-update-', 'weights-update-')) or p.name in current)]
        if stage == 'completed':
            include = ['COMPLETED.json', 'latest.json']
        receipt = publisher.publish(f'{stage}-{update:08d}-{uuid.uuid4().hex}', include=include)
        receipt.update(stage=stage, update=update)
        publications.append(receipt)
        write(args.receipts / 'publications.json', publications)

    for name, kind in (('evaluate_temporal_heatmap', 'evaluation'), ('save_torch', 'local_serialization'),
                       ('dual_sha256', 'local_hash')):
        timed(name, kind)
    summary: dict[str, Any] = dict(schema_version=1, architecture=args.architecture, frames=args.frames,
                                   code_commit=recipe['code_commit'], gpu=observed, runtime=asdict(runtime),
                                   model=asdict(model), budget=budget.payload(), timings=timings,
                                   publications=publications, status='failed')
    try:
        initialization: dict[str, Any] = {}
        if args.frames != 32:
            parent = args.root.parent / f'{args.architecture}-T32/weights-update-00000010.pt'
            initialization = dict(initial_artifact=parent, initial_artifact_sha256=dual_sha256(parent), parent_update=10)
        result = runner.train_temporal_heatmap(
            architecture=args.architecture, phase='pretrain' if args.frames == 32 else 'finetune',
            manifest=args.inputs / f'windows/train_T{args.frames}.json',
            evaluation_manifest=args.inputs / f'windows/evaluation_T{args.frames}.json',
            model_config=model, output=args.root, budget=budget, runtime=runtime, device=torch.device('cuda'),
            augmentation=None, distance_px=20., persist=persist, **initialization)
        summary.update(status='completed', result=result)
        returncode = 0
    except torch.OutOfMemoryError as error:
        summary.update(status='oom', exception=type(error).__name__, message=str(error))
        returncode = 2
    except ArtifactStoreError as error:
        summary.update(status='storage_failed', exception=type(error).__name__, message=str(error))
        returncode = 3
    except Exception as error:
        summary.update(exception=type(error).__name__, message=str(error))
        returncode = 1
    finally:
        for name, original in originals.items():
            setattr(runner, name, original)
        records = [json.loads(line) for line in (args.root / 'train.jsonl').read_text().splitlines()] if (args.root / 'train.jsonl').exists() else []
        summary.update(seconds=time.monotonic() - started, records=records,
                       completed_marker=(args.root / 'COMPLETED.json').exists(),
                       latest=json.loads((args.root / 'latest.json').read_text()) if (args.root / 'latest.json').exists() else None)
        write(args.receipts / 'condition.json', summary)
        print(json.dumps(dict(architecture=args.architecture, frames=args.frames, status=summary['status'],
                              seconds=summary['seconds'])), flush=True)
    return returncode


def matrix(args: argparse.Namespace) -> None:
    args.receipts.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    blocked = None
    for architecture, frames in CONDITIONS:
        name = f'{architecture}-T{frames}'
        if blocked or remaining(args.created_at) < 1200:
            records.append(dict(condition=name, status='unmeasured', reason=blocked or 'session_stop_reserve'))
            write(args.receipts / 'matrix.json', dict(schema_version=1, conditions=records))
            continue
        if frames != 32 and not (args.root / f'{architecture}-T32/COMPLETED.json').exists():
            records.append(dict(condition=name, status='unmeasured', reason='T32 pilot parent unavailable'))
            write(args.receipts / 'matrix.json', dict(schema_version=1, conditions=records))
            continue
        destination = args.receipts / name
        argv = [sys.executable, str(Path(__file__).resolve()), 'condition', '--architecture', architecture,
                '--frames', str(frames), '--inputs', str(args.inputs), '--root', str(args.root / name),
                '--receipts', str(destination), '--remote-root', args.remote_root + '/' + name]
        started = time.monotonic()
        with (args.receipts / f'{name}.log').open('wb') as log:
            process = subprocess.Popen(argv, stdout=log, stderr=log, start_new_session=True)
            try:
                returncode = process.wait(timeout=1180)
                status = json.loads((destination / 'condition.json').read_text())['status'] if (destination / 'condition.json').exists() else 'failed_before_receipt'
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                returncode, status = 124, 'timeout'
        records.append(dict(condition=name, status=status, returncode=returncode,
                            seconds=time.monotonic() - started, max_seconds=1200, argv=argv))
        write(args.receipts / 'matrix.json', dict(schema_version=1, created_at=args.created_at,
                                                conditions=records, observed_at=datetime.now(UTC).isoformat()))
        if returncode not in (0, 2, 124):
            blocked = f'{name}: {status}; investigate before continuing'
    if blocked:
        raise RuntimeError(blocked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'condition', 'matrix'):
        child = commands.add_parser(command)
        for name in ('inputs', 'receipts'):
            child.add_argument('--' + name, type=Path, required=True)
        child.add_argument('--remote-root', required=True)
        if command != 'prepare':
            child.add_argument('--root', type=Path, required=True)
        if command == 'condition':
            child.add_argument('--architecture', choices=('dpt', 'hybrid'), required=True)
            child.add_argument('--frames', type=int, choices=(32, 64, 128), required=True)
        if command == 'matrix':
            child.add_argument('--created-at', required=True)
    args = parser.parse_args()
    if args.command == 'condition':
        sys.exit(condition(args))
    elif args.command == 'prepare':
        prepare(args)
    else:
        matrix(args)


if __name__ == '__main__':
    main()
