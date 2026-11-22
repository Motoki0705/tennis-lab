"""CPU-only parsing, input/admission validation and planning of the saved command.

Never execute the command, model factories, native loaders or the plan. Temporary
stores stay outside the dataset; existing completion markers must remain intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
COMMAND = Path(__file__).with_name('full69-command.txt')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_command():
    lines = COMMAND.read_text().splitlines()
    start = lines.index("task_gpu_command=$(cat <<'GPU_COMMAND'") + 1
    end = lines.index('GPU_COMMAND', start)
    tokens = shlex.split('\n'.join(lines[start:end]))
    index = tokens.index('-m')
    assert tokens[index + 1] == 'src.tennis_scene.scripts.run_video_batch'
    env = {}
    assert tokens[0] == 'env'
    for token in tokens[1:]:
        if '=' not in token:
            break
        key, value = token.split('=', 1)
        env[key] = value
    return env, tokens[index + 2:]


def markers(dataset):
    return {str(p.relative_to(dataset)): sha(p) for p in
            dataset.glob('videos/video_000/clips/*/annotations/tennis_scene/annotation.json')}


def validate(destination):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU validation requires CUDA_VISIBLE_DEVICES empty')
    started = time.perf_counter()
    environment, arguments = parse_command()
    with patch.dict(os.environ, environment):
        import torch
        from hydra import compose, initialize_config_dir
        from src.submodules.models._base.inference_model import BaseInferenceModel
        from src.tennis_scene.configuration import parse_video_batch_config
        from src.tennis_scene.pipeline.artifacts import json_value
        from src.tennis_scene.pipeline.clip_inputs import prepare_video_inputs
        from src.tennis_scene.pipeline.components.court_side import PlayerCourtSideModule
        from src.tennis_scene.pipeline.definition import enabled_model_assets, standard_definition
        from src.tennis_scene.pipeline.execution_plan import ClipExecution, ExecutionPlan
        from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime
        from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
        from src.tennis_scene.pipeline.runner import PlanExecutor
        from src.tennis_scene.pipeline.storage.clip_store import ClipStore
        from src.tennis_scene.scripts.generate_dataset import _compose_pipeline_config
        from src.utils.configuration import PathRole

        def forbidden(*args, **kwargs):
            raise AssertionError('CPU command validation must not construct/load models or execute a plan')

        with ExitStack() as guards:
            for target in ('torch.cuda.init', 'torch.cuda._lazy_init', 'torch.nn.Module.__init__', 'torch.ops.load_library'):
                guards.enter_context(patch(target, side_effect=forbidden))
            guards.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
            guards.enter_context(patch.object(BaseInferenceModel, 'load', side_effect=forbidden))
            guards.enter_context(patch.object(InferenceRuntime, '__enter__', side_effect=forbidden))
            guards.enter_context(patch.object(PlanExecutor, 'run', side_effect=forbidden))
            with initialize_config_dir(config_dir=str(REPO / 'src/tennis_scene/configs'), version_base='1.3'):
                cli = compose(config_name='run_video_batch', overrides=arguments)
                batch = parse_video_batch_config(cli)
                _, cfg = _compose_pipeline_config(batch.generation, config_name=batch.pipeline_config)
            assert batch.generation.clip_ids is None
            assert cfg.people.runtime.dino_detector.precision == cfg.people.runtime.vitpose.precision == 'fp16'
            assert cfg.player_association.players_per_side == 2
            assert not cfg.reuse_clip_embeddings
            assert cfg.video_court.action == 'load' and cfg.video_court_side is None
            assert cfg.court_side_evidence == 'player' and cfg.omit_ball_components
            assert not cfg.enabled['ball_detection'] and not cfg.enabled['ball_reconstruction']
            assert 'ball' not in enabled_model_assets(cfg)
            before = markers(batch.generation.dataset_directory)
            dataset_id, prepared = prepare_video_inputs(batch.generation.dataset_directory, batch.video_id)
            assert dataset_id == 'meiji_4cam' and len(prepared) == 69
            frames = sum(clip.media.source.num_frames for clip in prepared)
            assert frames == 46228
            orchestrator = TennisSceneOrchestrator.from_runtime_config(cfg)
            court = orchestrator.prepare_video_court()
            assert court is not None
            for clip in prepared:
                for video in clip.media.source.videos:
                    cfg.resolver.validate(PathRole.DATA, video.path)
                court.bind(clip.media.source)
            side = orchestrator.prepare_video_court_side(court)
            assert side is None
            identity = orchestrator.publication_identity()
            assert 'video_court_side_digest' not in identity
            dtypes = Counter()
            stages = Counter()
            with tempfile.TemporaryDirectory(prefix='video-batch-plan-') as temporary:
                clips = []
                for clip in prepared:
                    source = clip.media.source
                    nodes = standard_definition(cfg, source, code_identity=orchestrator.code_identity,
                                                video_court=court, video_court_side=side)
                    for node in nodes:
                        stages[node.io.name] += 1
                        specs = getattr(node.component, 'model_specs', lambda: {})()
                        if node.io.name == 'person_detection':
                            assert specs['detector'].dtype == 'float16'
                            assert 'extension' in specs['detector'].weights
                            dtypes['dino.float16'] += 1
                        elif node.io.name == 'person_pose_features':
                            assert specs['pose'].dtype == 'float16'
                            dtypes['vitpose.float16'] += 1
                        elif node.io.name == 'person_tracking':
                            assert not specs  # Main runs tracking from stored pose/appearance features.
                            dtypes['tracking.cpu'] += 1
                        elif node.io.name == 'pose_estimation':
                            assert not specs  # Pose is reused from the feature tracker in main.
                            dtypes['pose.evidence_reuse'] += 1
                        elif node.io.name == 'court_side':
                            assert isinstance(node.component, PlayerCourtSideModule)
                            assert node.require_component is None
                            assert set(node.bindings.values()) == {'court_calibration', *(f'player_selection/{c}' for c in source.camera_ids)}
                        assert not node.name.startswith('ball_')
                        assert all(spec.kind != 'ball' for spec in specs.values())
                    store = ClipStore(Path(temporary) / source.clip_id, json_value(source), memory_entries=0)
                    clips.append(ClipExecution(source.clip_id, nodes, store, cfg.cache_overwrite))
                plan = ExecutionPlan(clips)
                # Four cameras: 4 court + calibration + 6 people stages/camera
                # + 9 clip stages (including separated HMR2 features), no ball.
                assert len(plan.nodes) == 69 * (4 + 1 + 6 * 4 + 9)
                assert not plan.references and not plan.statuses
                assert len(plan.clips) == 69
            assert markers(batch.generation.dataset_directory) == before
            assert not torch.cuda.is_initialized()
            report = {
                'command_sha256': sha(COMMAND),
                'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'origin_main': subprocess.check_output(['git', 'rev-parse', 'origin/main'], text=True).strip(),
                'code_identity': orchestrator.code_identity,
                'dataset_id': dataset_id, 'clips': len(prepared), 'frames_per_camera': frames,
                'camera_ids': list(prepared[0].media.source.camera_ids), 'plan_count': 1,
                'nodes': len(plan.nodes), 'stages': dict(stages), 'dtypes': dict(dtypes),
                'court_side_mode': 'clip', 'court_side_evidence': cfg.court_side_evidence,
                'ball_detection_enabled': cfg.enabled['ball_detection'],
                'ball_reconstruction_enabled': cfg.enabled['ball_reconstruction'],
                'court_digest': court.digest, 'side_digest': None,
                'dino_extension': environment['TENNIS_LAB_DINO_EXTENSION'],
                'dino_extension_sha256': sha(Path(environment['TENNIS_LAB_DINO_EXTENSION'])),
                'completed_markers_unchanged': len(before), 'model_factories_called': 0,
                'native_binaries_loaded': 0, 'plan_executed': False, 'cuda_initialized': False,
                'elapsed_seconds': time.perf_counter() - started,
                'limits': 'Parsing, all input/shared-court admission and plan construction only; player side decisions require future observations and were not executed for all69. Actual inference, export and GPU numerical equivalence were not executed.',
            }
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--report', type=Path, required=True)
    validate(parser.parse_args().report)
