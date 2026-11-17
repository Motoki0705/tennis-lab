"""Replay C1's stored observations on CPU, preserving its one frozen clip."""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
from dataclasses import replace
from hydra import initialize_config_dir, compose
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo, AssemblyContext
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.components.ball_points import BallPointsOutput
from src.tennis_scene.pipeline.components.court_calibration import CourtCalibrationOutput
from src.tennis_scene.pipeline.components.court_side import CourtSideModule
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionOutput
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.input_assembly.reconstruction import CourtSideInputAssembler
from src.tennis_scene.pipeline.artifacts import json_value
from src.utils.checksum import dual_sha256

ROOT = Path.cwd()
OUT = ROOT / 'outputs/hmr2_feature_parity'
C1 = ROOT.parents[2] / '.worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/datasets/repeat-1/videos/video_000/clips/clip_035/annotations/tennis_scene'
# root is .../.claude/worktrees/<task>; parents[2] is the main repository.

def configuration(device='cpu'):
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'src/tennis_scene/configs')):
        cfg = compose(config_name='pipeline', overrides=[f'device={device}', 'frame_sampling.max_frames=2048'])
    return PipelineRuntimeConfig.from_config(cfg)

def inputs():
    index = json.loads((C1 / 'scene.json').read_text())
    source_data = index['source']
    source = ClipSource(source_data['clip_id'], tuple(SourceVideo(**{**v, 'path': Path(v['path'])}) for v in source_data['videos']))
    store = ClipStore(OUT / 'c1_observations', source_data, memory_entries=0)
    ctx = AssemblyContext(source)
    artifacts = {'calibration': store.load(store.active('court_calibration'), ArtifactCodec(CourtCalibrationOutput))}
    for camera in source.camera_ids:
        for port, node, cls in [('ball_', 'ball_points', BallPointsOutput), ('pose_', 'pose_estimation', ObjectObservations), ('tracks_', 'player_selection', PlayerSelectionOutput)]:
            artifacts[port + camera] = store.load(store.active(f'{node}/{camera}'), ArtifactCodec(cls))
    return ctx, artifacts

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = json.loads((C1 / 'scene.json').read_text())
    for video in index['source']['videos']:
        if dual_sha256(Path(video['path'])) != video['sha256']:
            raise ValueError('C1 source video content changed')
    if not (OUT / 'c1_observations').exists():
        shutil.copytree(C1, OUT / 'c1_observations')
    cfg = configuration()
    ctx, artifacts = inputs()
    data = {'clip_id': ctx.source.clip_id, 'source': json_value(ctx.source),
        'source_index_sha256': dual_sha256(C1 / 'scene.json'),
        'input_sampling_max_frames': cfg.sampling_max_frames,
        'original_sampling_max_frames': 1024,
        'reason_for_explicit_validation_override': 'C1 stopped at 1055 sampled frames; replay keeps its complete grid and changes only the validation capacity, equally for both body paths.',
        'acceptance_scope': 'HMR2/GVHMR/body placement boundary parity; not C1 full-pipeline completion or speed benchmark.'}
    (OUT / 'input.json').write_text(json.dumps(data, indent=2))
    side = CourtSideModule(ctx.source.camera_ids, cfg.court_side, max_frames=cfg.sampling_max_frames).process(CourtSideInputAssembler(0.).assemble(ctx, artifacts))
    codec = ArtifactCodec(type(side))
    payload, arrays = codec.dump(side, OUT)
    (OUT / 'side.json').write_text(json.dumps({'payload': payload, 'arrays': arrays}, indent=2))
    print('CPU side ready', side.camera_ids, flush=True)

if __name__ == '__main__':
    main()
