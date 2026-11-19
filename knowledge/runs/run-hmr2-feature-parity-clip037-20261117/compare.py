"""Compare the pinned C3 body components with C6 using one qualified main clip."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

import base_gvhmr
import base_placement
import c1_recorder as recording
from cuda_guard import guard
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.body_placement import BodyPlacementModule, BodyPlacementOutput
from src.tennis_scene.pipeline.components.body_view_selection import BodyViewSelectionOutput
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import CourtCalibrationOutput
from src.tennis_scene.pipeline.components.gvhmr import GVHMRInput, GVHMRModule, GVHMROutput
from src.tennis_scene.pipeline.components.hmr2_features import HMR2FeaturesInput, HMR2FeaturesModule, HMR2FeaturesOutput
from src.tennis_scene.pipeline.components.identity import PlayerIdentitiesOutput
from src.tennis_scene.pipeline.components.triangulation import PlayerTriangulationOutput
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime
from src.tennis_scene.pipeline.input_assembly.body import BodyPlacementInputAssembler
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256

TOLERANCES = {
    'parameters_rad_and_betas': {'atol': 1e-5, 'rtol': 1e-5},
    'positions_vertices_transl_m': {'atol': 1e-4, 'rtol': 1e-5},
    'reprojection_px': {'atol': 1e-3, 'rtol': 1e-5},
}


def tolerance_group(path: str) -> str:
    if 'reprojection' in path:
        return 'reprojection_px'
    if any(k in path for k in ('position', 'vertices', 'transl')):
        return 'positions_vertices_transl_m'
    return 'parameters_rad_and_betas'


def compare(left: Any, right: Any, path: str, differences: dict[str, Any]) -> None:
    if is_dataclass(left):
        if [f.name for f in fields(left)] != [f.name for f in fields(right)]:
            raise AssertionError(f'{path}: dataclass fields changed')
        for field in fields(left):
            compare(getattr(left, field.name), getattr(right, field.name), f'{path}.{field.name}', differences)
    elif isinstance(left, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f'{path}: dtype or shape changed')
        if np.issubdtype(left.dtype, np.floating):
            group = tolerance_group(path)
            np.testing.assert_allclose(left, right, **TOLERANCES[group], equal_nan=True, err_msg=path)
            finite = np.isfinite(left) & np.isfinite(right)
            differences[path] = {'max_abs': float(np.max(np.abs(left[finite].astype(float) - right[finite]), initial=0)),
                'elements': int(left.size), 'tolerance': group}
        else:
            np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f'{path}: sequence length changed')
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            compare(a, b, f'{path}[{i}]', differences)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f'{path}: mapping keys changed')
        for key in left:
            compare(left[key], right[key], f'{path}.{key}', differences)
    elif isinstance(left, (float, np.floating)):
        group = tolerance_group(path)
        np.testing.assert_allclose(left, right, **TOLERANCES[group], equal_nan=True, err_msg=path)
        differences[path] = {'max_abs': float(abs(left - right)) if np.isfinite(left) and np.isfinite(right) else 0.,
            'elements': 1, 'tolerance': group}
    elif left != right:
        raise AssertionError(f'{path}: scalar changed: {left!r} != {right!r}')


def persist(value: Any, directory: Path) -> Any:
    directory.mkdir(parents=True, exist_ok=False)
    codec = ArtifactCodec(type(value))
    payload, arrays = codec.dump(value, directory)
    (directory / 'descriptor.json').write_text(json.dumps({'payload': payload, 'arrays': arrays}, indent=2))
    return codec.load(payload, directory, arrays)


def configuration(main_config: Path) -> PipelineRuntimeConfig:
    with initialize_config_dir(version_base='1.3', config_dir=str(Path.cwd() / 'src/tennis_scene/configs')):
        cfg = compose(config_name='pipeline', overrides=['device=cuda', 'player_association.players_per_side=2'])
    original = OmegaConf.load(main_config)
    for section in ('people_models', 'player_reconstruction', 'gvhmr', 'frame_sampling', 'person_observations', 'player_association'):
        if OmegaConf.to_container(cfg[section], resolve=True) != OmegaConf.to_container(original[section], resolve=True):
            raise AssertionError(f'Main and comparison settings disagree: {section}')
    return PipelineRuntimeConfig.from_config(cfg, bind_inputs=False)


def load_inputs(store: ClipStore) -> tuple[AssemblyContext, dict[str, Any], BodyViewSelectionOutput, GVHMROutput, BodyPlacementOutput]:
    document = store.source
    source = ClipSource(document['clip_id'], tuple(SourceVideo(**{**v, 'path': Path(v['path'])}) for v in document['videos']))
    def load(name: str, cls: Any, schema: str, version: int) -> Any:
        ref = store.active(name)
        if ref is None or (ref.schema, ref.version) != (schema, version):
            raise ValueError(f'Qualified main output missing or incompatible: {name}')
        return store.load(ref, ArtifactCodec(cls))
    artifacts = {
        'calibration': load('court_calibration', CourtCalibrationOutput, 'local_court_calibration', 1),
        'alignment': load('camera_alignment', CameraAlignmentOutput, 'aligned_cameras', 1),
        'identities': load('player_association', PlayerIdentitiesOutput, 'person_identities', 3),
        'skeleton': load('player_triangulation', PlayerTriangulationOutput, 'player_skeletons', 1),
    }
    for camera in source.camera_ids:
        artifacts[f'pose_{camera}'] = load(f'pose_estimation/{camera}', ObjectObservations, 'person_poses', 1)
    selection = load('body_view_selection', BodyViewSelectionOutput, 'body_view_selection', 1)
    recovered = load('gvhmr', GVHMROutput, 'body_parameters', 1)
    placed = load('body_placement', BodyPlacementOutput, 'placed_bodies', 1)
    if not selection.selections or not any(s.requests for s in selection.selections):
        raise ValueError('Main body selection is empty')
    if not recovered.bodies or placed.players is None or not placed.players.smpl_valid.any():
        raise ValueError('Main did not produce valid body placement')
    return AssemblyContext(source), artifacts, selection, recovered, placed


def seed() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def infer(component: Any, inputs: Any, name: str) -> Any:
    key = ('qualified_clip', name)
    with InferenceRuntime({key: component.model_specs()}, model_event=recording.model_event) as runtime:
        with runtime.node(key) as models, recording.span('inference', operation=name, device='cuda'):
            return component.process_with_models(inputs, models)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-store', type=Path, required=True)
    parser.add_argument('--main-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / 'frozen_main_store'
    if not frozen.exists():
        shutil.copytree(args.main_store, frozen)
    source_data = json.loads((frozen / 'scene.json').read_text())['source']
    store = ClipStore(frozen, source_data, memory_entries=0)
    context, artifacts, selection, main_recovered, main_placed = load_inputs(store)
    for video in context.source.videos:
        if dual_sha256(video.path) != video.sha256:
            raise ValueError(f'Qualified video content changed: {video.camera_id}')
    cfg = configuration(args.main_config)
    # Verify that the saved old components are the exact stated C3 base sources.
    ref = 'ab638fb2ab825e84275863552da3f47473e390ed'
    for name, path in [('base_gvhmr.py', 'src/tennis_scene/pipeline/components/gvhmr.py'),
                       ('base_placement.py', 'src/tennis_scene/pipeline/components/body_placement.py')]:
        original = subprocess.check_output(['git', 'show', f'{ref}:{path}'])
        if (Path(__file__).parent / name).read_bytes() != original:
            raise ValueError(f'Original component snapshot differs from pinned base: {name}')
    preflight = {'clip_id': context.source.clip_id, 'base_commit': ref,
        'separated_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'settings_match_main': True, 'players_per_side': 2, 'tolerances': TOLERANCES,
        'ids_frames_masks_rejection_codes': 'exact', 'seed_per_body_path': 42,
        'source_index_sha256': dual_sha256(frozen / 'scene.json'),
        'body_assets': {k: {'path': str(p), 'sha256': dual_sha256(p)} for k, p in cfg.people.body_assets().items()},
        'layout': [{'person_id': s.person_id, 'camera_id': s.camera_id, 'frames': [r.source_frames.tolist() for r in s.requests]} for s in selection.selections],
        'main_valid_placed_frames': int(main_placed.players.smpl_valid.sum()), 'preflight_only': args.preflight_only}
    (args.output / 'preflight.json').write_text(json.dumps(preflight, indent=2))
    if args.preflight_only:
        print(json.dumps({'main_valid_placed_frames': preflight['main_valid_placed_frames'], 'clip_id': context.source.clip_id}), flush=True)
        return
    torch.cuda.set_per_process_memory_fraction(10 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    seed()
    with recording.InferenceRecorder(args.output / 'base_recorder.json', devices=['cuda'], metadata={'phase': 'C3 base'}):
        before = infer(base_gvhmr.GVHMRModule(cfg.people), base_gvhmr.GVHMRInput(selection), 'base_gvhmr')
        before = persist(before, args.output / 'base_parameters')
        artifacts['recovered'] = before
        p = BodyPlacementInputAssembler(cfg.human_vis_threshold).assemble(context, artifacts)
        base_input = base_placement.BodyPlacementInput(**{f.name: getattr(p, f.name) for f in fields(p)})
        old_placed = infer(base_placement.BodyPlacementModule(context.source.camera_ids, cfg.people, cfg.player_placement,
            reprojection_px=cfg.player_reprojection_px), base_input, 'base_placement')
        old_placed = persist(old_placed, args.output / 'base_placement')
    seed()
    with recording.InferenceRecorder(args.output / 'separated_recorder.json', devices=['cuda'], metadata={'phase': 'C6 separated'}):
        images = infer(HMR2FeaturesModule(cfg.people), HMR2FeaturesInput(context.source, selection), 'hmr2_features')
        features_store = ClipStore(args.output / 'separated_store', json_value(context.source), memory_entries=0)
        sel_ref = features_store.publish('body_view_selection', selection, ArtifactCodec(BodyViewSelectionOutput),
            schema='body_view_selection', version=1, identity={'main_selection': json_value(store.active('body_view_selection'))}, dependencies={}, provenance={'origin': 'acceptance_copy'})
        image_ref = features_store.publish('hmr2_features', images, ArtifactCodec(HMR2FeaturesOutput),
            schema='hmr2_features', version=1, identity={'head': preflight['separated_commit']}, dependencies={'selection': sel_ref}, provenance={'origin': 'acceptance'})
        del images
        images = ClipStore(features_store.root, json_value(context.source), memory_entries=0).load(image_ref, ArtifactCodec(HMR2FeaturesOutput))
        after = infer(GVHMRModule(cfg.people), GVHMRInput(context.source, selection, images), 'separated_gvhmr')
        after = persist(after, args.output / 'separated_parameters')
        artifacts['recovered'] = after
        placed_input = BodyPlacementInputAssembler(cfg.human_vis_threshold).assemble(context, artifacts)
        new_placed = infer(BodyPlacementModule(context.source.camera_ids, cfg.people, cfg.player_placement,
            reprojection_px=cfg.player_reprojection_px), placed_input, 'separated_placement')
        new_placed = persist(new_placed, args.output / 'separated_placement')
    differences = {}
    compare(main_recovered, before, 'main_vs_base.gvhmr', differences)
    compare(main_placed, old_placed, 'main_vs_base.placement', differences)
    compare(before, after, 'base_vs_separated.gvhmr', differences)
    compare(old_placed, new_placed, 'base_vs_separated.placement', differences)
    if new_placed.players is None or not new_placed.players.smpl_valid.any():
        raise AssertionError('Separated path has no valid body placement')
    lifecycle = {}
    for name in ['base', 'separated']:
        r = json.loads((args.output / f'{name}_recorder.json').read_text())
        counts = Counter(f"{e['stage']}:{e['attributes']['model']}" for e in r['events'] if e['stage'] in ['model_load', 'model_unload'])
        expected = {f'model_{a}:{m}': 1 for a in ['load', 'unload'] for m in ['hmr2', 'gvhmr', 'body_geometry']}
        if dict(counts) != expected:
            raise AssertionError(f'Unexpected stage load counts: {name}: {counts}')
        lifecycle[name] = dict(counts)
    receipt = {'accepted': True, **preflight, 'differences': differences, 'lifecycle': lifecycle,
        'separated_valid_placed_frames': int(new_placed.players.smpl_valid.sum()),
        'max_abs': max((x['max_abs'] for x in differences.values()), default=0),
        'inputs_unchanged': all(dual_sha256(v.path) == v.sha256 for v in context.source.videos)}
    if not receipt['inputs_unchanged']:
        raise ValueError('Input video changed during comparison')
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in ['accepted', 'clip_id', 'main_valid_placed_frames', 'separated_valid_placed_frames', 'max_abs', 'lifecycle']}), flush=True)


if __name__ == '__main__':
    import sys
    if '--preflight-only' in sys.argv:
        main()
    else:
        output = Path(sys.argv[sys.argv.index('--output') + 1])
        with guard(output, 'base versus separated body inference'):
            main()
