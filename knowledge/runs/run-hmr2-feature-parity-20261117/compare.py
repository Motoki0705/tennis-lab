"""One frozen C1 clip, original vs separated body inference; run via GPU queue."""
from __future__ import annotations

import json
import random
import subprocess
from collections import Counter
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import c1_recorder as recording
import unsplit_gvhmr as original
from prepare import OUT, configuration, inputs
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.body_placement import BodyPlacementModule
from src.tennis_scene.pipeline.components.body_view_selection import BodyViewSelectionModule, BodyViewSelectionOutput
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentModule
from src.tennis_scene.pipeline.components.court_side import CourtSideOutput
from src.tennis_scene.pipeline.components.gvhmr import GVHMRInput, GVHMRModule
from src.tennis_scene.pipeline.components.hmr2_features import HMR2FeaturesInput, HMR2FeaturesModule, HMR2FeaturesOutput
from src.tennis_scene.pipeline.components.identity import PlayerAssociationModule
from src.tennis_scene.pipeline.components.triangulation import PlayerTriangulationModule
from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime
from src.tennis_scene.pipeline.input_assembly.body import BodyViewSelectionInputAssembler, BodyPlacementInputAssembler
from src.tennis_scene.pipeline.input_assembly.reconstruction import PlayerAssociationInputAssembler, CameraAlignmentInputAssembler, PlayerTriangulationInputAssembler
from src.tennis_scene.pipeline.model_resources import appearance_model
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tasks.player_association.appearance.encoders import build_encoder
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.utils.checksum import dual_sha256

TOLERANCES = {
    'parameters_rad_and_betas': {'atol': 1e-5, 'rtol': 1e-5},
    'positions_vertices_transl_m': {'atol': 1e-4, 'rtol': 1e-5},
    'reprojection_px': {'atol': 1e-3, 'rtol': 1e-5},
    'ids_frames_masks_rejection_codes': 'exact',
}


def infer(component: Any, value: Any, label: str) -> Any:
    node = ('clip_035', label)
    with InferenceRuntime({node: component.model_specs()}, model_event=recording.model_event) as runtime:
        with runtime.node(node) as models:
            return component.process_with_models(value, models)


def seed() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def persist(value: Any, directory: Path) -> Any:
    directory.mkdir(parents=True, exist_ok=False)
    codec = ArtifactCodec(type(value))
    payload, arrays = codec.dump(value, directory)
    (directory / 'descriptor.json').write_text(json.dumps({'payload': payload, 'arrays': arrays}, indent=2))
    return codec.load(payload, directory, arrays)


def differences(left: Any, right: Any, path: str, result: dict[str, Any]) -> None:
    if is_dataclass(left):
        if [f.name for f in fields(left)] != [f.name for f in fields(right)]:
            raise AssertionError(f'{path}: dataclass fields changed')
        for field in fields(left):
            differences(getattr(left, field.name), getattr(right, field.name), f'{path}.{field.name}', result)
    elif isinstance(left, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f'{path}: dtype/shape changed')
        if np.issubdtype(left.dtype, np.floating):
            group = 'reprojection_px' if 'reprojection' in path else 'positions_vertices_transl_m' if any(p in path for p in ('position', 'vertices', 'transl')) else 'parameters_rad_and_betas'
            np.testing.assert_allclose(left, right, **TOLERANCES[group], equal_nan=True, err_msg=path)
            finite = np.isfinite(left) & np.isfinite(right)
            result[path] = {'max_abs': float(np.max(np.abs(left[finite].astype(float) - right[finite]), initial=0)), 'elements': int(left.size), 'tolerance': group}
        else:
            np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f'{path}: sequence length changed')
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            differences(a, b, f'{path}[{index}]', result)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f'{path}: mapping keys changed')
        for key in left:
            differences(left[key], right[key], f'{path}.{key}', result)
    elif isinstance(left, (float, np.floating)):
        np.testing.assert_allclose(left, right, atol=1e-5, rtol=1e-5, equal_nan=True, err_msg=path)
    elif left != right:
        raise AssertionError(f'{path}: {left!r} != {right!r}')


def main() -> None:
    cfg = configuration('cuda')
    # The queue grants exclusive use. Keep the allocator below the 16 GiB GPU capacity.
    torch.cuda.set_per_process_memory_fraction(10 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    preflight = {
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'tolerances': TOLERANCES, 'seed_before_each_body_path': 42,
        'allocator_cap_bytes': 10 * 1024**3,
        'body_assets': {name: {'path': str(path), 'sha256': dual_sha256(path)} for name, path in cfg.people.body_assets().items()},
        'inputs': json.loads((OUT / 'input.json').read_text()),
    }
    (OUT / 'comparison_preflight.json').write_text(json.dumps(preflight, indent=2))
    ctx, artifacts = inputs()
    for video in ctx.source.videos:
        if dual_sha256(video.path) != video.sha256:
            raise ValueError('Frozen C1 video changed')
    data = json.loads((OUT / 'side.json').read_text())
    artifacts['side'] = ArtifactCodec(CourtSideOutput).load(data['payload'], OUT, data['arrays'])
    encoder = appearance_model(cfg.player_association.appearance.encoder, cfg.association_encoder_weights, cfg.device,
        lambda: build_encoder(cfg.player_association.appearance.encoder, checkpoint_root=cfg.roots.checkpoint_root,
            external_root=cfg.roots.external_asset_root, device=cfg.device))
    association = PlayerAssociationModule(ctx.source.camera_ids, cfg.player_association, sampling=CropSamplingConfig(), device=cfg.device, encoder=encoder, enabled=True)
    with recording.InferenceRecorder(OUT / 'upstream_recorder.json', devices=['cuda'], metadata={'phase': 'C1 saved 2D replay'}):
        artifacts['identities'] = infer(association, PlayerAssociationInputAssembler().assemble(ctx, artifacts), 'player_association')
        print('association ready', flush=True)
        alignment = CameraAlignmentModule(ctx.source.camera_ids, cfg.camera_geometry,
            player_reprojection_px=cfg.player_reprojection_px, ball_reprojection_px=cfg.ball_reprojection_px,
            joint_confidence=cfg.joint_confidence, max_frames=cfg.sampling_max_frames)
        artifacts['alignment'] = alignment.process(CameraAlignmentInputAssembler(cfg.human_vis_threshold, 0.).assemble(ctx, artifacts))
        print('alignment ready', flush=True)
        triangulate = PlayerTriangulationModule(ctx.source.camera_ids, reprojection_px=cfg.player_reprojection_px, joint_confidence=cfg.joint_confidence)
        artifacts['skeleton'] = triangulate.process(PlayerTriangulationInputAssembler(cfg.human_vis_threshold).assemble(ctx, artifacts))
        selection = BodyViewSelectionModule(ctx.source.camera_ids, max_frames=cfg.sampling_max_frames).process(BodyViewSelectionInputAssembler(cfg.human_vis_threshold).assemble(ctx, artifacts))
    if not selection.selections or not any(s.requests for s in selection.selections):
        raise AssertionError('No real selected body segments: parity acceptance cannot succeed')
    selection = persist(selection, OUT / 'body_selection')
    print('body inputs', [(s.person_id, s.camera_id, [len(r.source_frames) for r in s.requests]) for s in selection.selections], flush=True)
    baseline_input = original.GVHMRInput(selection)
    seed()
    with recording.InferenceRecorder(OUT / 'original_recorder.json', devices=['cuda'], metadata={'phase': 'original'}):
        before = infer(original.GVHMRModule(cfg.people), baseline_input, 'gvhmr_original')
        before = persist(before, OUT / 'original_parameters')
        artifacts['recovered'] = before
        placement = BodyPlacementModule(ctx.source.camera_ids, cfg.people, cfg.player_placement, reprojection_px=cfg.player_reprojection_px)
        old_placed = infer(placement, BodyPlacementInputAssembler(cfg.human_vis_threshold).assemble(ctx, artifacts), 'placement_original')
        old_placed = persist(old_placed, OUT / 'original_placement')
    seed()
    with recording.InferenceRecorder(OUT / 'separated_recorder.json', devices=['cuda'], metadata={'phase': 'separated'}):
        images = infer(HMR2FeaturesModule(cfg.people), HMR2FeaturesInput(ctx.source, selection), 'hmr2_features')
        feature_store = ClipStore(OUT / 'split_store', json_value(ctx.source), memory_entries=0)
        selection_ref = feature_store.publish('body_view_selection', selection, ArtifactCodec(BodyViewSelectionOutput),
            schema='body_view_selection', version=1, identity={'input': preflight['inputs']['source_index_sha256']}, dependencies={}, provenance={'origin': 'acceptance'})
        ref = feature_store.publish('hmr2_features', images, ArtifactCodec(HMR2FeaturesOutput), schema='hmr2_features', version=1,
            identity={'head': preflight['head']}, dependencies={'selection': selection_ref}, provenance={'origin': 'acceptance'})
        del images
        # Reopen the real immutable artifact, not an in-memory tensor.
        restarted = ClipStore(feature_store.root, json_value(ctx.source), memory_entries=0)
        images = restarted.load(ref, ArtifactCodec(HMR2FeaturesOutput))
        after = infer(GVHMRModule(cfg.people), GVHMRInput(ctx.source, selection, images), 'gvhmr_separated')
        after = persist(after, OUT / 'separated_parameters')
        artifacts['recovered'] = after
        new_placed = infer(placement, BodyPlacementInputAssembler(cfg.human_vis_threshold).assemble(ctx, artifacts), 'placement_separated')
        new_placed = persist(new_placed, OUT / 'separated_placement')
    diff: dict[str, Any] = {}
    differences(before, after, 'gvhmr', diff)
    differences(old_placed, new_placed, 'placement', diff)
    if old_placed.players is None or not old_placed.players.smpl_valid.any():
        raise AssertionError('No valid placed bodies: no successful placement acceptance')
    lifecycle = {}
    for label in ('original', 'separated'):
        record = json.loads((OUT / f'{label}_recorder.json').read_text())
        events = [e for e in record['events'] if e['stage'] in ('model_load', 'model_unload')]
        lifecycle[label] = dict(Counter(f"{e['stage']}:{e['attributes']['model']}" for e in events))
    receipt = {'accepted': True, 'clip_id': ctx.source.clip_id, 'tolerances': TOLERANCES,
        'body_layout': [{'person_id': s.person_id, 'camera_id': s.camera_id, 'segment_frames': [r.source_frames.tolist() for r in s.requests]} for s in selection.selections],
        'numeric_differences': diff, 'lifecycle': lifecycle,
        'valid_placed_frames': int(old_placed.players.smpl_valid.sum()),
        'max_absolute_error': max((v['max_abs'] for v in diff.values()), default=0.),
        'input_videos_unchanged': all(dual_sha256(v.path) == v.sha256 for v in ctx.source.videos)}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({key: receipt[key] for key in ('accepted', 'clip_id', 'valid_placed_frames', 'max_absolute_error', 'lifecycle')}), flush=True)

if __name__ == '__main__':
    main()
