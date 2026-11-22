"""Measure C9 on frozen #1752 prefix artifacts using the production CLIP stages."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

from src.tasks.person_tracking.sequence import CLIP_ENCODER
from src.tasks.player_association.appearance.encoders import (
    build_encoder,
    encoder_weights,
)
from src.tasks.player_association.appearance.sampling import (
    DEFAULT_BATCH_SIZE,
    CropSamplingConfig,
)
from src.tasks.player_association.association.associate import AssociationConfig
from src.tasks.player_association.association.config import load_association_config
from src.tennis_scene.pipeline.artifacts import (
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.clip_inputs import SourceInputs, prepare_clip_source
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import CourtSideOutput
from src.tennis_scene.pipeline.components.identity import (
    PlayerAssociationModule,
    PlayerIdentitiesOutput,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonAppearanceFeaturesInput,
    PersonAppearanceFeaturesModule,
)
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionOutput
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    PlayerAssociationInputAssembler,
)
from src.tennis_scene.pipeline.model_assets import file_identity
from src.tennis_scene.pipeline.model_resources import appearance_model
from src.tennis_scene.pipeline.storage.codec import (
    ArtifactCodec,
    restore_type,
    unpack_value,
)
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'run-hmr2-feature-parity-20261117'))
from c1_recorder import (  # noqa: E402
    InferenceRecorder,
    model_event,
    recording_context,
    span,
)

DEVICE = 'cuda:0'
COSINE_ATOL = 1e-6
DIAGNOSTIC_ATOL = 1e-4
MODES = ('recompute', 'reuse', 'reuse', 'recompute')


@dataclass
class FrozenClip:
    inputs: SourceInputs
    artifacts: dict[str, Any]
    receipt: dict[str, Any]
    config: AssociationConfig
    reference: PlayerIdentitiesOutput


def frozen_clip(root: Path, checkpoint_sha256: str) -> FrozenClip:
    index = json.loads((root / 'scene.json').read_text())
    source_doc = index['source']
    source_hash = document_digest(source_doc)
    videos = source_doc['videos']
    inputs = prepare_clip_source(tuple(Path(v['path']) for v in videos), tuple(v['camera_id'] for v in videos),
                                 clip_id=source_doc['clip_id'])
    if source_hash != index['source_sha256'] or json_value(inputs.source) != source_doc:
        raise ValueError(f'Media no longer matches the frozen source for {inputs.source.clip_id}')
    refs = index['artifacts']
    used: dict[str, Any] = {}

    def read(node: str, schema: str, version: int, dependencies: dict[str, str] | None = None) -> Any:
        ref = refs[node]
        desc = read_component_descriptor(root, ref, node=node, source_sha256=source_hash)
        if (ref['schema'], ref['version']) != (schema, version):
            raise ValueError(f'Unexpected frozen schema for {node}')
        if dependencies is not None and desc['dependencies'] != {port: refs[parent] for port, parent in dependencies.items()}:
            raise ValueError(f'Frozen inputs changed for {node}')
        used[node] = ref
        return desc

    def value(desc: Any, node: str) -> Any:
        return unpack_value(desc['payload'], (root / refs[node]['path']).parent, desc['arrays'])

    selected = {f'tracks_{c}': f'player_selection/{c}' for c in inputs.source.camera_ids}
    original = read('player_association', 'person_identities', 3,
                    {'calibration': 'court_calibration', 'side': 'court_side', **selected})
    settings = original['identity']['settings']
    config = load_association_config(players_per_side=settings['config']['players_per_side'])
    if json_value(config) != settings['config'] or json_value(CropSamplingConfig()) != settings['sampling']:
        raise ValueError('Association settings differ from the frozen run')
    if config.appearance is None or config.appearance.encoder != CLIP_ENCODER or settings['assets']['encoder']['sha256'] != checkpoint_sha256:
        raise ValueError('CLIP weight content differs from the frozen association')
    artifacts = {'calibration': restore_type(value(read('court_calibration', 'local_court_calibration', 1), 'court_calibration'),
                                             CourtCalibrationOutput),
                 'side': restore_type(value(read('court_side', 'court_side', 3), 'court_side'), CourtSideOutput)}
    for camera in inputs.source.camera_ids:
        detector, tracker, selection = (f'{stage}/{camera}' for stage in ('person_detection', 'person_tracking', 'player_selection'))
        artifacts[f'detections_{camera}'] = restore_type(value(read(detector, 'person_detections', 2), detector), PersonDetectionOutput)
        read(tracker, 'person_tracks', 5, {'detections': detector})
        desc = read(selection, 'selected_player_tracks', 2, {'tracks': tracker, 'calibration': 'court_calibration'})
        payload = value(desc, selection)
        if payload['tracks']['evidence'] is not None:
            # v2 stored valid detection-row addresses. Its appearance had no
            # model identity, and is deliberately excluded from reuse.
            payload['tracks']['evidence']['encoder_identity'] = None
        artifacts[f'tracks_{camera}'] = restore_type(payload, PlayerSelectionOutput)
    receipt = {'source': source_doc, 'source_sha256': source_hash, 'store': str(root), 'references': used,
               'association_config': settings['config'], 'sampling': settings['sampling']}
    reference = restore_type(value(original, 'player_association'), PlayerIdentitiesOutput)
    return FrozenClip(inputs, artifacts, receipt, config, reference)


def persist(directory: Path, value: Any) -> Any:
    directory.mkdir(parents=True)
    codec = ArtifactCodec(type(value))
    with span('save'):
        payload, arrays = codec.dump(value, directory)
        write_json_atomic(directory / 'artifact.json', {'payload': payload, 'arrays': arrays})
    with span('artifact_load'):
        document = json.loads((directory / 'artifact.json').read_text())
        return codec.load(document['payload'], directory, document['arrays'])


def compare_diagnostics(a: Any, b: Any, field: str) -> float:
    if type(a) is not type(b):
        raise ValueError(f'Diagnostic type changed: {field}')
    if isinstance(a, dict) and a.keys() == b.keys():
        return max((compare_diagnostics(a[k], b[k], f'{field}.{k}') for k in a), default=0.)
    if isinstance(a, list) and len(a) == len(b):
        return max((compare_diagnostics(x, y, f'{field}[{i}]') for i, (x, y) in enumerate(zip(a, b, strict=True))), default=0.)
    if isinstance(a, float):
        delta = abs(a - b)
        if not delta <= (COSINE_ATOL if field.endswith('.cosine') else DIAGNOSTIC_ATOL):
            raise ValueError(f'Diagnostic values differ outside tolerance: {field} {a} {b}')
        return delta
    if a != b:
        raise ValueError(f'Discrete diagnostic changed: {field}')
    return 0.


def sampling_record(result: PlayerIdentitiesOutput) -> dict[str, Any]:
    return {camera: [dict(track_id=t['track_id'], samples=t['samples'], rejected=t['rejected'],
        frames=sorted([*t['reused_frames'], *(f for v in t['recomputed_frames'].values() for f in v)])) for t in tracks]
        for camera, tracks in result.diagnostics['appearance']['tracks'].items()}


class MeasuredEncoder:
    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self.name, self.input_size = encoder.name, encoder.input_size

    def embed(self, crops: torch.Tensor) -> torch.Tensor:
        with span('inference', device=DEVICE, model=self.name, batch_shape=tuple(crops.shape), batch_capacity=DEFAULT_BATCH_SIZE):
            return self.encoder.embed(crops)


def run(clips: list[FrozenClip], output: Path, weights: Path, checkpoint_root: Path, external_root: Path) -> dict[str, Any]:
    torch.cuda.set_per_process_memory_fraction(10 * 1024**3 / torch.cuda.get_device_properties(DEVICE).total_memory, DEVICE)
    spec = appearance_model(CLIP_ENCODER, weights, DEVICE,
                            lambda: MeasuredEncoder(build_encoder(CLIP_ENCODER, checkpoint=weights, device=DEVICE,
                                checkpoint_root=checkpoint_root, external_root=external_root)))
    producer = PersonAppearanceFeaturesModule(spec, capture_frame_sha256=True)
    declarations = {(clip.inputs.source.clip_id, stage): {'encoder': spec} for clip in clips for stage in
        (*[f'features/{c}' for c in clip.inputs.source.camera_ids], *[str(i) for i in range(len(MODES))])}
    measurements: dict[str, Any] = {}
    recorder = InferenceRecorder(output / 'recorder.json', devices=(DEVICE,), metadata={'modes': list(MODES)})
    with recorder, InferenceRuntime(declarations, model_event=model_event) as runtime:
        for clip in clips:
            source, artifacts = clip.inputs.source, dict(clip.artifacts)
            name = source.clip_id.rsplit('/', 1)[-1]
            print(f'{source.clip_id}: prepare CLIP features', flush=True)
            for video in source.videos:
                camera = video.camera_id
                with recording_context(clip_id=source.clip_id, camera=camera, mode='prepare'), runtime.node((source.clip_id, f'features/{camera}')) as model:
                    with span('node', stage_id='person_appearance_features'):
                        features = producer.process_with_models(PersonAppearanceFeaturesInput(video, artifacts[f'detections_{camera}']), model)
                    artifacts[f'appearance_{camera}'] = persist(output / 'artifacts' / name / camera, features)
            baseline = None
            max_delta = 0.
            for i, mode in enumerate(MODES):
                print(f'{source.clip_id}: {mode}, comparison {i + 1}', flush=True)
                reuse = mode == 'reuse'
                module = PlayerAssociationModule(source.camera_ids, clip.config, sampling=CropSamplingConfig(),
                    encoder=spec, device=DEVICE, enabled=True, reuse_clip_embeddings=reuse)
                request = PlayerAssociationInputAssembler(reuse).assemble(AssemblyContext(source), artifacts)
                with recording_context(clip_id=source.clip_id, mode=mode, trial=i), runtime.node((source.clip_id, str(i))) as model:
                    with span('node', stage_id='player_association') as attributes:
                        result = module.process_with_models(request, model)
                        appearance = result.diagnostics['appearance']
                        attributes.update({key: appearance[key] for key in ('reused_samples', 'recomputed_samples', 'recomputed_by_reason')})
                    result = persist(output / 'artifacts' / name / f'{i}-{mode}', result)
                if baseline is None:
                    baseline = result
                np.testing.assert_array_equal(result.local_track_ids, baseline.local_track_ids)
                np.testing.assert_array_equal(result.player_ids, baseline.player_ids)
                if sampling_record(result) != sampling_record(baseline):
                    raise ValueError('Sampled frames or rejected diagnostics differ')
                max_delta = max(max_delta, compare_diagnostics(baseline.diagnostics['association'], result.diagnostics['association'], 'association'))
            clip.inputs.snapshot.assert_unchanged()
            assert baseline is not None
            measurements[source.clip_id] = {'max_diagnostic_delta': max_delta, 'sampled_frames_and_rejections_equal': True,
                'player_ids_equal': True, 'association_decisions_and_appearance_mask_equal': True,
                'frozen_player_ids_equal': bool(np.array_equal(baseline.player_ids, clip.reference.player_ids))}
    for clip_id, summary in measurements.items():
        events = [e for e in recorder.events if e['attributes'].get('clip_id') == clip_id]
        node_events = [e for e in events if e['stage'] == 'node' and e['attributes']['stage_id'] == 'player_association']
        inference_events = [e for e in events if e['stage'] == 'inference' and e['attributes']['mode'] in MODES]
        trial_results = []
        for event in node_events:
            attributes = event['attributes']
            seconds = (event['end_ns'] - event['start_ns']) / 1e9
            batches = [e for e in inference_events if e['attributes']['trial'] == attributes['trial']]
            inference_seconds = sum(e['end_ns'] - e['start_ns'] for e in batches) / 1e9
            batches_used = [e['attributes']['batch_items'] for e in batches]
            if sum(batches_used) != attributes['recomputed_samples']:
                raise ValueError('Measured CLIP batches disagree with recomputation diagnostics')
            trial_results.append({**attributes, 'association_seconds': seconds, 'inference_seconds': inference_seconds, 'batches': batches_used})
        summary['trials'] = trial_results
        summary['seconds_saved'] = {key: median(t[key] for t in trial_results if t['mode'] == 'recompute')
                                   - median(t[key] for t in trial_results if t['mode'] == 'reuse')
                                   for key in ('association_seconds', 'inference_seconds')}
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Frozen #1752 batch dataset')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clips', nargs='+', default=['clip_037', 'clip_000', 'clip_027'])
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    checkpoint_root, external_root = Path('ckpt').resolve(), Path('third_party').resolve()
    weights = encoder_weights(CLIP_ENCODER, checkpoint_root=checkpoint_root, external_root=external_root)
    checkpoint = file_identity(weights)
    clips = [frozen_clip(args.dataset.resolve() / 'videos' / 'video_000' / 'clips' / name / 'annotations' / 'tennis_scene', checkpoint['sha256'])
             for name in args.clips]
    args.output.mkdir(parents=True)
    write_json_atomic(args.output / 'inputs.json', {'checkpoint': checkpoint, 'cosine_atol': COSINE_ATOL,
        'diagnostic_atol': DIAGNOSTIC_ATOL, 'clips': [clip.receipt for clip in clips]})
    if not args.check_only:
        write_json_atomic(args.output / 'comparison.json', run(clips, args.output, weights, checkpoint_root, external_root))


if __name__ == '__main__':
    main()
