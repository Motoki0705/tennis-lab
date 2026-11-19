"""Replay final association with saved GPU features and require zero CLIP calls."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from compare import DEVICE, compare_diagnostics, frozen_clip, sampling_record

from src.tasks.person_tracking.sequence import CLIP_ENCODER
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.components.identity import (
    PlayerAssociationModule,
    PlayerIdentitiesOutput,
)
from src.tennis_scene.pipeline.components.person_features import (
    PersonAppearanceFeaturesOutput,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    PlayerAssociationInputAssembler,
)
from src.tennis_scene.pipeline.model_assets import file_identity
from src.tennis_scene.pipeline.model_resources import appearance_model
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec


def load(directory: Path, output_type: type) -> Any:
    document = json.loads((directory / 'artifact.json').read_text())
    return ArtifactCodec(output_type).load(document['payload'], directory, document['arrays'])


def no_encoder() -> Any:
    raise AssertionError('Replaying verified GPU features must not load CLIP')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    inputs = json.loads((args.run_dir / 'inputs.json').read_text())
    weights = Path(inputs['checkpoint']['path'])
    if file_identity(weights) != inputs['checkpoint']:
        raise ValueError('The checkpoint changed since the GPU comparison')
    encoder = appearance_model(CLIP_ENCODER, weights, DEVICE, no_encoder)
    results = {}
    for record in inputs['clips']:
        clip = frozen_clip(Path(record['store']), inputs['checkpoint']['sha256'])
        if clip.receipt != record:
            raise ValueError('A frozen clip or its artifacts changed')
        source = clip.inputs.source
        saved = args.run_dir / 'artifacts' / source.clip_id.rsplit('/', 1)[-1]
        artifacts = dict(clip.artifacts)
        for camera in source.camera_ids:
            artifacts[f'appearance_{camera}'] = load(saved / camera, PersonAppearanceFeaturesOutput)
        component = PlayerAssociationModule(source.camera_ids, clip.config, sampling=CropSamplingConfig(),
            encoder=encoder, device=DEVICE, enabled=True, reuse_clip_embeddings=True)
        request = PlayerAssociationInputAssembler(True).assemble(AssemblyContext(source), artifacts)
        result = component.process(request)
        gpu_reuse, gpu_recompute = (load(saved / mode, PlayerIdentitiesOutput) for mode in ('1-reuse', '0-recompute'))
        if result.diagnostics != gpu_reuse.diagnostics or sampling_record(result) != sampling_record(gpu_recompute):
            raise ValueError('Diagnostics changed since the GPU comparison')
        np.testing.assert_array_equal(result.local_track_ids, gpu_recompute.local_track_ids)
        np.testing.assert_array_equal(result.player_ids, gpu_recompute.player_ids)
        delta = compare_diagnostics(result.diagnostics['association'], gpu_recompute.diagnostics['association'], 'association')
        appearance = result.diagnostics['appearance']
        results[source.clip_id] = {'reused_samples': appearance['reused_samples'], 'recomputed_samples': appearance['recomputed_samples'],
            'max_diagnostic_delta_vs_gpu_recompute': delta, 'exact_replay_of_gpu_reuse': True}
    write_json_atomic(args.run_dir / 'saved-replay.json', results)


if __name__ == '__main__':
    main()
