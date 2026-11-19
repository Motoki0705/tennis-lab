"""Replay the real pre-split features on CPU and compare every tracking field."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from src.tasks.person_tracking.archive import load_features
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import PersonPoseFeaturesOutput, PersonAppearanceFeaturesOutput
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingInput, PersonTrackingModule, PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec


def compare(left, right):
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            compare(a, b)
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frames, provenance = load_features(args.baseline / 'pre-split-features.npz')
    offsets = np.array([0, *np.cumsum([len(frame.rows) for frame in frames])], np.int64)
    rows = np.concatenate([frame.rows for frame in frames])
    detections = PersonDetectionOutput('cam0', offsets, np.concatenate([frame.boxes for frame in frames]),
        np.concatenate([frame.scores for frame in frames]), rows)
    poses = PersonPoseFeaturesOutput('cam0', offsets, rows, np.concatenate([frame.require_poses() for frame in frames]))
    appearance = PersonAppearanceFeaturesOutput('cam0', offsets, rows,
        np.concatenate([frame.embeddings for frame in frames]), np.concatenate([frame.appearance_valid for frame in frames]),
        'clipreid_vitb16_market1501')
    with initialize_config_dir(version_base='1.3', config_dir=str(Path('src/tennis_scene/configs').resolve())):
        cfg = PipelineRuntimeConfig.from_config(compose(config_name='pipeline'), bind_inputs=False)
    serialized = json.loads((args.baseline / 'pre-split-tracks/artifact.json').read_text())
    expected = ArtifactCodec(PersonTrackingOutput).load(serialized['payload'], args.baseline / 'pre-split-tracks', serialized['arrays'])
    video = SourceVideo('cam0', Path('not-opened.mp4'), provenance['source'], len(frames), 60000 / 1001, 1920, 1080)
    actual = PersonTrackingModule(cfg.tracking, aflink_checkpoint=cfg.aflink_checkpoint, show_progress=False).process(
        PersonTrackingInput(video, detections, poses, appearance))
    compare(asdict(actual), asdict(expected))
    summary = {'frames': len(frames), 'detection_rows': len(rows), 'all_tracking_fields_exact': True,
        'track_ids': actual.track_ids.tolist(), 'real_observations': int(actual.observed.sum()),
        'gsi_interpolated': int(actual.reconstruction.interpolated.sum()), 'cuda_visible_devices': ''}
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
