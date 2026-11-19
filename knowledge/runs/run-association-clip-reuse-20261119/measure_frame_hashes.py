"""Measure isolated per-frame proof hashing with the completed C9 artifacts."""

import argparse
import json
from pathlib import Path
from time import perf_counter_ns, thread_time_ns

import numpy as np

from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.components.person_features import (
    PersonAppearanceFeaturesOutput,
    appearance_image_digest,
)
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.video import OpenCVVideoFrameReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    result = {}
    inputs = json.loads((args.run_dir / 'inputs.json').read_text())
    for clip in inputs['clips']:
        source = clip['source']
        clip_name = source['clip_id'].rsplit('/', 1)[-1]
        total, cpu_ns, wall_ns = 0, 0, 0
        for video in source['videos']:
            directory = args.run_dir / 'artifacts' / clip_name / video['camera_id']
            doc = json.loads((directory / 'artifact.json').read_text())
            features = ArtifactCodec(PersonAppearanceFeaturesOutput).load(doc['payload'], directory, doc['arrays'])
            expected = np.flatnonzero(np.diff(features.frame_offsets))
            matched = []
            for packet in OpenCVVideoFrameReader(video['path'], max_frames=video['num_frames']):
                frame = packet.index
                if features.frame_offsets[frame] == features.frame_offsets[frame + 1]:
                    continue
                wall_start, cpu_start = perf_counter_ns(), thread_time_ns()
                digest = appearance_image_digest(packet.frame)
                cpu_ns += thread_time_ns() - cpu_start
                wall_ns += perf_counter_ns() - wall_start
                np.testing.assert_array_equal(digest, features.frame_sha256[frame])
                matched.append(frame)
            np.testing.assert_array_equal(matched, expected)
            total += len(matched)
        result[source['clip_id']] = {'hashed_frames': total, 'hash_cpu_seconds': cpu_ns / 1e9, 'hash_wall_seconds': wall_ns / 1e9}
    write_json_atomic(args.run_dir / 'frame-hash-cost.json', result)


if __name__ == '__main__':
    main()
