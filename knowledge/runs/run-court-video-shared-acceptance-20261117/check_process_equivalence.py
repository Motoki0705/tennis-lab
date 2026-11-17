"""Verify process transport on real fit RGB, without reading evaluation truth."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from src.tasks.court_detection.geometry.video_shared.candidates import CourtSeed
from src.tasks.court_detection.geometry.video_shared.fit import SharedPaintConfig
from src.tennis_scene.pipeline.video_court.estimator import _measure_source
from build import save, sha


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--stage1', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = json.loads((args.stage1 / 'all-fit/protocol.json').read_text())
    frames = [r for r in source['frames'] if r['camera_id'] == 'cam0' and r['clip_id'] == 'video_000/clip_001']
    seed = CourtSeed('frozen-rgb-fit-initializer', 'video_000/clip_001', 0, np.array(source['candidates']['cam0'][0]['matrix']), False)
    tasks = [(seed, row['clip_id'], row['frame_index'],
              args.dataset / f'videos/video_000/clips/{row["clip_id"].split("/")[1]}/media/cam0.mp4',
              (1920, 1080), SharedPaintConfig()) for row in frames]
    started = time.monotonic()
    sequential = [_measure_source(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
        parallel = list(pool.map(_measure_source, tasks))
    rows = []
    for task, (left, left_rgb), (right, right_rgb) in zip(tasks, sequential, parallel, strict=True):
        assert left_rgb == right_rgb
        assert json.dumps(left.diagnostic, sort_keys=True) == json.dumps(right.diagnostic, sort_keys=True)
        assert len(left.groups) == len(right.groups) > 0
        for a, b in zip(left.groups, right.groups, strict=True):
            assert (a.clip_id, a.frame_index, a.line_index, a.endings) == (b.clip_id, b.frame_index, b.line_index, b.endings)
            assert a.points.shape == b.points.shape and a.points.dtype == b.points.dtype and a.points.tobytes() == b.points.tobytes()
        rows.append({'clip_id': task[1], 'frame_index': task[2], 'groups': len(left.groups), 'rgb_sha256': left_rgb, 'exact': True})
    save(args.output, {'cases': rows, 'seconds': time.monotonic() - started,
                      'driver_sha256': sha(Path(__file__)), 'estimator_sha256': sha(Path('src/tennis_scene/pipeline/video_court/estimator.py')),
                      'reference_h_used': False, 'heldout_frames_used': False})
    print(args.output.read_text())
