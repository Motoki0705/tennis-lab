"""Compare identical-input ball replays and produce masked plots / CPU review video."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, unpack_value
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path, read_component_descriptor
from src.tennis_scene.visualization.review.trajectory import masked_trajectory
from src.utils.checksum import dual_sha256
from src.utils.video.reader import RandomAccessVideoReader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--store', type=Path, required=True)
parser.add_argument('--replay', type=Path, required=True)
parser.add_argument('--repeat', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
index = json.loads((args.store / 'scene.json').read_text())
before = load_scene_result(indexed_scene_path(args.store / 'scene.json'))
after = load_scene_result(args.replay / 'scene.npz')
repeat = load_scene_result(args.repeat / 'scene.npz')
for name in ('ball_uv', 'ball_vis', 'ball_3d', 'ball_3d_valid', 'ball_rejection_code'):
    np.testing.assert_array_equal(getattr(after, name), getattr(repeat, name))
for camera in after.metadata['camera_ids']:
    first = args.replay / f'ball_{camera}'
    second = args.repeat / f'ball_{camera}'
    desc = json.loads((first / 'artifact.json').read_text())
    desc2 = json.loads((second / 'artifact.json').read_text())
    assert desc['replay'] == desc2['replay']
    assert desc['arrays'] == desc2['arrays']
first_manifest = json.loads((args.replay / 'replay.json').read_text())
second_manifest = json.loads((args.repeat / 'replay.json').read_text())
assert first_manifest == second_manifest
ref = index['artifacts']['camera_alignment']
descriptor = read_component_descriptor(args.store, ref, node='camera_alignment', source_sha256=index['source_sha256'])
alignment = ArtifactCodec(CameraAlignmentOutput).load(descriptor['payload'], (args.store / ref['path']).parent, descriptor['arrays'])
common = before.ball_3d_valid & after.ball_3d_valid
raw_errors = [[], []]
for camera in alignment.geometry.cameras:
    view = after.metadata['camera_ids'].index(camera.camera_id)
    mask = common & before.ball_vis[view] & after.ball_vis[view]
    raw = before.ball_uv[view] * [before.width, before.height]
    for target, xyz in zip(raw_errors, (before.ball_3d, after.ball_3d), strict=True):
        target.extend(np.linalg.norm(camera.project(xyz[mask])[0] - raw[mask], axis=1).tolist())
common_adjacent = common[1:] & common[:-1]
metrics = {
    'deterministic_scene_ball_arrays': True, 'deterministic_component_arrays_and_provenance': True,
    'deterministic_replay_manifest': True, 'common_valid_frames': int(common.sum()),
    'common_raw_observation_count': len(raw_errors[0]),
    'raw_reprojection_before_median_px': float(np.median(raw_errors[0])),
    'raw_reprojection_after_median_px': float(np.median(raw_errors[1])),
    'raw_reprojection_before_p95_px': float(np.quantile(raw_errors[0], .95)),
    'raw_reprojection_after_p95_px': float(np.quantile(raw_errors[1], .95)),
    'common_adjacent_step_before_p95_m': float(np.quantile(np.linalg.norm(np.diff(before.ball_3d, axis=0), axis=1)[common_adjacent], .95)),
    'common_adjacent_step_after_p95_m': float(np.quantile(np.linalg.norm(np.diff(after.ball_3d, axis=0), axis=1)[common_adjacent], .95)),
    'comparison_is_ground_truth': False,
}
t = np.arange(after.num_frames) / after.fps
fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), constrained_layout=True)
for scene, label, color in ((before, 'Saved detector + triangulation', '#de8c29'),
                            (after, 'Motion consensus + LOESS + triangulation', '#168666')):
    xyz = masked_trajectory(scene.ball_3d, scene.ball_3d_valid)
    axes[0].plot(t, xyz[:, 2], color=color, label=label, lw=1)
    step = np.linalg.norm(np.diff(scene.ball_3d, axis=0), axis=1)
    step[~(scene.ball_3d_valid[1:] & scene.ball_3d_valid[:-1])] = np.nan
    axes[1].plot(t[1:], step, color=color, lw=1)
    axes[2].plot(xyz[:, 0], xyz[:, 1], color=color, lw=1)
axes[0].set(ylabel='Court height Z (m)', xlabel='Time (s)', title='Frozen cameras and people; gaps remain gaps (no 3D ground truth)')
axes[0].legend(loc='upper right', fontsize=8)
axes[1].set(ylabel='Adjacent 3D step (m)', xlabel='Time (s)')
axes[2].set(ylabel='Court Y (m)', xlabel='Court X (m)')
for ax in axes:
    ax.grid(alpha=.2)
fig.savefig(args.output / 'comparison.png', dpi=120)
plt.close(fig)

camera = 'cam1'
view = after.metadata['camera_ids'].index(camera)
directory = args.replay / f'ball_{camera}'
desc = json.loads((directory / 'artifact.json').read_text())
row = unpack_value(desc['payload'], directory, desc['arrays'])
smoothing = row['smoothing']
raw_xy = before.ball_uv[view] * [before.width, before.height]
filtered_xy = row['uv_px']
valids = (before.ball_vis[view], row['observed'])
xy_rows = (raw_xy, filtered_xy)
video_path = Path(after.metadata['video_paths'][view])
cv2.setNumThreads(2)
writer = cv2.VideoWriter(str(args.output / 'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), after.fps / 6, (1280, 400))
if not writer.isOpened():
    raise RuntimeError('Comparison video writer unavailable')
stills = []
with RandomAccessVideoReader(video_path) as reader:
    for frame in range(0, after.num_frames, 6):
        source = reader.read(frame)
        panels = []
        for i, (xy, valid) in enumerate(zip(xy_rows, valids, strict=True)):
            panel = cv2.resize(source, (640, 360), interpolation=cv2.INTER_AREA)
            color = (30, 150, 255) if i == 0 else (90, 255, 80)
            for k in range(max(1, frame - 30), frame + 1):
                if valid[k - 1] and valid[k]:
                    a, b = (tuple(np.rint(xy[j] / 3).astype(int)) for j in (k - 1, k))
                    cv2.line(panel, a, b, color, 2, cv2.LINE_AA)
            if valid[frame]:
                cv2.circle(panel, tuple(np.rint(xy[frame] / 3).astype(int)), 6, color, 2, cv2.LINE_AA)
            panels.append(panel)
        canvas = np.zeros((400, 1280, 3), np.uint8)
        canvas[40:, :640], canvas[40:, 640:] = panels
        cv2.putText(canvas, f'SAVED DETECTOR  |  frame {frame}  |  3D valid={int(before.ball_3d_valid[frame])}', (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .58, (30, 150, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'CONSENSUS + LOESS  |  real point={int(row["observed"][frame])}  |  3D valid={int(after.ball_3d_valid[frame])}', (650, 25), cv2.FONT_HERSHEY_SIMPLEX, .58, (90, 255, 80), 1, cv2.LINE_AA)
        writer.write(canvas)
        if frame in (180, 360, 540, 720):
            stills.append(cv2.resize(canvas, (768, 240)))
writer.release()
cv2.imwrite(str(args.output / 'overlay_contact_sheet.png'), np.concatenate(stills, axis=0))
metrics['comparison_mp4_sha256'] = dual_sha256(args.output / 'comparison.mp4')
(args.output / 'comparison.json').write_text(json.dumps(metrics, indent=2) + '\n')
print(json.dumps(metrics, indent=2))
