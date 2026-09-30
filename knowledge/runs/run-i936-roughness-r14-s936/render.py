"""CPU-only video of the preregistered first validation ID, every raw frame."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter


def main() -> None:
    bundle = Path(__file__).resolve().parent
    source = bundle.parent / 'run-i936-anchored-t128-flow-regression-r13-s936/collected'
    manifest = json.loads((source / 'manifest.json').read_text())
    rally = sorted(manifest['arms']['flow']['validation'][-1]['windows'])[0]
    windows = manifest['arms']['flow']['validation'][-1]['windows'][rally]
    paths = [source / 'baseline_predictions' / (rally + '.npz')]
    with np.load(paths[0], allow_pickle=False) as saved:
        trajectories = [saved[key] for key in ('truth', 'mixture_mean', 'mixture_mean_rts')]
    for arm in ('flow', 'regression'):
        path = source / arm / 'predictions' / (rally + '.npz')
        paths.append(path)
        with np.load(path, allow_pickle=False) as saved:
            trajectories.append(saved['mean_m'])
            times = saved['timestamps_seconds']
    combined = np.concatenate(trajectories)
    lower = np.minimum(combined.min(0), [-5.485, -11.885, 0.])
    upper = np.maximum(combined.max(0), [5.485, 11.885, 1.])
    margin = .08 * (upper - lower)
    lower, upper = lower - margin, upper + margin
    court = np.array([[-5.485, -11.885, 0], [5.485, -11.885, 0],
                      [5.485, 11.885, 0], [-5.485, 11.885, 0], [-5.485, -11.885, 0]])
    colors = ['#15803d', '#db4848', '#1976bf', '#8555c3', '#d37c14']
    titles = ['Ground truth', 'Mixture mean', '#929 RTS', 'Flow 20k mean', 'Regression 20k']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#475569', 'text.color': '#172b46'})
    fig = plt.figure(figsize=(18, 10), facecolor='#f5f7fa')
    grid = fig.add_gridspec(3, 5, left=.055, right=.98, top=.88, bottom=.085,
                           height_ratios=[1.15, .72, 1.1], hspace=.32, wspace=.27)
    fig.text(.055, .97, '3D ball trajectories', fontsize=24, weight='bold')
    fig.text(.055, .931, 'Same rally, same axes. Raw positions; no smoothing or discarded frames.', fontsize=12)
    clock = fig.text(.98, .955, '', ha='right', fontsize=13)
    artists: list[tuple[Any, Any, Any, Any, bool]] = []
    for column, (trajectory, color, title) in enumerate(zip(trajectories, colors, titles, strict=True)):
        for row in range(3):
            dimension = ((0, 1), (1, 2))[row] if row < 2 else (0, 1, 2)
            ax = fig.add_subplot(grid[row, column], projection='3d' if row == 2 else None)
            ax.set_facecolor('#ffffff')
            if row == 0:
                ax.set_title(title, color=color, fontsize=14, weight='bold', pad=12)
            if row < 2:
                i, j = dimension[0], dimension[1]
                ax.plot(court[:, i], court[:, j], color='#9aaaba', lw=.8)
                if row == 0:
                    ax.plot([-5.485, 5.485], [0, 0], color='#9aaaba', lw=.8)
                ax.plot(trajectories[0][:, i], trajectories[0][:, j], '--', color=colors[0], alpha=.55, lw=.9)
                ax.plot(trajectory[:, i], trajectory[:, j], color=color, alpha=.24, lw=.8)
                trail, = ax.plot([], [], color=color, lw=1.8)
                dot, = ax.plot([], [], 'o', color=color, ms=5)
                ax.set(xlim=(lower[i], upper[i]), ylim=(lower[j], upper[j]),
                       xlabel=('X (m)' if row == 0 else 'Y (m)'),
                       ylabel=('Top · Y (m)' if row == 0 else 'Side · Z (m)') if column == 0 else '')
                ax.grid(alpha=.16)
                if row == 0:
                    ax.set_aspect('equal', adjustable='box')
            else:
                ax.plot(*court.T, color='#9aaaba', lw=.8)
                ax.plot(*trajectories[0].T, '--', color=colors[0], alpha=.5, lw=.8)
                ax.plot(*trajectory.T, color=color, alpha=.24, lw=.8)
                trail, = ax.plot([], [], [], color=color, lw=1.8)
                dot, = ax.plot([], [], [], 'o', color=color, ms=5)
                ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
                       xlabel='X', ylabel='Y', zlabel='Z')
                ax.set_box_aspect(upper - lower)
                ax.view_init(elev=23, azim=-58)
                ax.tick_params(labelsize=8, pad=0)
            artists.append((trail, dot, trajectory, dimension, row == 2))
    fig.text(.055, .035, f'{rally} · first validation ID · all {len(times)} frames · source 59.94 Hz · playback 15 fps (0.25×)', fontsize=11)
    fig.text(.98, .035, 'Dashed green: GT   |   Color: last 24 frames   |   Faint: full raw trace', fontsize=10, ha='right')
    movie = bundle / 'val-00000-3d-comparison.mp4'
    if movie.exists():
        raise FileExistsError(movie)
    writer = FFMpegWriter(fps=15, codec='libx264', extra_args=['-crf', '20', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-threads', '1'])
    seams = [window['owned_start'] for window in windows[1:]]
    poster_frame = seams[0] if seams else len(times) // 2
    with writer.saving(fig, str(movie), dpi=100):
        for index, seconds in enumerate(times):
            owner = next(i for i, window in enumerate(windows) if window['owned_start'] <= index < window['owned_stop'])
            seam = '  ·  WINDOW SEAM' if any(abs(index - s) <= 2 for s in seams) else ''
            clock.set_text(f't = {seconds:.3f} s   |   frame {index:03d}/{len(times)-1}\nWindow {owner+1}/{len(windows)}{seam}')
            for trail, dot, trajectory, dimension, is3d in artists:
                selected = trajectory[max(0, index - 23):index + 1]
                trail.set_data(selected[:, dimension[0]], selected[:, dimension[1]])
                dot.set_data([trajectory[index, dimension[0]]], [trajectory[index, dimension[1]]])
                if is3d:
                    trail.set_3d_properties(selected[:, 2])
                    dot.set_3d_properties([trajectory[index, 2]])
            writer.grab_frame()
            if index == poster_frame:
                fig.savefig(bundle / 'video-poster.png', dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    report = {'rally': rally, 'rule': 'lexicographically first validation ID, full rally, fixed 20k; chosen before rendering',
              'rule_url': 'https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5914941598',
              'frames': len(times), 'source_fps': 60000 / 1001, 'playback_fps': 15,
              'views': ['top XY', 'side YZ', 'oblique 3D'], 'common_limits': [lower.tolist(), upper.tolist()],
              'smoothing': False, 'trajectory_selection': False, 'discarded_frames': 0,
              'poster_frame': poster_frame, 'seams': seams, 'device': 'cpu', 'ffmpeg_threads': 1,
              'sources': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'video_sha256': hashlib.sha256(movie.read_bytes()).hexdigest(), 'video_bytes': movie.stat().st_size}
    (bundle / 'video.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
