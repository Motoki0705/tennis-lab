"""Standalone loss/validation plots from the durable per-update records."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib


def plot_dev_curves(output: Path) -> None:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in (output / 'updates.jsonl').read_text().splitlines()]
    manifest = json.loads((output / 'manifest.json').read_text())
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, key in zip(axes.flat, ('loss', 'x0', 'reprojection', 'physics', 'event'), strict=False):
        axis.plot([r['update'] for r in rows], [r[key] for r in rows], linewidth=.6, label='train')
        axis.plot([r['update'] for r in manifest['validation']], [r['loss'][key] for r in manifest['validation']], 'o-', label='val')
        axis.set(title=key, xlabel='update', yscale='symlog')
        axis.legend()
    axis = axes.flat[-1]
    for key in ('overall', 'gap', 'event_pm5'):
        axis.plot([r['update'] for r in manifest['validation']],
                  [r['metrics']['mean']['rmse_m_' + key]['value'] for r in manifest['validation']], 'o-', label=key)
    baselines = json.loads((output.parent / 'baselines.json').read_text())
    for method in ('mixture_mean', 'top_component', 'mixture_mean_rts'):
        axis.axhline(baselines['methods'][method]['metrics']['rmse_m_overall']['value'], linestyle='--', linewidth=.8, label=method)
    axis.set(title='validation mean trajectory RMSE', xlabel='update', ylabel='m')
    axis.legend()
    figure.suptitle(manifest['objective'] + ' — synthetic dev only')
    figure.savefig(output / 'curves.png', dpi=150)
    plt.close(figure)
