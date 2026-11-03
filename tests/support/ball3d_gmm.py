"""Independent pinhole and Gaussian evidence fixture, never real-clip labels."""

from dataclasses import replace

import numpy as np

from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tennis_scene.pipeline.components.ball_evidence import BallHeatmapEvidence
from tests.support.ball3d_evaluation import make_inputs


def make_gmm_inputs(frames=5, *, distractors=False, missing=True):
    inputs = make_inputs(frames)
    xyz = np.column_stack((1 + np.arange(frames)/10, 2 - np.arange(frames)/5, 1 + np.arange(frames)/20))
    views = []
    h, w, p = 121, 161, 5
    yy, xx = np.mgrid[:h, :w]
    for i, view in enumerate(inputs.views):
        center = view.camera.center
        depth = center[2] - xyz[:, 2]
        # Formulas independent of production projection.
        uv = np.column_stack((view.camera.intrinsic[0, 0]*(xyz[:, 0]-center[0])/depth+160,
                              view.camera.intrinsic[1, 1]*(center[1]-xyz[:, 1])/depth+120))
        means = uv / [319, 239] * [w-1, h-1]
        maps = np.zeros((frames, h, w), np.float32)
        k = 2 if distractors else 1
        cells = np.zeros((frames, k, 2), np.int64)
        peaks = np.zeros((frames, k, 2), np.float32)
        scores = np.zeros((frames, k), np.float32)
        masks = np.ones((frames, k), bool)
        patches = np.zeros((frames, k, p, p), np.float32)
        patch_valid = np.ones(patches.shape, bool)
        for f in range(frames):
            if missing and f == 2:
                masks[f] = False
                patch_valid[f] = False
                continue
            mode = [means[f]]
            if distractors:
                mode.append(means[f] + [[-18, 17], [26, 23], [-13, -31]][i])
            for j, mean in enumerate(mode):
                maps[f] += ((0.80 if j == 0 else 0.94)*np.exp(-0.5*((xx-mean[0])**2/1.3+(yy-mean[1])**2/0.7))).astype(np.float32)
            order = sorted(range(k), key=lambda j: -float(maps[f, int(round(mode[j][1])), int(round(mode[j][0]))]))
            for rank, j in enumerate(order):
                x, y = np.rint(mode[j]).astype(int)
                cells[f, rank] = [x, y]
                peaks[f, rank] = mode[j] / [w-1, h-1] * [319, 239]
                scores[f, rank] = maps[f, y, x]
                patches[f, rank] = maps[f, y-2:y+3, x-2:x+3]
        evidence = BallHeatmapEvidence(inputs.source_size_wh,
            BallCandidateConfig(max_candidates=k, nms_kernel=3, patch_size=p), maps,
            peaks, scores, masks, cells, patches, patch_valid,
            np.arange(frames, dtype=np.int64), np.zeros(frames, np.int64))
        views.append(replace(view, detections=replace(view.detections, evidence=evidence)))
    return replace(inputs, views=tuple(views)), xyz
