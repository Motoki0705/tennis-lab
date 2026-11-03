"""Asymmetric, independently projected candidate fixtures for association tests."""

from dataclasses import replace

import numpy as np

from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tennis_scene.pipeline.components.ball_evidence import BallHeatmapEvidence
from tests.support.ball3d_evaluation import make_inputs


def candidate_evidence(uv, scores, valid, size=(320, 240)):
    frames, k = scores.shape
    # Pack valid peaks, descending scores, preserving each source candidate identity.
    order = np.argsort(np.where(valid, -scores, np.inf), axis=1, kind="stable")
    uv = np.take_along_axis(uv, order[..., None], axis=1)
    scores = np.take_along_axis(scores, order, axis=1)
    valid = np.take_along_axis(valid, order, axis=1)
    uv = np.where(valid[..., None], uv, 0).astype(np.float32)
    scores = np.where(valid, scores, 0).astype(np.float32)
    cells = np.rint(uv / (np.array(size) - 1) * [159, 119]).astype(np.int64)
    maps = np.zeros((frames, 120, 160), np.float32)
    for t, j in zip(*np.nonzero(valid), strict=True):
        maps[t, cells[t, j, 1], cells[t, j, 0]] = scores[t, j]
    return BallHeatmapEvidence(size, BallCandidateConfig(max_candidates=k, nms_kernel=3, patch_size=1),
        maps, uv, scores, valid, cells, scores[..., None, None].copy(), valid[..., None, None].copy(),
        np.arange(frames, dtype=np.int64), np.zeros(frames, np.int64))


def projected_pixels(camera, xyz):
    # Fixtures use axis-aligned cameras, deliberately different centres/focals.
    center, intrinsic = camera.center, camera.intrinsic
    return np.column_stack((intrinsic[0, 0] * (xyz[:, 0] - center[0]) / (center[2] - xyz[:, 2]) + 160,
                            intrinsic[1, 1] * (center[1] - xyz[:, 1]) / (center[2] - xyz[:, 2]) + 120))


def make_topk_inputs(frames=18, *, missing=(), turn=False):
    inputs = make_inputs(frames=frames)
    t = np.arange(frames)
    xyz = np.column_stack((0.3 + 0.08 * t, 1.8 - 0.03 * t, 1.1 + 0.02 * t))
    if turn:
        xyz[:, 1] = 0.2 + 0.1 * np.abs(t - 8)
        xyz[:, 2] = 0.4 + 0.06 * np.abs(t - 8)
    views = []
    true_indices = []
    for i, view in enumerate(inputs.views):
        distractor = np.tile([2.4 - 1.8 * i, -1.9 + 1.4 * i, 1.5], (frames, 1))
        # The first camera sticks to one object; the others alternate distractors.
        if i:
            distractor[:, 0] += np.where(t % 2, 0.8, -0.6)
        uv = np.stack([projected_pixels(view.camera, xyz), projected_pixels(view.camera, distractor)], axis=1)
        scores = np.tile([0.8, 0.12], (frames, 1))
        if i == 0:
            scores[4:15] = [0.35, 0.96]
        valid = np.ones((frames, 2), bool)
        for start, stop in missing:
            valid[start:stop] = False
        evidence = candidate_evidence(uv, scores, valid)
        true_indices.append(np.where(valid[:, 0], scores[:, 1] > scores[:, 0], -1))
        views.append(replace(view, detections=replace(view.detections, evidence=evidence)))
    return replace(inputs, views=tuple(views)), xyz, np.array(true_indices)
