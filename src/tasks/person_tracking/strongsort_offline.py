"""Paper-based AFLink and Gaussian-smoothed interpolation; CPU inference only.

Source/weight provenance and restrictions: strongsort_NOTICE.md. Interpolated
boxes are a separate reconstruction and never become observed detections.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.utils.checksum import dual_sha256

AF_WEIGHT_SHA256 = 'b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4'


class _Temporal(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, (7, 1), bias=False)
        self.bnf = nn.BatchNorm2d(outputs)
        self.bnx = nn.BatchNorm2d(outputs)
        self.bny = nn.BatchNorm2d(outputs)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = self.conv(value)
        return torch.relu(torch.cat((self.bnf(features[..., 0:1]), self.bnx(features[..., 1:2]),
                                     self.bny(features[..., 2:3])), dim=-1))


class _Fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(256, 256, (1, 3), bias=False)
        self.bn = nn.BatchNorm2d(256)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.conv(value))).mean(dim=(-2, -1))


class _Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(value)))


class AFLinkNetwork(nn.Module):
    """Names retain compatibility with the published state dict, not its code."""
    def __init__(self) -> None:
        super().__init__()
        channels = (1, 32, 64, 128, 256)
        self.TemporalModule_1 = nn.Sequential(*[_Temporal(a, b) for a, b in zip(channels[:-1], channels[1:], strict=True)])
        self.TemporalModule_2 = nn.Sequential(*[_Temporal(a, b) for a, b in zip(channels[:-1], channels[1:], strict=True)])
        self.FusionBlock_1 = _Fusion()
        self.FusionBlock_2 = _Fusion()
        self.classifier = _Classifier()

    def forward(self, former: torch.Tensor, later: torch.Tensor) -> torch.Tensor:
        left = self.FusionBlock_1(self.TemporalModule_1(former))
        right = self.FusionBlock_2(self.TemporalModule_2(later))
        return torch.softmax(self.classifier(torch.cat((left, right), dim=1)), dim=1)


def aflink_inputs(former: np.ndarray, later: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Temporal endpoints, zero padding, then shared per-coordinate [-1,1] range."""
    if any(x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all() for x in (former, later)):
        raise ValueError('AFLink requires nonempty finite frame/x/y tracklets')
    left: np.ndarray = np.zeros((30, 3), np.float64)
    right: np.ndarray = np.zeros((30, 3), np.float64)
    left[-min(len(former), 30):] = former[-30:]
    right[:min(len(later), 30)] = later[:30]
    joined = np.concatenate((left, right))
    minimum, maximum = joined.min(0), joined.max(0)
    # Epsilon prevents zero extent; matches the reference preprocessing contract.
    scale = (maximum - minimum) / 2 + 1e-5
    center = (maximum + minimum) / 2
    return (torch.from_numpy(((left - center) / scale).astype(np.float32))[None, None],
            torch.from_numpy(((right - center) / scale).astype(np.float32))[None, None])


class AFLink:
    def __init__(self, checkpoint: Path) -> None:
        if dual_sha256(checkpoint) != AF_WEIGHT_SHA256:
            raise ValueError('AFLink checkpoint SHA-256 mismatch')
        self.model = AFLinkNetwork().eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)

    def links(self, boxes: np.ndarray, observed: np.ndarray) -> tuple[dict[int, int], list[dict[str, float | int]]]:
        if boxes.shape != (*observed.shape, 4) or observed.dtype != np.bool_:
            raise ValueError('AFLink needs aligned boxes and observed masks')
        count = len(boxes)
        sequences = []
        for row in range(count):
            frames = np.flatnonzero(observed[row])
            if not len(frames):
                raise ValueError('AFLink input has an empty track')
            sequences.append(np.column_stack((frames, boxes[row, frames, :2])))
        costs: np.ndarray = np.ones((count, count), np.float64)
        records: list[dict[str, float | int]] = []
        with torch.inference_mode():
            for i, left in enumerate(sequences):
                for j, right in enumerate(sequences):
                    gap = float(right[0, 0] - left[-1, 0])
                    distance = float(np.linalg.norm(right[0, 1:] - left[-1, 1:]))
                    if i == j or not 0 < gap < 30 or distance > 75:
                        continue
                    probability = float(self.model(*aflink_inputs(left, right))[0, 1])
                    costs[i, j] = 1 - probability
                    records.append({'former_row': i, 'later_row': j, 'gap': gap,
                                    'distance_px': distance, 'link_probability': probability})
        left_rows, right_rows = linear_sum_assignment(costs)
        predecessor = {int(j): int(i) for i, j in zip(left_rows, right_rows, strict=True) if costs[i, j] < .05}
        roots = {}
        for row in range(count):
            root, visited = row, set()
            while root in predecessor:
                if root in visited:
                    raise ValueError('AFLink produced a cycle')
                visited.add(root)
                root = predecessor[root]
            roots[row] = root
        return roots, records


@dataclass(frozen=True)
class ReconstructedTracks:
    boxes: np.ndarray
    observed: np.ndarray
    interpolated: np.ndarray


def gaussian_interpolation(boxes: np.ndarray, observed: np.ndarray) -> ReconstructedTracks:
    if boxes.shape != (*observed.shape, 4) or observed.dtype != np.bool_:
        raise ValueError('GSI needs aligned boxes and observed mask')
    smoothed = np.zeros_like(boxes)
    filled = np.zeros_like(observed)
    for row in range(len(boxes)):
        times = np.flatnonzero(observed[row])
        if not len(times):
            continue
        xywh = boxes[row, times].astype(np.float64).copy()
        xywh[:, 2:] -= xywh[:, :2]
        at = list(times)
        values = list(xywh)
        for k in range(len(times) - 1):
            gap = int(times[k + 1] - times[k])
            if 1 < gap < 20:
                for time in range(int(times[k]) + 1, int(times[k + 1])):
                    ratio = (time - times[k]) / gap
                    at.append(time)
                    values.append(xywh[k] * (1 - ratio) + xywh[k + 1] * ratio)
                    filled[row, time] = True
        order = np.argsort(at)
        frames = np.asarray(at)[order]
        length = float(np.clip(10 * np.log(1000 / len(frames)), .1, 100))
        kernel = np.exp(-.5 * ((frames[:, None] - frames[None]) / length) ** 2)
        noisy = kernel.copy()
        noisy.flat[::len(frames) + 1] += 1e-10
        result = kernel @ cho_solve(cho_factor(noisy, lower=True), np.asarray(values)[order])
        if not np.isfinite(result).all() or (result[:, 2:] <= 0).any():
            raise ValueError('GSI produced nonfinite or nonpositive boxes')
        result[:, 2:] += result[:, :2]
        smoothed[row, frames] = result
    return ReconstructedTracks(smoothed, observed.copy(), filled)
