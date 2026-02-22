"""Peak extraction utilities for event detection."""

from __future__ import annotations

import torch
from torch import Tensor


def find_peaks_1d(
    values: Tensor,
    *,
    threshold: float,
    min_distance: int,
    top_k: int | None,
) -> tuple[list[int], list[float]]:
    """Find peak indices and scores in a 1D tensor.

    Args:
        values: 1D tensor of scores.
        threshold: Minimum score for a peak.
        min_distance: Minimum index distance between peaks.
        top_k: Optional limit on number of peaks (by score).

    Returns:
        Tuple of (indices, scores) as Python lists.
    """
    if values.numel() == 0:
        return [], []

    threshold = float(threshold)
    min_distance = max(int(min_distance), 1)
    values = values.detach().cpu()

    left = torch.cat([values[:1], values[:-1]])
    right = torch.cat([values[1:], values[-1:]])
    is_peak = (values >= left) & (values >= right) & (values >= threshold)

    idx = torch.nonzero(is_peak).flatten().tolist()
    scores = values[is_peak].tolist()

    if not idx:
        return [], []

    if min_distance > 1:
        order = sorted(range(len(idx)), key=lambda i: scores[i], reverse=True)
        selected_idx: list[int] = []
        selected_scores: list[float] = []
        for i in order:
            t = idx[i]
            if all(abs(t - s) >= min_distance for s in selected_idx):
                selected_idx.append(t)
                selected_scores.append(float(scores[i]))
        idx = selected_idx
        scores = selected_scores

    if top_k is not None and len(idx) > int(top_k):
        order = sorted(range(len(idx)), key=lambda i: scores[i], reverse=True)[: int(top_k)]
        idx = [idx[i] for i in order]
        scores = [scores[i] for i in order]

    order_time = sorted(range(len(idx)), key=lambda i: idx[i])
    idx = [idx[i] for i in order_time]
    scores = [scores[i] for i in order_time]
    return idx, scores


def extract_event_peaks(
    probs: Tensor,
    seq_len: Tensor | None,
    *,
    threshold: float,
    min_distance: int,
    top_k: int | None,
) -> tuple[list[list[list[int]]], list[list[list[float]]]]:
    """Extract per-event peak indices and scores.

    Args:
        probs: Event probabilities of shape (B, T, E).
        seq_len: Optional sequence lengths of shape (B,).
        threshold: Minimum score for a peak.
        min_distance: Minimum index distance between peaks.
        top_k: Optional limit on number of peaks (by score).

    Returns:
        Tuple of (peaks, peak_scores), both shaped [B][E][N].
    """
    B, T, E = probs.shape
    peaks: list[list[list[int]]] = []
    peak_scores: list[list[list[float]]] = []
    seq_len_cpu = seq_len.detach().cpu() if seq_len is not None else None

    for b in range(B):
        b_peaks: list[list[int]] = []
        b_scores: list[list[float]] = []
        t_len = int(seq_len_cpu[b].item()) if seq_len_cpu is not None else T
        t_len = max(0, min(T, t_len))
        for e in range(E):
            series = probs[b, :t_len, e]
            idx, scores = find_peaks_1d(
                series,
                threshold=threshold,
                min_distance=min_distance,
                top_k=top_k,
            )
            b_peaks.append(idx)
            b_scores.append(scores)
        peaks.append(b_peaks)
        peak_scores.append(b_scores)

    return peaks, peak_scores


if __name__ == "__main__":
    dummy = torch.tensor([[0.1, 0.5, 0.2, 0.7, 0.1]])
    idx, scores = find_peaks_1d(dummy[0], threshold=0.3, min_distance=1, top_k=None)
    assert idx == [1, 3]
    assert len(scores) == 2
    probs = dummy.view(1, 5, 1)
    peaks, _ = extract_event_peaks(probs, None, threshold=0.3, min_distance=1, top_k=None)
    assert peaks[0][0] == [1, 3]
    print("peaks smoke ok")
