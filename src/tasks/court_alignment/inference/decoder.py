"""Pure decoding and instance-association APIs for Court Alignment KP maps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils.data.heatmaps import heatmaps_to_peaks, refine_peaks_log_parabolic

NUM_KEYPOINTS = 14


@dataclass(frozen=True, slots=True)
class CourtPeakDetections:
    """Multi-peak detections retaining the semantic channel axis.

    Coordinates and vote offsets are in output-image pixels and have shape
    ``(B,14,K,2)`` / ``(B,14,K,2)``.  ``valid`` marks padded slots.
    """

    keypoints_px: Tensor
    scores: Tensor
    valid: Tensor
    center_votes_px: Tensor

    def __post_init__(self) -> None:
        if self.keypoints_px.ndim != 4 or self.keypoints_px.shape[-1] != 2:
            raise ValueError("keypoints_px must have shape (B,14,K,2).")
        if self.keypoints_px.shape[1] != NUM_KEYPOINTS:
            raise ValueError("keypoints_px must have fourteen semantic channels.")
        expected = self.keypoints_px.shape[:-1]
        if self.scores.shape != expected or self.valid.shape != expected:
            raise ValueError("scores and valid must have shape (B,14,K).")
        if self.center_votes_px.shape != self.keypoints_px.shape:
            raise ValueError("center_votes_px must have shape (B,14,K,2).")
        if self.valid.dtype != torch.bool:
            raise TypeError("valid must have boolean dtype.")
        for name, value in (
            ("keypoints_px", self.keypoints_px),
            ("scores", self.scores),
            ("center_votes_px", self.center_votes_px),
        ):
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values.")

    @property
    def keypoints(self) -> Tensor:
        return self.keypoints_px

    @property
    def coordinates_px(self) -> Tensor:
        return self.keypoints_px

    @property
    def coords_px(self) -> Tensor:
        return self.keypoints_px

    @property
    def votes_px(self) -> Tensor:
        """Alias for sampled center-vote offsets."""
        return self.center_votes_px


@dataclass(frozen=True, slots=True)
class CourtInstanceBatch:
    """Variable-length court instances decoded from one batch sample."""

    keypoints_px: Tensor  # (N,14,2), zero-filled where a semantic KP is absent
    scores: Tensor  # (N,14)
    valid: Tensor  # (N,14)
    centers_px: Tensor  # (N,2)

    def __post_init__(self) -> None:
        if self.keypoints_px.ndim != 3 or self.keypoints_px.shape[1:] != (NUM_KEYPOINTS, 2):
            raise ValueError("instance keypoints_px must have shape (N,14,2).")
        if self.scores.shape != self.valid.shape or self.scores.shape != self.keypoints_px.shape[:2]:
            raise ValueError("instance scores and valid must have shape (N,14).")
        if self.centers_px.shape != (self.keypoints_px.shape[0], 2):
            raise ValueError("instance centers_px must have shape (N,2).")
        if self.valid.dtype != torch.bool:
            raise TypeError("instance valid must have boolean dtype.")
        for name, value in (
            ("keypoints_px", self.keypoints_px),
            ("scores", self.scores),
            ("centers_px", self.centers_px),
        ):
            if not value.is_floating_point():
                raise TypeError(f"instance {name} must be floating point.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"instance {name} must contain only finite values.")
        if (
            self.scores.device != self.keypoints_px.device
            or self.valid.device != self.keypoints_px.device
            or self.centers_px.device != self.keypoints_px.device
        ):
            raise ValueError("instance tensors must share a device.")

    @property
    def num_instances(self) -> int:
        return int(self.keypoints_px.shape[0])

    @property
    def keypoints(self) -> Tensor:
        return self.keypoints_px


@dataclass(frozen=True, slots=True)
class CourtInstances:
    """Batch wrapper for variable-length :class:`CourtInstanceBatch` values."""

    samples: tuple[CourtInstanceBatch, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.samples, tuple) or any(
            not isinstance(sample, CourtInstanceBatch) for sample in self.samples
        ):
            raise TypeError("samples must be a tuple of CourtInstanceBatch values.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CourtInstanceBatch:
        return self.samples[index]

    @property
    def instances(self) -> tuple[CourtInstanceBatch, ...]:
        """Alias exposing the variable-length per-sample results."""
        return self.samples

    @property
    def num_instances(self) -> Tensor:
        device = self.samples[0].keypoints_px.device if self.samples else None
        return torch.tensor(
            [sample.num_instances for sample in self.samples],
            dtype=torch.long,
            device=device,
        )


@dataclass(slots=True)
class _Cluster:
    center: Tensor
    points: list[tuple[int, Tensor, Tensor]]
    vote_centers: list[Tensor]


def _validate_decoder_inputs(heatmap_logits: Tensor, center_votes: Tensor) -> None:
    if not isinstance(heatmap_logits, Tensor) or heatmap_logits.ndim != 4:
        raise ValueError("heatmap_logits must have shape (B,14,H,W).")
    if heatmap_logits.shape[1] != NUM_KEYPOINTS or any(size <= 0 for size in heatmap_logits.shape):
        raise ValueError("heatmap_logits must have shape (B,14,H,W).")
    if not heatmap_logits.is_floating_point() or not bool(torch.isfinite(heatmap_logits).all()):
        raise ValueError("heatmap_logits must be finite floating point.")
    if not isinstance(center_votes, Tensor) or center_votes.shape != (heatmap_logits.shape[0], 2, *heatmap_logits.shape[-2:]):
        raise ValueError("center_votes must have shape (B,2,H,W) matching heatmap_logits.")
    if not center_votes.is_floating_point() or not bool(torch.isfinite(center_votes).all()):
        raise ValueError("center_votes must be finite floating point.")
    if center_votes.device != heatmap_logits.device:
        raise ValueError("heatmap_logits and center_votes must be on the same device.")


def _validate_decode_options(threshold: float, nms_kernel: int, max_peaks: int) -> None:
    if not math.isfinite(float(threshold)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0,1].")
    if type(nms_kernel) is not int or nms_kernel <= 0 or nms_kernel % 2 == 0:
        raise ValueError("nms_kernel must be a positive odd integer.")
    if type(max_peaks) is not int or max_peaks <= 0:
        raise ValueError("max_peaks must be a positive integer.")


def decode_keypoint_peaks(
    heatmap_logits: Tensor,
    center_votes: Tensor,
    *,
    threshold: float = 0.25,
    nms_kernel: int = 3,
    max_peaks: int = 8,
    subpixel_refine: bool = True,
) -> CourtPeakDetections:
    """Decode all local KP peaks and sample the corresponding vote offsets."""
    _validate_decoder_inputs(heatmap_logits, center_votes)
    _validate_decode_options(threshold, nms_kernel, max_peaks)
    probabilities = heatmap_logits.sigmoid()
    coords_norm, scores, valid = heatmaps_to_peaks(
        probabilities,
        threshold=threshold,
        nms_kernel=nms_kernel,
        max_peaks=max_peaks,
    )
    if subpixel_refine:
        coords_norm = refine_peaks_log_parabolic(probabilities, coords_norm)
    height, width = heatmap_logits.shape[-2:]
    scale = coords_norm.new_tensor((float(max(width - 1, 0)), float(max(height - 1, 0))))
    keypoints_px = coords_norm * scale

    # Center votes are dense maps but only their values at a detected peak are
    # semantically meaningful.  Sampling the nearest lattice cell keeps the
    # decoder deterministic and matches the target rasterisation contract.
    x_index = (coords_norm[..., 0] * float(max(width - 1, 0))).round().long().clamp(0, width - 1)
    y_index = (coords_norm[..., 1] * float(max(height - 1, 0))).round().long().clamp(0, height - 1)
    votes = center_votes.permute(0, 2, 3, 1)
    batch_index = torch.arange(heatmap_logits.shape[0], device=heatmap_logits.device)[:, None, None]
    sampled_votes = votes[batch_index, y_index, x_index]
    if sampled_votes.shape != keypoints_px.shape:
        raise RuntimeError("Internal center-vote sampling shape mismatch.")
    sampled_votes = sampled_votes * valid.unsqueeze(-1).to(sampled_votes.dtype)
    return CourtPeakDetections(keypoints_px, scores, valid, sampled_votes)


def _batched_tensor(value: Tensor, *, last_dim: int, name: str) -> tuple[Tensor, bool]:
    if value.ndim == 3 and value.shape[-1] == last_dim:
        return value.unsqueeze(0), True
    if value.ndim == 4 and value.shape[-1] == last_dim:
        return value, False
    raise ValueError(f"{name} must have shape (14,K,{last_dim}) or (B,14,K,{last_dim}).")


def group_peak_votes(
    keypoints_px: Tensor,
    center_votes_px: Tensor,
    valid: Tensor,
    scores: Tensor | None = None,
    *,
    cluster_distance_px: float = 12.0,
    max_instances: int | None = None,
) -> CourtInstances | CourtInstanceBatch:
    """Group semantic KP peaks by their voted court center.

    A point votes for ``keypoint_px + center_votes_px``.  Clustering is
    performed in vote space, then each cluster retains the highest-scoring
    peak for each semantic channel.  Thus the channel-wise confidence ranking
    may be different for every court without changing instance association.
    """
    kp, squeezed = _batched_tensor(keypoints_px, last_dim=2, name="keypoints_px")
    votes, vote_squeezed = _batched_tensor(center_votes_px, last_dim=2, name="center_votes_px")
    if squeezed != vote_squeezed or votes.shape != kp.shape:
        raise ValueError("keypoints_px and center_votes_px must share shape.")
    if valid.ndim == 2:
        valid_b = valid.unsqueeze(0)
    elif valid.ndim == 3:
        valid_b = valid
    else:
        raise ValueError("valid must have shape (14,K) or (B,14,K).")
    if valid_b.shape != kp.shape[:-1] or valid_b.dtype != torch.bool:
        raise ValueError("valid must be boolean with shape matching keypoint peaks.")
    if kp.device != votes.device or kp.device != valid_b.device:
        raise ValueError("keypoints, votes, and valid must share a device.")
    if not kp.is_floating_point() or not votes.is_floating_point():
        raise TypeError("keypoints_px and center_votes_px must be floating point.")
    if not bool(torch.isfinite(kp).all()) or not bool(torch.isfinite(votes).all()):
        raise ValueError("keypoints_px and center_votes_px must be finite.")
    if scores is None:
        scores_b = torch.ones_like(valid_b, dtype=kp.dtype)
    else:
        scores_b = scores.unsqueeze(0) if scores.ndim == 2 else scores
        if scores_b.shape != valid_b.shape or not scores_b.is_floating_point():
            raise ValueError("scores must be floating point with shape matching valid.")
        if scores_b.device != kp.device or not bool(torch.isfinite(scores_b).all()):
            raise ValueError("scores must be finite and share the keypoint device.")
    if not math.isfinite(float(cluster_distance_px)) or cluster_distance_px <= 0:
        raise ValueError("cluster_distance_px must be finite and positive.")
    if max_instances is not None and (type(max_instances) is not int or max_instances <= 0):
        raise ValueError("max_instances must be a positive integer or None.")

    outputs: list[CourtInstanceBatch] = []
    for sample_kp, sample_votes, sample_valid, sample_scores in zip(kp, votes, valid_b, scores_b, strict=True):
        candidates: list[tuple[int, Tensor, Tensor, Tensor]] = []
        for channel in range(NUM_KEYPOINTS):
            for peak_index in torch.nonzero(sample_valid[channel], as_tuple=False).flatten().tolist():
                point = sample_kp[channel, peak_index]
                vote_center = point + sample_votes[channel, peak_index]
                candidates.append((channel, point, vote_center, sample_scores[channel, peak_index]))
        # Confidence ordering only determines stable cluster IDs; association
        # itself uses vote-space distance and is independent of channel ranks.
        candidates.sort(key=lambda item: (-float(item[3]), item[0], float(item[2][1]), float(item[2][0])))
        clusters: list[_Cluster] = []
        for channel, point, vote_center, score in candidates:
            if not clusters:
                clusters.append(
                    _Cluster(
                        center=vote_center.clone(),
                        points=[(channel, point, score)],
                        vote_centers=[vote_center.clone()],
                    )
                )
                continue
            distances = torch.stack(
                [torch.linalg.vector_norm(vote_center - cluster.center) for cluster in clusters]
            )
            nearest = int(distances.argmin().item())
            if float(distances[nearest]) <= cluster_distance_px:
                cluster = clusters[nearest]
                cluster.points.append((channel, point, score))
                # Mean vote center is robust to small vote noise.  Keep the
                # original vote values, rather than recomputing from a point
                # after it has been assigned to a cluster.
                cluster.vote_centers.append(vote_center)
                cluster.center = torch.stack(cluster.vote_centers).mean(dim=0)
            else:
                clusters.append(
                    _Cluster(
                        center=vote_center.clone(),
                        points=[(channel, point, score)],
                        vote_centers=[vote_center.clone()],
                    )
                )
        clusters.sort(key=lambda cluster: (float(cluster.center[1]), float(cluster.center[0])))
        if max_instances is not None:
            clusters = clusters[:max_instances]
        count = len(clusters)
        instance_kp = sample_kp.new_zeros((count, NUM_KEYPOINTS, 2))
        instance_scores = sample_scores.new_zeros((count, NUM_KEYPOINTS))
        instance_valid = torch.zeros((count, NUM_KEYPOINTS), dtype=torch.bool, device=sample_kp.device)
        instance_centers = sample_kp.new_zeros((count, 2))
        for instance_index, cluster in enumerate(clusters):
            entries = cluster.points
            instance_centers[instance_index] = cluster.center
            for channel, point, score in entries:
                if not bool(instance_valid[instance_index, channel]) or score > instance_scores[instance_index, channel]:
                    instance_kp[instance_index, channel] = point
                    instance_scores[instance_index, channel] = score
                    instance_valid[instance_index, channel] = True
        outputs.append(CourtInstanceBatch(instance_kp, instance_scores, instance_valid, instance_centers))
    result = CourtInstances(tuple(outputs))
    return result[0] if squeezed else result


def decode_court_instances(
    heatmap_logits: Tensor,
    center_votes: Tensor,
    *,
    threshold: float = 0.25,
    nms_kernel: int = 3,
    max_peaks: int = 8,
    subpixel_refine: bool = True,
    cluster_distance_px: float = 12.0,
    max_instances: int | None = None,
) -> CourtInstances:
    """Decode KP maps and immediately associate their peaks into instances."""
    detections = decode_keypoint_peaks(
        heatmap_logits,
        center_votes,
        threshold=threshold,
        nms_kernel=nms_kernel,
        max_peaks=max_peaks,
        subpixel_refine=subpixel_refine,
    )
    grouped = group_peak_votes(
        detections.keypoints_px,
        detections.center_votes_px,
        detections.valid,
        detections.scores,
        cluster_distance_px=cluster_distance_px,
        max_instances=max_instances,
    )
    if isinstance(grouped, CourtInstanceBatch):
        return CourtInstances((grouped,))
    return grouped


# Descriptive aliases used by experiment notebooks and older prototype code.
extract_keypoint_peaks = decode_keypoint_peaks
decode_multi_peak_keypoints = decode_keypoint_peaks
group_center_votes = group_peak_votes


__all__ = [
    "CourtInstanceBatch",
    "CourtInstances",
    "CourtPeakDetections",
    "decode_court_instances",
    "decode_keypoint_peaks",
    "decode_multi_peak_keypoints",
    "extract_keypoint_peaks",
    "group_center_votes",
    "group_peak_votes",
]
