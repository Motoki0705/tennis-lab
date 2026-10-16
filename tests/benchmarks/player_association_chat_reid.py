"""Compare appearance encoders on ``chat-player-v1`` (single-view broadcast, #933).

Per ``(clip, track)`` of a split, up to ``--samples`` crops are taken evenly in
time from ``observed`` boxes that are neither occluded nor truncated, at least
``--min-height`` px tall and inside the image (inferred and unresolved boxes
are never used for evaluation). Each track is cut into its earlier and later
half; a half's descriptor is the normalized mean of its crop embeddings.

Pairs of half-track descriptors:

* positive: the two halves of one track;
* negative: halves of different tracks of the same clip, and of tracks of
  different sources. Tracks of the same source but another clip are never
  paired (the same person may carry another name there).

Per encoder: ROC AUC of cosine similarity (positive vs. negative) and rank-1
(the earlier half retrieves its own later half among all allowed later halves).
On the ``--fit-split`` split a logistic model ``P(same) = sigmoid((cos - c0) / tau)``
is fitted and reported; the association score uses it without seeing the Meiji
labels. Writes ``chat_reid.json`` under ``--report``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.tasks.player_association.appearance.encoders import (
    ENCODER_CANDIDATES,
    build_encoder,
)
from src.tasks.player_association.appearance.sampling import crop
from src.tasks.player_detection.data.store import (
    BBOX_SOURCE_CODES,
    PlayerFrameStore,
    Split,
)
from src.tennis_scene.pipeline.artifacts import write_json_atomic

OBSERVED = BBOX_SOURCE_CODES["observed"]


def track_crops(store: PlayerFrameStore, split: Split, *, samples: int, min_height: float, border: float
                ) -> list[tuple[str, str, int, list[tuple[int, NDArray[np.float32]]]]]:
    """``(source, clip, track index, [(frame index, box)])`` of every usable track, crops chosen evenly in time."""
    usable: dict[tuple[int, int], list[tuple[int, int, NDArray[np.float32]]]] = {}
    for row in store.split_frames(split).tolist():
        clip = store.clip_of(row)
        instances = store.instances_of(row)
        for track, box, source, occluded, truncated in zip(instances.track_index.tolist(), instances.boxes_xyxy,
                                                           instances.bbox_source.tolist(), instances.occluded.tolist(),
                                                           instances.truncated.tolist(), strict=True):
            if source != OBSERVED or occluded or truncated or not np.isfinite(box).all():
                continue
            if box[3] - box[1] < min_height or box[0] < border or box[1] < border or box[2] > clip.width - border or box[3] > clip.height - border:
                continue
            usable.setdefault((clip.index, track), []).append((int(store.frames["frame_index"][row]), row, box))
    result = []
    for (clip_index, track), items in sorted(usable.items()):
        items.sort(key=lambda item: item[0])
        if len(items) < 2:
            continue
        chosen = np.unique(np.linspace(0, len(items) - 1, min(samples, len(items))).round().astype(int))
        clip = store.clips[clip_index]
        result.append((clip.source_id, clip.clip_id, track, [(items[i][1], items[i][2]) for i in chosen]))
    return result


def half_descriptors(embeddings: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    half = len(embeddings) // 2
    early, late = embeddings[:half].mean(0), embeddings[half:].mean(0)
    return early / np.linalg.norm(early), late / np.linalg.norm(late)


def roc_auc(positive: NDArray[np.float64], negative: NDArray[np.float64]) -> float | None:
    if not len(positive) or not len(negative):
        return None
    return float((positive[:, None] > negative[None, :]).mean() + (positive[:, None] == negative[None, :]).mean() / 2)


def fit_logistic(positive: NDArray[np.float64], negative: NDArray[np.float64]) -> dict[str, float]:
    """Class-balanced maximum-likelihood ``sigmoid((cos - c0) / tau)``."""
    cos = np.concatenate((positive, negative))
    target = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    weight = np.where(target == 1, .5 / len(positive), .5 / len(negative))

    def loss(params: NDArray[np.float64]) -> float:
        logits = (cos - params[0]) * np.exp(params[1])
        return float((weight * (np.logaddexp(0, logits) - target * logits)).sum())

    result = minimize(loss, np.array([float(np.median(cos)), np.log(20.)]), method="Nelder-Mead")
    if not result.success:
        raise RuntimeError(f"Logistic calibration did not converge: {result.message}")
    return {"c0": float(result.x[0]), "tau": float(np.exp(-result.x[1])), "balanced_log_loss": float(result.fun)}


def evaluate_split(tracks: list[tuple[str, str, int, list[tuple[int, NDArray[np.float32]]]]], embeddings: list[NDArray[np.float32]]
                   ) -> tuple[dict[str, Any], NDArray[np.float64], NDArray[np.float64]]:
    halves = [half_descriptors(e) for e in embeddings]
    positive = np.asarray([early @ late for early, late in halves], np.float64)
    negative: list[float] = []
    hits = 0
    for i, (source_i, clip_i, _, _) in enumerate(tracks):
        allowed = [j for j, (source_j, clip_j, _, _) in enumerate(tracks) if j == i or clip_j == clip_i or source_j != source_i]
        scores = {j: float(halves[i][0] @ halves[j][1]) for j in allowed}
        hits += max(scores, key=lambda j: scores[j]) == i
        negative += [score for j, score in scores.items() if j != i]
    negatives = np.asarray(negative, np.float64)
    return ({"tracks": len(tracks), "positives": len(positive), "negatives": len(negatives), "auc": roc_auc(positive, negatives),
             "rank1": hits / len(tracks) if tracks else None}, positive, negatives)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True, help="data/player_detection/chat-player-v1")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--encoder", action="append", choices=ENCODER_CANDIDATES, default=[])
    parser.add_argument("--split", action="append", choices=("train", "val", "test"), default=[])
    parser.add_argument("--fit-split", choices=("train", "val"), default="val")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--min-height", type=float, default=48.)
    parser.add_argument("--border", type=float, default=4.)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    store = PlayerFrameStore(args.store.resolve())
    splits: list[Split] = args.split or ["val", "test"]
    if args.fit_split not in splits:
        raise ValueError(f"--fit-split {args.fit_split} must be one of the evaluated splits {splits}")
    names = args.encoder or list(ENCODER_CANDIDATES)
    encoders = [build_encoder(name, checkpoint_root=args.repo / "ckpt", external_root=args.repo / "third_party", device=args.device) for name in names]
    results: dict[str, Any] = {name: {} for name in names}
    for split in splits:
        tracks = track_crops(store, split, samples=args.samples, min_height=args.min_height, border=args.border)
        crops_by_size: dict[tuple[int, int], NDArray[np.float32]] = {}
        order = [(t, row, box) for t, (_, _, _, items) in enumerate(tracks) for row, box in items]
        for encoder in encoders:
            if encoder.input_size not in crops_by_size:
                crops_by_size[encoder.input_size] = np.stack([crop(store.read_bgr(row), box, encoder.input_size) for _, row, box in order])
            images = torch.from_numpy(crops_by_size[encoder.input_size])
            vectors = torch.cat([encoder.embed(images[i:i + 64]) for i in range(0, len(images), 64)]).numpy()
            owner = np.asarray([t for t, _, _ in order])
            summary, positive, negative = evaluate_split(tracks, [vectors[owner == t] for t in range(len(tracks))])
            if split == args.fit_split:
                summary["calibration"] = fit_logistic(positive, negative)
            results[encoder.name][split] = summary
            print(json.dumps({"encoder": encoder.name, "split": split, **summary}), flush=True)
    write_json_atomic(report / "chat_reid.json", {"schema": "player_association_chat_reid_v1", "store": str(args.store.resolve()),
                                                  "samples": args.samples, "min_height": args.min_height, "border": args.border,
                                                  "fit_split": args.fit_split, "results": results})


if __name__ == "__main__":
    main()
