"""Compare appearance encoders on the labelled Meiji clips (cross-camera Re-ID).

For every labelled clip, the tracks of the observation run (``--observe``,
the report of ``player_association_clips.py``) are matched to the labels; a
track takes the majority labelled person of its boxes and is *pure* when at
least ``--min-purity`` of its matched boxes carry that person. Each track is
sampled (``appearance.sampling``) and embedded by every ``--encoder``; its
appearance is the normalized mean of its sample embeddings.

Per encoder, over pure tracks with samples in different cameras:

* ``auc_players``: ROC AUC of cosine similarity, same player vs. different
  players.
* ``auc_all``: same, with non-player tracks added as negatives.
* ``top1``: for each pure player track and each other camera, whether the most
  similar track there (any role) is the same player.

Embeddings and per-pair similarities go to ``--report`` (``embeddings/<clip>.npz``,
``appearance.json``) for later association experiments. Runs on CPU by default.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.appearance.encoders import (
    ENCODER_CANDIDATES,
    build_encoder,
)
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    embed_samples,
    sample_tracks,
)
from src.tasks.player_association.evaluation import (
    CameraPrediction,
    ClipLabels,
    match_to_labels,
)
from src.tasks.player_association.evaluation.dataset_labels import discover_labels
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec


def roc_auc(positive: NDArray[np.float64], negative: NDArray[np.float64]) -> float | None:
    """Probability that a positive scores above a negative (ties count half)."""
    if not len(positive) or not len(negative):
        return None
    greater = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(greater + ties / 2)


def clip_tracks(dataset: Path, observe: Path, clip_id: str) -> tuple[Any, dict[str, PersonTrackingOutput]]:
    video_id, clip_name = clip_id.split("/")
    manifest = ClipManifest.load(dataset / "videos" / video_id / "clips" / clip_name)
    source = build_clip_source(tuple(manifest.media_path(c) for c in manifest.camera_ids), tuple(manifest.camera_ids), clip_id=manifest.clip_id)
    store = ClipStore(observe / "stores" / clip_id, json_value(source))
    tracks = {}
    for camera in source.camera_ids:
        reference = store.active(f"person_tracking/{camera}")
        if reference is None:
            raise RuntimeError(f"{clip_id} has no person_tracking/{camera} in {observe}")
        tracks[camera] = store.load(reference, ArtifactCodec(PersonTrackingOutput))
    return source, tracks


def track_identities(labels: ClipLabels, tracks: dict[str, PersonTrackingOutput], min_purity: float) -> dict[tuple[str, int], dict[str, Any]]:
    """Majority labelled person, role and purity of every observed track."""
    predictions = {camera: CameraPrediction(t.track_ids, t.boxes_xyxy.astype(np.float64), t.observed, np.full(t.observed.shape, -1, np.int64))
                   for camera, t in tracks.items()}
    matched = match_to_labels(labels, predictions, .5)
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for camera, t in tracks.items():
        for row, track in enumerate(t.track_ids.tolist()):
            persons = matched[camera][row][matched[camera][row] >= 0]
            if not len(persons):
                result[(camera, track)] = {"person": None, "role": None, "purity": 0.0, "pure": False}
                continue
            values, counts = np.unique(persons, return_counts=True)
            person = int(values[counts.argmax()])
            purity = float(counts.max() / len(persons))
            result[(camera, track)] = {"person": labels.people[person].person_id, "role": labels.people[person].role,
                                       "purity": purity, "pure": purity >= min_purity}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Root holding ckpt/ and third_party/")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--observe", type=Path, required=True, help="Report directory of player_association_clips.py observe")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--encoder", action="append", choices=ENCODER_CANDIDATES, default=[])
    parser.add_argument("--min-purity", type=float, default=.9)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    names = args.encoder or list(ENCODER_CANDIDATES)
    encoders = [build_encoder(name, checkpoint_root=args.repo / "ckpt", external_root=args.repo / "third_party", device=args.device) for name in names]
    sampling = CropSamplingConfig()
    pairs: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    clips: dict[str, Any] = {}
    for path in discover_labels(args.dataset.resolve()):
        labels = ClipLabels.load(path)
        source, tracks = clip_tracks(args.dataset.resolve(), args.observe.resolve(), labels.clip_id)
        identities = track_identities(labels, tracks, args.min_purity)
        means: dict[str, dict[tuple[str, int], NDArray[np.float32]]] = {name: {} for name in names}
        arrays: dict[str, NDArray[Any]] = {}
        sampled: dict[str, Any] = {}
        for video in source.videos:
            t = tracks[video.camera_id]
            samples = sample_tracks(t.boxes_xyxy, t.observed, source.size, sampling)
            embedded = embed_samples(video.path, t.boxes_xyxy, samples, encoders)
            for row, track in enumerate(t.track_ids.tolist()):
                key = (video.camera_id, track)
                sampled[f"{video.camera_id}:{track}"] = {"frames": samples[row].frames.tolist(), "rejected": samples[row].rejected,
                                                         **identities[key]}
                arrays[f"{video.camera_id}_{track}_frames"] = samples[row].frames
                for name in names:
                    vectors = embedded[name][row]
                    arrays[f"{video.camera_id}_{track}_{name}"] = vectors
                    if len(vectors):
                        mean = vectors.mean(0)
                        means[name][key] = mean / np.linalg.norm(mean)
        (report / "embeddings" / labels.clip_id).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(report / "embeddings" / f"{labels.clip_id}.npz", **arrays)
        clips[labels.clip_id] = sampled
        for name in names:
            for a, b in combinations(sorted(means[name]), 2):
                if a[0] == b[0]:
                    continue
                pairs[name].append({"clip": labels.clip_id, "a": f"{a[0]}:{a[1]}", "b": f"{b[0]}:{b[1]}",
                                    "cosine": float(means[name][a] @ means[name][b]),
                                    "a_person": identities[a]["person"], "b_person": identities[b]["person"],
                                    "a_role": identities[a]["role"], "b_role": identities[b]["role"],
                                    "pure": identities[a]["pure"] and identities[b]["pure"]})
        print(json.dumps({"clip": labels.clip_id, "tracks_with_samples": {n: len(means[n]) for n in names}}), flush=True)
    summary = {}
    for name in names:
        pure = [p for p in pairs[name] if p["pure"]]
        players = [p for p in pure if p["a_role"] == p["b_role"] == "player"]
        same = np.asarray([p["cosine"] for p in players if p["a_person"] == p["b_person"]])
        different = np.asarray([p["cosine"] for p in players if p["a_person"] != p["b_person"]])
        with_non = np.asarray([p["cosine"] for p in pure if not (p["a_role"] == p["b_role"] == "player" and p["a_person"] == p["b_person"])])
        hits = total = 0
        per_clip: dict[str, dict[str, int]] = {}
        for clip_id in clips:
            clip_pairs = [p for p in pure if p["clip"] == clip_id]
            for anchor in {p[side] for p in clip_pairs for side in ("a", "b")}:
                rows = [p for p in clip_pairs if anchor in (p["a"], p["b"])]
                own = rows[0]["a_person"] if rows[0]["a"] == anchor else rows[0]["b_person"]
                role = rows[0]["a_role"] if rows[0]["a"] == anchor else rows[0]["b_role"]
                if role != "player":
                    continue
                for camera in {(p["b"] if p["a"] == anchor else p["a"]).split(":")[0] for p in rows}:
                    candidates = [(p["cosine"], p["b_person"] if p["a"] == anchor else p["a_person"]) for p in rows
                                  if (p["b"] if p["a"] == anchor else p["a"]).startswith(camera + ":")]
                    if not any(person == own for _, person in candidates):
                        continue
                    hit = max(candidates)[1] == own
                    hits += hit
                    total += 1
                    counts = per_clip.setdefault(clip_id, {"hits": 0, "total": 0})
                    counts["hits"] += hit
                    counts["total"] += 1
        summary[name] = {"auc_players": roc_auc(same, different), "auc_all": roc_auc(same, with_non),
                         "top1": hits / total if total else None, "top1_hits": hits, "top1_total": total, "top1_per_clip": per_clip,
                         "same_pairs": len(same), "different_player_pairs": len(different), "negatives_with_non_players": len(with_non),
                         "same_cosine_min": float(same.min()) if len(same) else None,
                         "different_cosine_max": float(different.max()) if len(different) else None}
        print(json.dumps({"encoder": name, **{k: v for k, v in summary[name].items() if k != "top1_per_clip"}}), flush=True)
    write_json_atomic(report / "appearance.json", {"schema": "player_association_appearance_v1", "encoders": names,
                                                   "sampling": json_value(sampling), "min_purity": args.min_purity,
                                                   "observe": str(args.observe.resolve()), "labels_dataset": str(args.dataset.resolve()),
                                                   "summary": summary, "tracks": clips, "pairs": pairs})


if __name__ == "__main__":
    main()
