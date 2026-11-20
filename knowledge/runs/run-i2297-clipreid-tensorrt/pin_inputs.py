"""Capture experiment crop coordinates from the named completed scene stores."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from inputs import CAMERAS, check_split, sha256_file

FRAMES = {
    "clip_000": [32, 73, 112, 178, 221, 260, 301, 328],
    "clip_037": [22, 52, 101, 146, 172, 207, 243, 270],
    "clip_027": [
        8,
        23,
        37,
        52,
        66,
        81,
        95,
        110,
        124,
        139,
        153,
        168,
        182,
        197,
        211,
        226,
    ],
}


def read_component(store, reference):
    path = store / reference["path"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError(f"Component receipt changed: {path}")
    document = json.loads(raw)

    def array(pointer):
        name = pointer["array"]
        if sha256_file(path.parent / name) != document["arrays"][name]["sha256"]:
            raise ValueError(f"Component array changed: {path.parent / name}")
        return np.load(path.parent / name, allow_pickle=False)

    return document["payload"], array


def pin_clip(dataset, clip_dir, store, split):
    meta = json.loads((dataset / clip_dir / "clip.json").read_text())
    clip = meta["clip_name"]
    scene = json.loads((store / "scene.json").read_text())
    source = scene["source"]
    components = scene["artifacts"]
    if source["clip_id"] != meta["clip_id"]:
        raise ValueError(f"Store clip does not match {meta['clip_id']}: {store}")
    video_hashes = {row["camera_id"]: row["sha256"] for row in source["videos"]}
    if split == "eval":
        association = components["player_association"]
        payload, array = read_component(store, association)
        association_ids = array(payload["player_ids"])
        local_ids = array(payload["local_track_ids"])
        indices = {cam: i for i, cam in enumerate(payload["camera_ids"])}
    streams = []
    for cam in CAMERAS:
        ref = components[f"player_selection/{cam}"]
        payload, array = read_component(store, ref)
        tracks = payload["tracks"]
        boxes = array(tracks["boxes_xyxy"])
        rows = array(tracks["evidence"]["detection_rows"])
        observed = array(tracks["observed"])
        valid = array(tracks["evidence"]["appearance_valid"])
        if (valid & ~observed).any() or not np.array_equal(observed, rows >= 0):
            raise ValueError(f"Inconsistent detection evidence in {clip}/{cam}")
        if split == "eval":
            track_ids = array(tracks["track_ids"])
            by_track = {
                int(track): ids
                for track, ids in zip(
                    local_ids[indices[cam]], association_ids[indices[cam]], strict=True
                )
                if track >= 0
            }
        camera = next(c for c in meta["cameras"] if c["camera_id"] == cam)
        video = dataset / clip_dir / camera["video"]
        if sha256_file(video) != video_hashes[cam]:
            raise ValueError(f"Clip media changed: {video}")
        crops = []
        for frame in FRAMES[clip]:
            for track_row in np.flatnonzero(valid[:, frame]):
                player = (
                    None
                    if split == "calibration"
                    else int(by_track[int(track_ids[track_row])][frame])
                )
                if player == -1:
                    continue
                crops.append(
                    {
                        "frame": frame,
                        "detection_row": int(rows[track_row, frame]),
                        "player": player,
                        "xyxy": boxes[track_row, frame].tolist(),
                    }
                )
        streams.append(
            {
                "camera": cam,
                "media": str(clip_dir / camera["video"]),
                "sha256": video_hashes[cam],
                "source_frames_inclusive": [
                    camera["source_frame_start"],
                    camera["source_frame_end"],
                ],
                "selection_artifact_sha256": ref["sha256"],
                "sampled_frames": FRAMES[clip],
                "crops": crops,
            }
        )
    result = {
        "clip": meta["clip_id"],
        "split": split,
        "num_frames": meta["num_frames"],
        "streams": streams,
    }
    if split == "eval":
        result["association_artifact_sha256"] = association["sha256"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--eval-000-store", type=Path, required=True)
    parser.add_argument("--eval-037-store", type=Path, required=True)
    parser.add_argument("--calibration-027-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads((args.dataset / "dataset.json").read_text())
    clips = {c["clip_name"]: Path(c["path"]) for c in metadata["clips"]}
    book = {
        "dataset": metadata["dataset_id"],
        "clips": [
            pin_clip(args.dataset, clips[name], store, split)
            for name, store, split in [
                ("clip_000", args.eval_000_store, "eval"),
                ("clip_037", args.eval_037_store, "eval"),
                ("clip_027", args.calibration_027_store, "calibration"),
            ]
        ],
    }
    check_split(book)
    with args.output.open("x") as output:
        json.dump(book, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
