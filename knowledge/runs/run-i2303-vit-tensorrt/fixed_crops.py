"""Freeze ViTPose detector crops and HMR2's actual body-view input crops."""

import argparse
import json
from contextlib import closing
from pathlib import Path

import numpy as np
from inputs import HERE as CLIP_BUNDLE
from inputs import SPLITS, check_split, sha256_file
from pin_inputs import read_component

HERE = Path(__file__).resolve().parent
PINS = CLIP_BUNDLE / "inputs.json"
DATASET = Path("data/tennis_multiview/processed/meiji_4cam/dataset")
BODY_PINS = HERE / "body-boxes.json"
LOCK = HERE / "tensor-lock.json"
SHAPES = {"vitpose_h": (3, 256, 192), "hmr2": (3, 256, 256)}
CLIPS = ("clip_000", "clip_037", "clip_027")
PROFILE_ROW = ("video_000/clip_000", "cam0", 178, 0)


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def body_rows(payload, read_array, streams):
    rows = []
    for body in payload["selections"]:
        cam = body["camera_id"]
        target = streams[cam]["sampled_frames"]
        for segment, request in enumerate(body["requests"]):
            frames = read_array(request["source_frames"])
            boxes = read_array(request["boxes_xys"])
            poses = read_array(request["keypoints"])
            if (
                frames.ndim != 1
                or not np.all(np.diff(frames) > 0)
                or boxes.shape != (len(frames), 3)
                or poses.shape != (len(frames), 17, 3)
                or not np.isfinite(boxes).all()
                or not np.all(boxes[:, 2] > 0)
            ):
                raise ValueError(
                    f"Invalid body-view segment {body['person_id']}/{segment}"
                )
            for i in np.flatnonzero(np.isin(frames, target)):
                rows.append(
                    {
                        "camera": cam,
                        "frame": int(frames[i]),
                        "person": body["person_id"],
                        "segment": segment,
                        "boxes_xys": boxes[i].tolist(),
                        "any_positive_joint": bool(np.any(poses[i, :, 2] > 0)),
                    }
                )
    return sorted(rows, key=lambda row: (row["camera"], row["frame"], row["person"]))


def pin_body(book, stores, output):
    split_ids = check_split(book)
    pinned = []
    for clip, store in zip(book["clips"], stores, strict=True):
        scene = read_json(store / "scene.json")
        source = scene["source"]
        streams = {s["camera"]: s for s in clip["streams"]}
        if source["clip_id"] != clip["clip"] or {
            video["camera_id"]: (video["sha256"], video["num_frames"])
            for video in source["videos"]
        } != {cam: (s["sha256"], clip["num_frames"]) for cam, s in streams.items()}:
            raise ValueError(f"Body-view store differs from fixed source: {store}")
        ref = scene["artifacts"]["body_view_selection"]
        payload, array = read_component(store, ref)
        rows = body_rows(payload, array, streams)
        if not rows:
            raise ValueError(f"No body-view inputs on the fixed frames: {store}")
        pinned.append(
            {"clip": clip["clip"], "artifact_sha256": ref["sha256"], "rows": rows}
        )
    write_json(output, {"split": split_ids, "clips": pinned})


def tensors_info(path):
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": sha256_file(path),
    }


def addresses_for(model, book, bodies, split):
    for clip in book["clips"]:
        if clip["split"] != split:
            continue
        if model == "vitpose_h":
            for stream in clip["streams"]:
                for row in stream["crops"]:
                    yield (
                        clip["clip"],
                        stream["camera"],
                        row["frame"],
                        row["player"],
                        row["detection_row"],
                    )
        else:
            for row in bodies[clip["clip"]]["rows"]:
                yield (
                    clip["clip"],
                    row["camera"],
                    row["frame"],
                    row["person"],
                    row["segment"],
                )


def check_manifests(book, pinned):
    split = check_split(book)
    bodies = {clip["clip"]: clip for clip in pinned["clips"]}
    if (
        pinned["split"] != split
        or len(bodies) != len(pinned["clips"])
        or set(bodies) != {clip["clip"] for clip in book["clips"]}
    ):
        raise ValueError("Body boxes differ from the fixed split")
    return split, bodies


def prepare(dataset, body_pins, output, *, record_lock):
    import cv2
    import torch

    from src.submodules.models._base.crops import iter_frame_person_crops
    from src.tasks.person_tracking.features import DEFAULT_CONFIG, pose_boxes
    from src.utils.video import OpenCVVideoFrameReader

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    book, pinned = read_json(PINS), read_json(body_pins)
    split, bodies = check_manifests(book, pinned)
    result = {model: {kind: [] for kind in SPLITS} for model in SHAPES}
    boxes = {model: {kind: [] for kind in SPLITS} for model in SHAPES}
    for clip in book["clips"]:
        kind = clip["split"]
        for stream in clip["streams"]:
            media = dataset / stream["media"]
            if sha256_file(media) != stream["sha256"]:
                raise ValueError(f"Camera media changed: {media}")
            body = [
                row
                for row in bodies[clip["clip"]]["rows"]
                if row["camera"] == stream["camera"]
            ]
            detections = stream["crops"]
            for model, rows in (("vitpose_h", detections), ("hmr2", body)):
                if not rows:
                    continue
                frames = torch.tensor([row["frame"] for row in rows], dtype=torch.int64)
                if not set(frames.tolist()) <= set(stream["sampled_frames"]):
                    raise ValueError(f"Crop frame outside the selection: {media}")
                if model == "vitpose_h":
                    xys = pose_boxes(
                        np.array([row["xyxy"] for row in rows], dtype=np.float32)
                    )
                else:
                    xys = torch.tensor(
                        [row["boxes_xys"] for row in rows], dtype=torch.float32
                    )
                with closing(
                    iter(
                        OpenCVVideoFrameReader(media, max_frames=int(frames.max()) + 1)
                    )
                ) as reader:
                    for pixels, used_boxes in iter_frame_person_crops(
                        ((packet.index, packet.frame) for packet in reader),
                        xys,
                        frames,
                        batch_size=8,
                    ):
                        if model == "vitpose_h":
                            pixels = pixels[:, :, :, 32:224]
                        result[model][kind].append(pixels.contiguous().numpy())
                        boxes[model][kind].append(used_boxes.numpy())

    receipt = {
        "split": split,
        "cross_split_shared_raw_frame_intervals": 0,
        "cross_split_shared_media": 0,
        "opencv": cv2.__version__,
        "pins_sha256": sha256_file(PINS),
        "body_pins_sha256": sha256_file(body_pins),
        "vitpose_bbox_enlarge": DEFAULT_CONFIG.bbox_enlarge,
        "models": {},
    }
    for model, shape in SHAPES.items():
        root = output / model
        root.mkdir(parents=True, exist_ok=False)
        info = {}
        for kind in SPLITS:
            data = np.concatenate(result[model][kind])
            xys = np.concatenate(boxes[model][kind])
            addresses = list(addresses_for(model, book, bodies, kind))
            if (
                len(set(addresses)) != len(addresses)
                or data.shape != (len(addresses), *shape)
                or xys.shape != (len(addresses), 3)
                or data.dtype != np.float32
                or not np.isfinite(data).all()
            ):
                raise ValueError(f"Invalid {model}/{kind} crop tensors")
            np.save(root / f"{kind}.npy", data, allow_pickle=False)
            np.save(root / f"{kind}-boxes.npy", xys, allow_pickle=False)
            info[kind] = tensors_info(root / f"{kind}.npy")
            info[f"{kind}-boxes"] = tensors_info(root / f"{kind}-boxes.npy")
        address = next(
            address
            for address in addresses_for(model, book, bodies, "eval")
            if address[:4] == PROFILE_ROW
        )
        receipt["models"][model] = {"tensors": info, "profile_address": list(address)}
    if not record_lock and receipt != read_json(LOCK):
        raise ValueError("Prepared tensors differ from tensor-lock.json")
    write_json(output / "tensors.json", receipt)


def verify(model, directory):
    lock = read_json(LOCK)
    if read_json(directory.parent / "tensors.json") != lock:
        raise ValueError("Input receipt differs from the fixed tensor lock")
    if (
        sha256_file(PINS) != lock["pins_sha256"]
        or sha256_file(BODY_PINS) != lock["body_pins_sha256"]
    ):
        raise ValueError("Crop addresses differ from the fixed tensor lock")
    spec = lock["models"][model]
    for key, info in spec["tensors"].items():
        if tensors_info(directory / f"{key}.npy") != info:
            raise ValueError(f"Fixed {model}/{key} tensor changed")
    book = read_json(PINS)
    _split, bodies = check_manifests(book, read_json(BODY_PINS))
    return spec, list(addresses_for(model, book, bodies, "eval"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--body-pins", type=Path, default=BODY_PINS)
    parser.add_argument("--pin-body-from", nargs=3, type=Path, metavar="STORE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--record-lock",
        action="store_true",
        help="write a candidate receipt to --output",
    )
    args = parser.parse_args()
    if args.pin_body_from:
        pin_body(read_json(PINS), args.pin_body_from, args.output)
    else:
        prepare(args.dataset, args.body_pins, args.output, record_lock=args.record_lock)


if __name__ == "__main__":
    main()
