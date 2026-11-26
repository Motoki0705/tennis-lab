"""Measure apparent ball footprints and prepare a fixed visual audit of Meiji 4cam."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

CAMERAS = ("cam0", "cam1", "cam2", "cam3")
RESOLUTIONS = ((1920, 1080), (1280, 720), (512, 288))
STATUS = ("observed", "interpolated", "candidate_supported", "estimated")
REFERENCE = (-6, -3, 3, 6)
AUDIT_OFFSETS = (-4, -2, 0, 2, 4)
NATIVE_WIDTH, NATIVE_HEIGHT = RESOLUTIONS[0]
LABEL_DIR = "clip_000-068_ball_positions"
SEED = 42


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        table = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        table.writeheader()
        table.writerows(rows)


def rank(purpose: str, key: tuple[str, str, int]) -> bytes:
    return hashlib.sha256(f"{SEED}/{purpose}/{key[0]}/{key[1]}/{key[2]}".encode()).digest()


def clip_xy(row: dict, width: int, height: int) -> tuple[float, float]:
    x, y = float(row["x"]), float(row["y"])
    if not all(math.isfinite(v) for v in (x, y)) or width <= 0 or height <= 0:
        raise ValueError("Invalid point or image extent")
    return x, y


def visibility_flag(value: object) -> str:
    value = str(value).casefold()
    if value in {"true", "yes", "visible", "verified"}:
        return "reported_visible"
    if value in {"false", "no"}:
        return "reported_no"
    if value in {"out_of_frame", "unverified"}:
        return value
    raise ValueError(f"Unexpected visibility value: {value!r}")


def read_labels(root: Path, clip_names: list[str]):
    data = {}
    videos = []
    distribution = Counter()
    observed_sample = []
    other_strata = defaultdict(list)
    for clip_name in clip_names:
        directory = root / "clips" / clip_name
        index_path = directory / "clip.json"
        meta = json.loads(index_path.read_bytes())
        annotation = root / LABEL_DIR / f"{clip_name}_ball_positions.json"
        raw = json.loads(annotation.read_bytes())
        count = meta["num_frames"]
        if (meta["clip_id"] != f"video_000/{clip_name}" or tuple(meta["camera_ids"]) != CAMERAS
                or (meta["width"], meta["height"]) != RESOLUTIONS[0]
                or set(raw["cameras"]) != {f"{clip_name}_{camera}" for camera in CAMERAS}
                or raw["metadata"]["frames_per_camera"] != count):
            raise ValueError(f"Clip or annotation metadata mismatch: {clip_name}")
        video_paths = {camera["camera_id"]: directory / camera["video"] for camera in meta["cameras"]}
        for camera in CAMERAS:
            label = raw["cameras"][f"{clip_name}_{camera}"]
            rows = label["frames"]
            if (label["width"], label["height"], label["frame_count"], len(rows)) != (*RESOLUTIONS[0], count, count):
                raise ValueError(f"Annotation shape mismatch: {clip_name}/{camera}")
            if not math.isclose(label["fps"], meta["fps"], rel_tol=1e-7):
                raise ValueError(f"Annotation FPS mismatch: {clip_name}/{camera}")
            full_observed = []
            for frame, row in enumerate(rows):
                if row["frame"] != frame or row["status"] not in STATUS:
                    raise ValueError(f"Invalid frame ID or status: {clip_name}/{camera}/{frame}")
                x, y = clip_xy(row, *RESOLUTIONS[0])
                inside = 0 <= x < NATIVE_WIDTH and 0 <= y < NATIVE_HEIGHT
                flag = visibility_flag(row["visible_in_frame"])
                distribution[camera, row["status"], "inside" if inside else "outside", flag] += 1
                key = clip_name, camera, frame
                if inside and row["status"] == "observed":
                    if -REFERENCE[0] <= frame < count - REFERENCE[-1]:
                        full_observed.append(key)
                else:
                    other_strata[camera, "other_inside" if inside else "outside_coord"].append(key)
            selected = sorted(full_observed, key=lambda key: rank("footprint", key))[:8]
            observed_sample.extend(selected)
            video = video_paths[camera]
            videos.append(dict(
                clip=clip_name, camera=camera, clip_json_sha256=sha(index_path),
                label_sha256=sha(annotation), video_path=video.relative_to(root).as_posix(),
                video_sha256=sha(video), frames=count, fps=meta["fps"],
                container_mbps=round(video.stat().st_size * 8 * meta["fps"] / count / 1e6, 4),
                eligible_observed=len(full_observed), sampled_observed=len(selected),
            ))
            data[clip_name, camera] = rows
    audit = []
    for camera in CAMERAS:
        strata = {
            "observed_inside": [key for key in observed_sample if key[1] == camera],
            "outside_coord": other_strata[camera, "outside_coord"],
            "other_inside": other_strata[camera, "other_inside"],
        }
        for stratum, keys in strata.items():
            count = 8 if stratum == "observed_inside" else 4
            ranked = sorted(keys, key=lambda key: rank("audit", key))
            for clip, cam, frame in ranked[:count]:
                audit.append(dict(clip=clip, camera=cam, frame=frame, stratum=stratum))
    audit.sort(key=lambda r: (r["camera"], r["stratum"], r["clip"], r["frame"]))
    for i, item in enumerate(audit, 1):
        item["id"] = f"{i:02d}"
    return data, videos, distribution, observed_sample, audit


def measure(gray: np.ndarray, references: list[np.ndarray], xy: tuple[float, float], scale: float) -> tuple:
    """Measure an annotation-local foreground component in the sampled image grid."""
    height, width = gray.shape
    # OpenCV resize maps the source's pixel centers onto the destination grid.
    x, y = ((coord + 0.5) * scale - 0.5 for coord in xy)
    u, v = round(x), round(y)
    half = max(7, math.ceil(28 * scale))
    left, right = max(0, u - half), min(width, u + half + 1)
    top, bottom = max(0, v - half), min(height, v + half + 1)
    crop = np.s_[top:bottom, left:right]
    background = np.median([reference[crop] for reference in references], axis=0)
    residual = gray[crop].astype(np.float32) - background
    absolute = np.abs(residual)
    median = float(np.median(absolute))
    noise = float(1.4826 * np.median(np.abs(absolute - median)))
    yy, xx = np.ogrid[top:bottom, left:right]
    search = (xx - x) ** 2 + (yy - y) ** 2 <= max(2, 12 * scale) ** 2
    if not search.any():
        return "no_search", None, None, None, None, noise, None
    y_peak, x_peak = np.unravel_index(np.where(search, absolute, -1).argmax(), absolute.shape)
    peak = float(absolute[y_peak, x_peak])
    floor = max(6., median + 3.5 * noise)
    if peak < floor:
        return "low_contrast", None, None, None, peak, noise, None
    signed = residual if residual[y_peak, x_peak] >= 0 else -residual
    mask = (signed >= max(floor, 0.30 * peak)).astype(np.uint8)
    _, components, stats, centroids = cv2.connectedComponentsWithStats(mask)
    label = components[y_peak, x_peak]
    x0, y0, w, h, area = stats[label]
    if x0 == 0 or y0 == 0 or x0 + w == mask.shape[1] or y0 + h == mask.shape[0]:
        return "component_at_crop_edge", None, None, None, peak, noise, None
    if math.dist(centroids[label], (x - left, y - top)) > max(2, 12 * scale):
        return "component_far", None, None, None, peak, noise, None
    if area < 3:
        return "insufficient_pixels", None, None, None, peak, noise, None
    if max(w, h) > max(5, 44 * scale):
        return "component_too_large", None, None, None, peak, noise, None
    coords = np.column_stack(np.nonzero(components == label)).astype(np.float64)
    centered = coords - coords.mean(axis=0)
    minor, major = np.linalg.eigvalsh((centered.T @ centered) / area)
    # Disk D = 4*sqrt(var); independent uniform line variance is L²/12.
    diameter = 4 * math.sqrt(minor + 1 / 12)
    major_axis = 4 * math.sqrt(major + 1 / 12)
    if diameter > max(4, 26 * scale):
        return "component_too_large", None, None, None, peak, noise, None
    blur = math.sqrt(12 * max(0., major - minor))
    centre_offset = centroids[label] - (x - left, y - top)
    return "measured", diameter, blur, major_axis, peak, noise, centre_offset


def patch_for_audit(image: np.ndarray, row: dict) -> np.ndarray:
    x = min(NATIVE_WIDTH - 1, max(0, round(row["x"])))
    y = min(NATIVE_HEIGHT - 1, max(0, round(row["y"])))
    left = min(NATIVE_WIDTH - 64, max(0, x - 32))
    top = min(NATIVE_HEIGHT - 64, max(0, y - 32))
    patch = cv2.resize(image[top:top + 64, left:left + 64], (128, 128), interpolation=cv2.INTER_NEAREST)
    u, v = 2 * (x - left), 2 * (y - top)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.line(patch, (u + 11 * dx, v + 11 * dy), (u + 19 * dx, v + 19 * dy), (255, 220, 20), 1)
    return patch


def title(image: np.ndarray, label: str) -> np.ndarray:
    result = cv2.copyMakeBorder(image, 24, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(result, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def measure_video(job):
    root, output, video, rows, sample, audit = job
    cv2.setNumThreads(1)
    count = video["frames"]
    read_gray = {t + delta for t in sample for delta in (0, *REFERENCE)}
    read_bgr = {min(count - 1, max(0, e["frame"] + d)) for e in audit for d in AUDIT_OFFSETS}
    gray, bgr = {}, {}
    path = root / video["video_path"]
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        if (not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != count
                or not math.isclose(capture.get(cv2.CAP_PROP_FPS), video["fps"], rel_tol=1e-6)):
            raise ValueError(f"Video header disagrees with clip.json: {path}")
        for frame in range(count):
            if not capture.grab():
                raise ValueError(f"Video ended at frame {frame}, expected {count}: {path}")
            if frame not in read_gray and frame not in read_bgr:
                continue
            ok, image = capture.retrieve()
            if not ok or image.shape != (NATIVE_HEIGHT, NATIVE_WIDTH, 3):
                raise ValueError(f"Unexpected decoded frame: {path} #{frame}")
            if frame in read_gray:
                native = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray[frame] = [native, *(cv2.resize(native, size, interpolation=cv2.INTER_LINEAR) for size in RESOLUTIONS[1:])]
            if frame in read_bgr:
                bgr[frame] = image
        if capture.grab():
            raise ValueError(f"Video contains more than {count} frames: {path}")
    finally:
        capture.release()

    measurements = []
    for frame in sample:
        row = rows[frame]
        for i, (width, height) in enumerate(RESOLUTIONS):
            status, diameter, blur, major, peak, noise, centre_offset = measure(
                gray[frame][i], [gray[frame + delta][i] for delta in REFERENCE],
                clip_xy(row, width, height), width / NATIVE_WIDTH,
            )
            measurements.append(dict(
                clip=video["clip"], camera=video["camera"], frame=frame, width=width, height=height,
                measurement=status, diameter_px=rounded(diameter), blur_proxy_px=rounded(blur),
                major_axis_px=rounded(major), peak_delta=rounded(peak), noise_mad=rounded(noise),
                centre_dx_px=rounded(None if centre_offset is None else centre_offset[0]),
                centre_dy_px=rounded(None if centre_offset is None else centre_offset[1]),
            ))
    for item in audit:
        frame = item["frame"]
        frames = [min(count - 1, max(0, frame + delta)) for delta in AUDIT_OFFSETS]
        patches = [title(patch_for_audit(bgr[t], rows[t]), str(t)) for t in frames]
        stem = output / "audit-media" / f"{item['id']}-{video['camera']}"
        save_image(stem.with_suffix(".png"), np.hstack(patches))
        overview = cv2.resize(bgr[frame], (480, 270), interpolation=cv2.INTER_AREA)
        x, y = (round(rows[frame][coordinate] / 4) for coordinate in ("x", "y"))
        cv2.circle(overview, (x, y), 8, (255, 220, 20), 1)
        save_image(stem.with_suffix(".jpg"), title(overview, f"{video['clip']} {video['camera']} frame {frame}"))
    return measurements


def rounded(value):
    return None if value is None else round(value, 4)


def audit_boards(output: Path, audit: list[dict]) -> None:
    groups = defaultdict(list)
    for item in audit:
        rows, overviews = [], []
        for camera in CAMERAS:
            stem = output / "audit-media" / f"{item['id']}-{camera}"
            row = cv2.imread(str(stem.with_suffix(".png")))
            overview = cv2.imread(str(stem.with_suffix(".jpg")))
            if row is None or overview is None:
                raise ValueError(f"Missing audit material for {stem.name}")
            label = f"{item['id']} {item['clip']} {camera} @ {item['frame']}"
            rows.append(title(row, label))
            overviews.append(overview)
            if camera == item["camera"]:
                groups[camera, item["stratum"]].append(title(row, label))
        save_image(output / "audit-media" / f"case-{item['id']}-context.png", np.vstack(rows))
        save_image(output / "audit-media" / f"case-{item['id']}-overview.jpg",
                   np.vstack([np.hstack(overviews[:2]), np.hstack(overviews[2:])]))
    for (camera, stratum), rows in groups.items():
        save_image(output / "audit-media" / f"{camera}-{stratum}.png", np.vstack(rows))


def run(root: Path, output: Path, workers: int, smoke: bool) -> None:
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit-media").mkdir(exist_ok=True)
    clips = ["clip_000"] if smoke else [f"clip_{i:03d}" for i in range(69)]
    labels, videos, distribution, sample, audit = read_labels(root, clips)
    if not smoke and (len(audit) != 64 or sum(distribution.values()) != 184_912):
        raise ValueError("Unexpected full dataset or audit size")
    shot_by_clip = defaultdict(list)
    sample_by_camera = defaultdict(list)
    for item in audit:
        shot_by_clip[item["clip"]].append(item)
    for clip, camera, frame in sample:
        sample_by_camera[clip, camera].append(frame)
    jobs = [(root, output, video, labels[video["clip"], video["camera"]],
             sorted(sample_by_camera[video["clip"], video["camera"]]), shot_by_clip[video["clip"]])
            for video in videos]
    data = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, entries in enumerate(executor.map(measure_video, jobs), 1):
            for entry in entries:
                data[entry["camera"]].append(entry)
            if i % 20 == 0:
                print(f"Decoded {i}/{len(jobs)} camera-clips", flush=True)
    if sum(map(len, data.values())) != len(sample) * len(RESOLUTIONS):
        raise ValueError("Missing auto-measurement results")
    for camera in CAMERAS:
        rows = data[camera]
        write_csv(output / f"footprints-{camera}.csv", rows[0].keys(), rows)
    write_csv(output / "media.csv", videos[0].keys(), videos)
    flags = [
        dict(camera=camera, status=status, centre=centre, visibility_flag=flag, count=count)
        for (camera, status, centre, flag), count in sorted(distribution.items())
    ]
    write_csv(output / "visibility-flags.csv", flags[0].keys(), flags)
    write_csv(output / "audit-plan.csv", audit[0].keys(), audit)
    audit_boards(output, audit)
    summary = dict(frames=sum(distribution.values()), video_count=len(videos), audit_count=len(audit),
                   footprint_count=len(sample), opencv=cv2.__version__, seconds=time.perf_counter() - start)
    (output / "size-execution.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def self_check() -> None:
    background = np.full((84, 96), 45, dtype=np.uint8)
    centre = (49, 37)
    disk = background.copy()
    cv2.circle(disk, centre, 5, 210, -1)
    still = measure(disk, [background] * 4, centre, 1.0)
    assert still[0] == "measured" and 9.5 < still[1] < 10.7 and still[2] == 0
    blurred = cv2.blur(disk, (15, 1))
    trace = measure(blurred, [background] * 4, centre, 1.0)
    assert trace[0] == "measured" and 7 < trace[1] < 12 and trace[2] > 12
    assert measure(background, [background] * 4, centre, 1.0)[0] == "low_contrast"
    assert visibility_flag(False) == "reported_no" and visibility_flag("out_of_frame") == "out_of_frame"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.root is None or args.output is None:
        parser.error("--root and --output are required")
    run(args.root, args.output, args.workers, args.smoke)


if __name__ == "__main__":
    main()
