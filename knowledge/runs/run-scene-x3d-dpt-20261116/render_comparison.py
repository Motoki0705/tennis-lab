"""CPU-only four-camera side-by-side detections and masked 3D trajectories."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH
from src.utils.video import VideoWriter

COLORS = ((40, 170, 255), (255, 220, 30))  # BGR, i935 / X3D


def text(image, value, point, *, color=(245, 245, 245), scale=.55):
    cv2.putText(image, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def project(points):
    points = np.asarray(points)
    return np.column_stack((300 + points[:, 0] * 22 + points[:, 1] * 11,
                            260 + points[:, 0] * 9 - points[:, 1] * 16 - points[:, 2] * 32)).round().astype(np.int32)


def court():
    image = np.full((540, 600, 3), (40, 65, 42), np.uint8)
    x, y = HALF_DOUBLES_WIDTH, HALF_LENGTH
    outline = project([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0]])
    cv2.polylines(image, [outline], True, (185, 185, 185), 1, cv2.LINE_AA)
    cv2.polylines(image, [project([[-x, 0, 0], [x, 0, 0]])], False, (230, 230, 230), 2, cv2.LINE_AA)
    return image


def camera_video(source, meta, camera, camera_row, arrays, output):
    paths = dict(zip(meta["camera_ids"], meta["video_paths"], strict=True))
    reader = cv2.VideoCapture(str(source / paths[camera]))
    with VideoWriter(output, fps=meta["fps"], crf=18) as writer:
        for frame in range(meta["num_frames"]):
            ok, original = reader.read()
            if not ok:
                raise ValueError("Video ended before its completed export timeline")
            panels = []
            for variant, (data, color) in enumerate(zip(arrays, COLORS)):
                panel = original.copy()
                previous = None
                for index in range(max(0, frame - 12), frame + 1):
                    if not data["observed"][camera_row, index]:
                        previous = None
                        continue
                    point = tuple(np.rint(data["uv_px"][camera_row, index]).astype(int))
                    if previous is not None:
                        cv2.line(panel, previous, point, color, 2, cv2.LINE_AA)
                    previous = point
                    if index == frame:
                        cv2.circle(panel, point, 12, color, 2, cv2.LINE_AA)
                label = "i935" if variant == 0 else "X3D/CSN + DPT T32"
                state = "detected" if data["observed"][camera_row, frame] else "absent / rejected"
                text(panel, f"{camera} | {label} | {state} | frame {frame}", (20, 38), color=color, scale=1.)
                panels.append(panel)
            writer.write_frame(cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_BGR2RGB))
    reader.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--camera-videos", action="store_true")
    args = parser.parse_args()
    for clip in args.clips:
        source = args.dataset / "videos/video_000/clips" / clip
        meta = json.loads((source / "clip.json").read_text())
        arrays, metrics = [], []
        for variant in ("i935", "x3d"):
            folder = args.results / variant / clip
            with np.load(folder / "comparison-arrays.npz") as data:
                arrays.append({k: data[k] for k in data.files})
            metrics.append(json.loads((folder / "metrics.json").read_text()))
        cameras = metrics[0]["camera_ids"]
        if cameras != metrics[1]["camera_ids"]:
            raise ValueError("Comparison camera sets differ")
        paths = dict(zip(meta["camera_ids"], meta["video_paths"], strict=True))
        readers = [cv2.VideoCapture(str(source / paths[camera])) for camera in cameras]
        output = args.results / f"{clip}-i935-vs-x3d.mp4"
        with VideoWriter(output, fps=meta["fps"], crf=18) as writer:
            for frame in range(meta["num_frames"]):
                canvas = np.zeros((1080, 1560, 3), np.uint8)
                for camera, reader in enumerate(readers):
                    ok, rgb = reader.read()
                    if not ok:
                        raise ValueError("Video ended before its completed export timeline")
                    small = cv2.resize(rgb, (480, 270), interpolation=cv2.INTER_AREA)
                    for variant, (data, record, color) in enumerate(zip(arrays, metrics, COLORS)):
                        panel = small.copy()
                        recent = range(max(0, frame - 12), frame + 1)
                        previous = None
                        for index in recent:
                            if not data["observed"][camera, index]:
                                previous = None
                                continue
                            point = tuple(np.rint(data["uv_px"][camera, index] * [480 / meta["width"], 270 / meta["height"]]).astype(int))
                            if previous is not None:
                                cv2.line(panel, previous, point, color, 2, cv2.LINE_AA)
                            previous = point
                            if index == frame:
                                cv2.circle(panel, point, 6, color, 2, cv2.LINE_AA)
                        label = "i935" if variant == 0 else "X3D/CSN + DPT T32"
                        confidence = float(data["confidence"][camera, frame])
                        text(panel, f"{cameras[camera]} | {label} | score {confidence:.3f}", (8, 22), color=color)
                        canvas[camera * 270:(camera + 1) * 270, variant * 480:(variant + 1) * 480] = panel
                for variant, (data, record, color) in enumerate(zip(arrays, metrics, COLORS)):
                    panel = court()
                    label = "i935 3D" if variant == 0 else "X3D/CSN + DPT T32 3D"
                    text(panel, label, (16, 24), color=color)
                    text(panel, f"{clip} | frame {frame}/{meta['num_frames'] - 1} | {frame / meta['fps']:.2f}s", (16, 50))
                    if "ball_3d_valid" in data:
                        valid = data["ball_3d_valid"]
                        xyz = data["ball_3d"]
                        previous = None
                        for index in range(max(0, frame - 30), frame + 1):
                            if not valid[index]:
                                previous = None
                                continue
                            point = tuple(project(xyz[index:index + 1])[0])
                            if previous is not None:
                                cv2.line(panel, previous, point, color, 2, cv2.LINE_AA)
                            previous = point
                        if valid[frame]:
                            point = tuple(project(xyz[frame:frame + 1])[0])
                            cv2.circle(panel, point, 7, color, -1, cv2.LINE_AA)
                            text(panel, f"height {xyz[frame, 2]:.2f} m", (16, 450), color=color)
                        else:
                            text(panel, "3D missing", (16, 450), color=color)
                        text(panel, f"valid 3D {record['ball_3d_valid_frames']}/{meta['num_frames']}", (16, 478))
                        error = record["reprojection_mean_px"]
                        text(panel, f"mean reprojection {error:.2f} source px" if error is not None else "No valid reprojection", (16, 504))
                    else:
                        text(panel, "3D reconstruction unavailable", (16, 450), color=color)
                        text(panel, record["failure"]["reason"], (16, 478), scale=.45)
                    canvas[variant * 540:(variant + 1) * 540, 960:] = panel
                writer.write_frame(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        for reader in readers:
            reader.release()
        print(output, flush=True)
        if args.camera_videos:
            for row, camera in enumerate(cameras):
                path = args.results / f"{clip}-{camera}-i935-vs-x3d-2d.mp4"
                camera_video(source, meta, camera, row, arrays, path)
                print(path, flush=True)


if __name__ == "__main__":
    main()
