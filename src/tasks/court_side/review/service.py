"""CPU-only frame access and saved evidence lookup; no artifact writers."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2

from src.tasks.court_side.review.diagnostics import SavedDiagnostics
from src.tasks.court_side.review.store import StoreCase
from src.utils.checksum import dual_sha256

POINT_KINDS = ("unresolved", "observed", "interpolated", "occlusion_estimated")


class ReviewService:
    def __init__(
        self,
        store: Path,
        *,
        diagnostics: Path | None = None,
        comparison_stores: tuple[Path, ...] = (),
    ) -> None:
        self.cases = [StoreCase(path) for path in (store, *comparison_stores)]
        self.decode_lock = threading.Lock()
        self.verified_videos: set[tuple[Path, str]] = set()
        self.image_cache: dict[tuple[int, str, int], bytes] = {}
        if diagnostics is not None:
            case = self.cases[0]
            case.diagnostics = SavedDiagnostics(diagnostics, case)
            case.decision = case.diagnostics.decision

    def case(self, case_id: int) -> StoreCase:
        if not 0 <= case_id < len(self.cases):
            raise IndexError("Unknown review case")
        case = self.cases[case_id]
        case.assert_unchanged()
        return case

    def catalog(self) -> dict[str, Any]:
        return {
            "cases": [
                {"id": i, **self.case(i).summary()} for i in range(len(self.cases))
            ]
        }

    def frame(self, case_id: int, frame: int) -> dict[str, Any]:
        case = self.case(case_id)
        if not 0 <= frame < case.frames:
            raise IndexError("Frame outside the saved source timeline")
        cameras = []
        for video in case.videos:
            camera = video["camera_id"]
            points = case.points[camera]
            point = None
            if points is not None:
                kind = int(points["kinds"][frame])
                uv = points["uv"][frame].tolist() if kind != 0 else None
                point = {
                    "uv_px": uv,
                    "kind": POINT_KINDS[kind],
                    "observed": bool(points["observed"][frame]),
                    "confidence": float(points["confidence"][frame]),
                    "in_frame": None
                    if uv is None
                    else 0 <= uv[0] < video["width"] and 0 <= uv[1] < video["height"],
                }
            cameras.append(
                {
                    "id": camera,
                    "width": video["width"],
                    "height": video["height"],
                    "image_available": Path(video["path"]).is_file(),
                    "point": point,
                }
            )
        evidence = (
            {"state": "not_saved", "scores": None}
            if case.diagnostics is None
            else case.diagnostics.frame(frame)
        )
        return {
            "frame": frame,
            "seconds": frame / case.fps,
            "cameras": cameras,
            "evidence": evidence,
            "observing_cameras": [
                c["id"]
                for c in cameras
                if c["point"] is not None and c["point"]["observed"]
            ],
        }

    def timeline(self, case_id: int) -> dict[str, Any]:
        case = self.case(case_id)
        evidence = (
            []
            if case.diagnostics is None
            else [
                {"frame": frame, **row} for frame, row in case.diagnostics.rows.items()
            ]
        )
        return {
            "frames": case.frames,
            "observed": {
                camera: None
                if points is None
                else points["observed"].astype(int).tolist()
                for camera, points in case.points.items()
            },
            "evidence": evidence,
        }

    def image(self, case_id: int, camera: str, frame: int) -> bytes:
        # Check snapshot even on a cache hit.
        case = self.case(case_id)
        if camera not in case.camera_ids or not 0 <= frame < case.frames:
            raise IndexError("Unknown camera/frame")
        return self._image(case_id, camera, frame)

    def _image(self, case_id: int, camera: str, frame: int) -> bytes:
        case = self.cases[case_id]
        video = case.videos[case.camera_ids.index(camera)]
        path = Path(video["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Source RGB is unavailable: {camera}")
        with self.decode_lock:
            key = (case_id, camera, frame)
            if key in self.image_cache:
                return self.image_cache[key]
            identity = (path, video["sha256"])
            if identity not in self.verified_videos:
                if dual_sha256(path) != video["sha256"]:
                    raise ValueError(f"Source video checksum mismatch: {camera}")
                self.verified_videos.add(identity)
            capture = cv2.VideoCapture(
                str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            try:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame):
                    raise OSError(f"Cannot seek {camera} frame {frame}")
                okay, image = capture.read()
                if not okay or image.shape[:2] != (video["height"], video["width"]):
                    raise OSError(
                        f"Cannot decode declared source frame: {camera}/{frame}"
                    )
                okay, encoded = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                if not okay:
                    raise OSError("Cannot encode preview")
                result = encoded.tobytes()
                if len(self.image_cache) >= 18:
                    self.image_cache.pop(next(iter(self.image_cache)))
                self.image_cache[key] = result
                return result
            finally:
                capture.release()
