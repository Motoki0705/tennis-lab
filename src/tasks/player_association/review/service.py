"""Bounded read-only dataset review, preserving missing/ambiguous states."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

import cv2
import numpy as np

from src.tasks.player_association.evaluation.dataset_labels import (
    discover_labels,
    label_path,
)
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.review.diagnostics import SavedDiagnostics
from src.tasks.player_association.review.reader import (
    ReviewClip,
    identity_spans,
    load_review_clip,
    within,
)
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.utils.checksum import dual_sha256
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
)


def label_state(clip: ReviewClip, person_index: int) -> dict[str, Any]:
    if person_index == -2:
        return {
            "label_state": "unmatched",
            "person_id": None,
            "role": None,
            "description": "ラベル未対応",
        }
    if person_index == -1:
        return {
            "label_state": "ambiguous",
            "person_id": None,
            "role": None,
            "description": "曖昧box: 採点対象外",
        }
    person = clip.labels.people[person_index]
    return {
        "label_state": person.role,
        "person_id": person.person_id,
        "role": person.role,
        "description": person.description,
    }


class AssociationReviewService:
    def __init__(
        self,
        dataset: Path,
        artifact_root: Path,
        *,
        sides: Path | None = None,
        score_reports: tuple[Path, ...] = (),
    ) -> None:
        self.dataset, self.artifact_root = dataset.resolve(), artifact_root.resolve()
        self.sides = sides
        self.manifest = load_dataset_manifest(self.dataset)
        labelled = [
            ClipLabels.load(within(self.dataset, path))
            for path in discover_labels(self.dataset)
        ]
        self.labels = {labels.clip_id: labels for labels in labelled}
        if not set(self.labels).issubset(self.manifest.clips):
            raise ValueError("Labels reference clips outside the dataset manifest")
        self.diagnostics = SavedDiagnostics(score_reports, self.artifact_root)
        self._clips: OrderedDict[str, ReviewClip] = OrderedDict()
        self._images: OrderedDict[tuple[str, str, int], np.ndarray] = OrderedDict()
        self._lock = RLock()

    def catalog(self) -> dict[str, Any]:
        return {
            "dataset": self.manifest.dataset_id,
            "dataset_root": str(self.dataset),
            "source_clips": len(self.manifest.clips),
            "labelled_clips": len(self.labels),
            "unlabelled_clips": len(self.manifest.clips) - len(self.labels),
            "label_boxes": sum(
                len(camera.frames)
                for labels in self.labels.values()
                for camera in labels.cameras.values()
            ),
            "clips": [
                {
                    "id": key,
                    "frames": self.labels[key].num_frames,
                    "boxes": sum(
                        len(x.frames) for x in self.labels[key].cameras.values()
                    ),
                    "reference": "追加部分参照 / Codex目視・第2注釈者なし"
                    if self.labels[key].provenance.get("blind_to_association")
                    else "旧dev参照 / 設計・評価に使用済み",
                }
                for key in sorted(self.labels)
            ],
            "score_reports": self.diagnostics.catalog(),
            "court": {
                "half_width": HALF_DOUBLES_WIDTH,
                "half_length": HALF_LENGTH,
                "singles_half_width": HALF_SINGLES_WIDTH,
                "service_y": SERVICE_LINE_DISTANCE,
            },
            "current_method": "main: classical CLIP-ReID + box足元幾何 + MILP / 採用尺度A (run14)",
            "research": "learned view associationは研究・歴史的artifact。旧multi_object入力は現物なし。mainの現在の推論器ではない。",
            "label_scope": "既検出boxだけの部分参照。未検出人物を含むrecallは保証しない。匿名IDはclip内のみ。non_playerは除外対象でcamera間同一性は未確認。",
        }

    def clip(self, clip_id: str) -> ReviewClip:
        with self._lock:
            return self._clip(clip_id)

    def _clip(self, clip_id: str) -> ReviewClip:
        if clip_id not in self.labels:
            raise KeyError(f"Unknown labelled clip: {clip_id}")
        if clip_id not in self._clips:
            record = self.manifest.clips[clip_id]
            manifest = ClipManifest.load(
                within(self.dataset, self.dataset / record.path)
            )
            self._clips[clip_id] = load_review_clip(
                manifest, self.labels[clip_id], self.artifact_root, self.sides
            )
            if len(self._clips) > 2:
                self._clips.popitem(last=False)
        self._clips.move_to_end(clip_id)
        return self._clips[clip_id]

    def detail(self, clip_id: str) -> dict[str, Any]:
        clip = self.clip(clip_id)
        timelines, coverage = [], []
        for camera in clip.manifest.camera_ids:
            track = clip.tracks.get(camera)
            match = clip.matched[camera]
            total = len(clip.labels.cameras[camera].frames)
            observed = 0 if track is None else int(track.observed.sum())
            paired = int((match >= -1).sum())
            coverage.append(
                {
                    "camera": camera,
                    "raw_available": track is not None,
                    "raw_observed_boxes": observed,
                    "label_boxes": total,
                    "matched_boxes": paired,
                    "ambiguous_matches": int((match == -1).sum()),
                    "unmatched_raw_boxes": observed - paired
                    if track is not None
                    else None,
                    "unmatched_label_boxes": total - paired
                    if track is not None
                    else None,
                    "label_coverage": paired / total
                    if track is not None and total
                    else None,
                }
            )
            if track is None:
                continue
            for row, track_id in enumerate(track.ids):
                spans = [
                    {**span, **label_state(clip, span["person_index"])}
                    for span in identity_spans(match[row], track.observed[row])
                ]
                known = [
                    (int(f), int(match[row, f]))
                    for f in np.flatnonzero(track.observed[row] & (match[row] >= 0))
                ]
                events = [
                    f
                    for (_, previous), (f, current) in zip(
                        known, known[1:], strict=False
                    )
                    if previous != current
                ]
                timelines.append(
                    {
                        "camera": camera,
                        "track_id": int(track_id),
                        "source_track_ids": list(track.source_ids[row]),
                        "observed_frames": int(track.observed[row].sum()),
                        "spans": spans,
                        "label_transitions": events,
                    }
                )
        return {
            "clip": clip_id,
            "frames": clip.manifest.num_frames,
            "fps": clip.manifest.fps,
            "width": clip.manifest.width,
            "height": clip.manifest.height,
            "camera_ids": list(clip.manifest.camera_ids),
            "people": [
                {"id": x.person_id, "role": x.role, "description": x.description}
                for x in clip.labels.people
            ],
            "coverage": coverage,
            "timelines": timelines,
            "geometry_reason": clip.geometry_reason,
            "provenance": {
                **clip.provenance,
                "clip_manifest": str(clip.manifest.clip_dir / "clip.json"),
                "clip_manifest_sha256": dual_sha256(
                    clip.manifest.clip_dir / "clip.json"
                ),
                "labels_sha256": dual_sha256(label_path(self.dataset, clip_id)),
            },
        }

    def frame(
        self,
        clip_id: str,
        frame: int,
        *,
        report_id: int | None = None,
        camera: str | None = None,
        track_id: int | None = None,
    ) -> dict[str, Any]:
        clip = self.clip(clip_id)
        if not 0 <= frame < clip.manifest.num_frames:
            raise ValueError("Frame outside the clip timeline")
        if (camera is None) != (track_id is None):
            raise ValueError("A selected raw ID requires both camera and track")
        if camera is not None and (
            camera not in clip.tracks or track_id not in clip.tracks[camera].ids
        ):
            raise ValueError("Unknown selected camera/raw track ID")
        observations = []
        for camera_id in clip.manifest.camera_ids:
            track = clip.tracks.get(camera_id)
            if track is None:
                rows = clip.labels.cameras[camera_id].at(frame)
                for person_index, box in zip(
                    clip.labels.cameras[camera_id].person_index[rows],
                    clip.labels.cameras[camera_id].boxes_xyxy[rows],
                    strict=True,
                ):
                    observations.append(
                        {
                            "camera": camera_id,
                            "track_id": None,
                            "box": box.tolist(),
                            "footpoint": None,
                            "footpoint_reason": "raw観測なし: ラベルboxのみ",
                            **label_state(clip, int(person_index)),
                        }
                    )
                continue
            for row in np.flatnonzero(track.observed[:, frame]):
                box = track.boxes[row, frame]
                valid = camera_id in clip.valid and bool(
                    clip.valid[camera_id][row, frame]
                )
                reason = None
                if not valid:
                    reason = clip.geometry_reason or "camera校正なし"
                    if camera_id in clip.cameras:
                        reason = (
                            "box下端が画像境界: 足が切れている"
                            if box[3] > clip.manifest.height - 4
                            else "光線が前方のz=0面と交差しない"
                        )
                observations.append(
                    {
                        "camera": camera_id,
                        "track_id": int(track.ids[row]),
                        "source_track_ids": list(track.source_ids[row]),
                        "box": box.tolist(),
                        "footpoint": clip.points[camera_id][row, frame].tolist()
                        if valid
                        else None,
                        "footpoint_reason": reason,
                        **label_state(clip, int(clip.matched[camera_id][row, frame])),
                    }
                )
        scores = (
            {"available": False, "reason": "保存score report未指定"}
            if report_id is None
            else self.diagnostics.pairs_at(clip, report_id, frame, camera, track_id)
        )
        return {
            "clip": clip_id,
            "frame": frame,
            "seconds": frame / clip.manifest.fps,
            "time_basis": "同期clip manifestのframe / fps",
            "observations": observations,
            "scores": scores,
        }

    def image(
        self, clip_id: str, camera: str, frame: int, track_id: int | None = None
    ) -> bytes:
        with self._lock:
            return self._image(clip_id, camera, frame, track_id)

    def _image(
        self, clip_id: str, camera: str, frame: int, track_id: int | None
    ) -> bytes:
        clip = self.clip(clip_id)
        if (
            camera not in clip.manifest.camera_ids
            or not 0 <= frame < clip.manifest.num_frames
        ):
            raise ValueError("Invalid camera/frame")
        key = (clip_id, camera, frame)
        if key not in self._images:
            path = within(self.dataset, clip.manifest.media_path(camera))
            capture = cv2.VideoCapture(str(path))
            try:
                seek_ok = capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            finally:
                capture.release()
            if (
                not seek_ok
                or not ok
                or abs(position - (frame + 1)) > 1e-3
                or image.shape[:2] != (clip.manifest.height, clip.manifest.width)
            ):
                raise ValueError(
                    "Source RGB frame is unavailable or differs from the manifest"
                )
            self._images[key] = image
            if len(self._images) > 6:
                self._images.popitem(last=False)
        self._images.move_to_end(key)
        image = self._images[key]
        if track_id is not None:
            tracks = clip.tracks.get(camera)
            rows = (
                []
                if tracks is None
                else np.flatnonzero(tracks.ids == track_id).tolist()
            )
            if len(rows) != 1 or tracks is None or not tracks.observed[rows[0], frame]:
                raise ValueError("Selected raw track has no observed box at this frame")
            x1, y1, x2, y2 = tracks.boxes[rows[0], frame]
            left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
            right, bottom = (
                min(image.shape[1], int(np.ceil(x2))),
                min(image.shape[0], int(np.ceil(y2))),
            )
            if right <= left or bottom <= top:
                raise ValueError("Observed crop lies outside RGB")
            image = image[top:bottom, left:right]
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError("Could not encode source RGB")
        return encoded.tobytes()
