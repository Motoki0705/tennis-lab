"""Saved historical classical pair evidence, never a model/solver invocation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.tasks.player_association.review.reader import ReviewClip, within
from src.utils.checksum import dual_sha256


class SavedDiagnostics:
    def __init__(self, paths: tuple[Path, ...], artifact_root: Path) -> None:
        self.reports: list[dict[str, Any]] = []
        for path in paths:
            path = within(artifact_root, path)
            report = json.loads(path.read_text())
            if report.get("schema") != "player_association_evaluate_v1":
                raise ValueError(f"Unsupported saved score report: {path}")
            self.reports.append(
                {
                    "id": len(self.reports),
                    "name": path.parent.name,
                    "path": str(path),
                    "sha256": dual_sha256(path),
                    "data": report,
                }
            )

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {key: report[key] for key in ("id", "name", "path", "sha256")}
            for report in self.reports
        ]

    def for_clip(self, clip: ReviewClip, report_id: int) -> dict[str, Any]:
        if not 0 <= report_id < len(self.reports):
            raise ValueError("Unknown saved score report")
        report = self.reports[report_id]
        data = report["data"]
        store = clip.provenance.get("store")
        if store is None or Path(data["observe"]).resolve() != Path(store).parents[2]:
            return {
                "available": False,
                "reason": "保存scoreの観測runがこのraw storeと異なるため結合しない",
            }
        if any(track.version != 3 for track in clip.tracks.values()):
            return {
                "available": False,
                "reason": "旧evaluate v1はperson_tracks v3入力。異なる世代のrawへscoreを結合しない",
            }
        side_report = clip.provenance.get("side_report")
        if side_report and Path(data["sides"]).resolve() != Path(side_report).resolve():
            return {
                "available": False,
                "reason": "保存scoreと表示足元のside reportが異なるため結合しない",
            }
        record = data["clips"].get(clip.manifest.clip_id)
        if record is None:
            return {"available": False, "reason": "このclipの保存scoreなし"}
        diagnostics = record.get("diagnostics", {})
        if (
            diagnostics.get("frames") != clip.manifest.num_frames
            or abs(diagnostics.get("fps", 0) - clip.manifest.fps) > 1e-5
        ):
            raise ValueError("Saved score time grid differs from the raw input")
        candidates = diagnostics.get("candidates", [])
        segments = diagnostics.get("segments", [])
        items = []
        for segment_id in candidates:
            if type(segment_id) is not int or not 0 <= segment_id < len(segments):
                raise ValueError("Saved candidate references an unknown segment")
            segment = segments[segment_id]
            track = clip.tracks.get(segment["camera"])
            if (
                track is None
                or segment["track_id"] not in track.ids
                or not 0
                <= segment["start"]
                < segment["end"]
                <= clip.manifest.num_frames
            ):
                raise ValueError(
                    "Saved score segment disagrees with raw tracks/timeline"
                )
            row = track.ids.tolist().index(segment["track_id"])
            if (
                int(track.observed[row, segment["start"] : segment["end"]].sum())
                != segment["observed_frames"]
            ):
                raise ValueError(
                    "Saved score observed-frame count differs from the raw input"
                )
            items.append(segment)
        for pair in diagnostics.get("pairs", []):
            if any(
                type(pair[key]) is not int or not 0 <= pair[key] < len(items)
                for key in ("a", "b")
            ):
                raise ValueError("Saved pair references an unknown candidate")
        return {
            "available": True,
            "status": record["status"],
            "reason": record.get("reason"),
            "method": "historical classical / geometry-only"
            if data["geometry_only"]
            else "historical classical / CLIP + geometry",
            "report": {key: report[key] for key in ("id", "name", "path", "sha256")},
            "config_path": data["config_path"],
            "metrics": record.get("metrics"),
            "binding": "同run・v3・camera/raw ID・区間/実観測数・時間格子を照合。旧reportにbox内容hashは保存されていない。",
            "items": items,
            "diagnostics": diagnostics,
        }

    def pairs_at(
        self,
        clip: ReviewClip,
        report_id: int,
        frame: int,
        camera: str | None,
        track_id: int | None,
    ) -> dict[str, Any]:
        result = self.for_clip(clip, report_id)
        if not result["available"]:
            return result
        diagnostics, items = result.pop("diagnostics"), result.pop("items")
        pairs = []
        for pair in diagnostics.get("pairs", []):
            a, b = items[pair["a"]], items[pair["b"]]
            if a["camera"] == b["camera"]:
                continue
            if camera is not None and track_id is not None:

                def selected(item: dict[str, Any]) -> bool:
                    return bool(
                        item["camera"] == camera
                        and item["track_id"] == track_id
                        and item["start"] <= frame < item["end"]
                    )

                if selected(b):
                    a, b = b, a
                elif not selected(a):
                    continue
            pairs.append(
                {
                    **pair,
                    "from": a,
                    "to": b,
                    "active_now": a["start"] <= frame < a["end"]
                    and b["start"] <= frame < b["end"],
                }
            )
        result["pairs"] = pairs
        result["scope"] = (
            "scoreは保存segment全区間の証拠。現在frameで再推論していない。"
        )
        return result
