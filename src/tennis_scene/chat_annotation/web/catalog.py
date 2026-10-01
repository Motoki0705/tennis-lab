"""Manifest-based inventory with explicit accepted/draft/submission provenance."""

from __future__ import annotations

import hashlib
import threading
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..layout import done_video_path, video_path
from ..runtime.contracts import (
    ClipManifest,
    SupportedAnnotation,
    annotation_clip_id,
    loads_json,
    parse_annotation,
)
from .metrics import inspect_annotation, missing_summary

TARGETS = ("ball", "player")
MAX_JSON_BYTES = 64 * 1024 * 1024


class RevisionConflict(ValueError):
    pass


@dataclass
class Version:
    id: str
    clip_id: str
    target: str
    path: Path
    origin: str
    generation: int | None = None
    expected_sha: str | None = None
    member: str | None = None
    parent_verified: bool = False
    provenance_errors: list[str] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    sha256: str | None = None
    file_signature: tuple[int, int] | None = None


@dataclass
class Clip:
    id: str
    manifest_path: Path
    manifest: ClipManifest
    manifest_sha: str
    video: Path | None
    location: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    versions: dict[str, list[Version]] = field(
        default_factory=lambda: {target: [] for target in TARGETS}
    )
    active: bool = False
    video_signature: tuple[int, int] | None = None


class Catalog:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("root must be a directory")
        self.lock = threading.RLock()
        self.clips: dict[str, Clip] = {}
        self.versions: dict[str, Version] = {}
        self.diagnostics: list[str] = []
        self.campaign: dict[str, Any] = {}
        self.revision = ""
        self.loaded_at = 0.0
        self._fingerprints: list[str] = []
        self.refresh()

    def safe(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("出力root外の参照は使用できません")
        return resolved

    def relative(self, path: Path) -> str:
        return str(self.safe(path).relative_to(self.root))

    def _track(self, path: Path) -> None:
        path = self.safe(path)
        if path.is_file():
            stat = path.stat()
            self._fingerprints.append(
                f"{self.relative(path)}:{stat.st_size}:{stat.st_mtime_ns}"
            )

    def _json(self, path: Path) -> Any:
        path = self.safe(path)
        self._track(path)
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("JSON size limit exceeded")
        return loads_json(path.read_bytes())

    def _version(
        self, clip_id: str, target: str, path: Path, origin: str, **kwargs: Any
    ) -> None:
        if clip_id not in self.clips or target not in TARGETS:
            self.diagnostics.append(f"manifestに対応しない注釈: {clip_id} / {target}")
            return
        try:
            path = self.safe(path)
            self._track(path)
        except (OSError, ValueError) as error:
            self.clips[clip_id].warnings.append(f"注釈参照エラー: {error}")
            return
        key = f"{clip_id}:{self.relative(path)}:{kwargs.get('member', '')}:{target}"
        version_id = hashlib.sha256(key.encode()).hexdigest()[:24]
        if version_id in self.versions:
            return
        version = Version(version_id, clip_id, target, path, origin, **kwargs)
        if path.is_file():
            stat = path.stat()
            version.file_signature = (stat.st_size, stat.st_mtime_ns)
        self.versions[version_id] = version
        self.clips[clip_id].versions[target].append(version)

    def refresh(self) -> None:
        with self.lock:
            self.clips = {}
            self.versions = {}
            self.diagnostics = []
            self._fingerprints = []
            self.campaign = {}
            for path in sorted(
                (self.root / "_preparation").glob("*/*/clips/*/clip_manifest.json")
            ):
                try:
                    raw = self._json(path)
                    manifest = ClipManifest.model_validate(raw)
                    clip_id = annotation_clip_id(manifest)
                    if clip_id in self.clips:
                        self.clips[clip_id].errors.append(
                            "同一clip IDのmanifestが複数あります"
                        )
                        continue
                    candidates = [
                        self.safe(video_path(self.root, manifest)),
                        self.safe(done_video_path(self.root, manifest)),
                    ]
                    existing = [p for p in candidates if p.is_file()]
                    clip = Clip(
                        clip_id,
                        path,
                        manifest,
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                        existing[0] if len(existing) == 1 else None,
                        "done"
                        if existing and existing[0] == candidates[1]
                        else "videos",
                    )
                    if len(existing) != 1:
                        clip.errors.append(
                            "動画が見つかりません"
                            if not existing
                            else "videosとdoneに動画が重複しています"
                        )
                    elif existing[0].stat().st_size != manifest.bytes:
                        clip.errors.append("動画のサイズがmanifestと一致しません")
                    if len(existing) == 1:
                        stat = existing[0].stat()
                        clip.video_signature = (stat.st_size, stat.st_mtime_ns)
                    for video in existing:
                        self._track(video)
                    self.clips[clip_id] = clip
                except (OSError, ValueError, TypeError, KeyError) as error:
                    self.diagnostics.append(f"manifest {path.name}: {error}")
            for target in TARGETS:
                for path in sorted(
                    (self.root / "annotated" / "processed" / target).glob("*.json")
                ):
                    self._version(
                        path.stem, target, path, "accepted", parent_verified=True
                    )
            self._campaign_versions()
            self._submission_versions()
            # Only default views are eagerly inspected. Historical versions are lazy.
            for clip in self.clips.values():
                for target in TARGETS:
                    self.default_version(clip, target, "accepted")
                    self.default_version(clip, target, "working")
            self.revision = hashlib.sha256(
                "\n".join(sorted(set(self._fingerprints))).encode()
            ).hexdigest()[:20]
            self.loaded_at = time.monotonic()

    def _campaign_versions(self) -> None:
        base = self.root / "codex_campaign"
        try:
            state = (
                self._json(base / "state.json")
                if (base / "state.json").is_file()
                else {}
            )
            if not isinstance(state, dict):
                raise ValueError("campaign state must be an object")
            self.campaign = {
                key: state.get(key)
                for key in ("status", "updated_at", "ended_at", "allow_worker_refill")
            }
            workers = state.get("active_workers", {})
            if not isinstance(workers, dict):
                raise ValueError("active_workers must be an object")
            for worker in workers.values():
                if not isinstance(worker, dict):
                    raise ValueError("worker entry must be an object")
                for item in self._json(Path(worker["assignment"])):
                    if item.get("clip_id") in self.clips:
                        self.clips[item["clip_id"]].active = True
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.diagnostics.append(f"campaign state: {error}")
        for directory in sorted(base.glob("generation_*")):
            try:
                generation = int(directory.name.removeprefix("generation_"))
                result_path = directory / "result.json"
                export_path = directory / "exported.json"
                if not result_path.is_file() and not export_path.is_file():
                    continue
                result = self._json(result_path) if result_path.is_file() else {}
                if not isinstance(result, dict):
                    raise ValueError("result must be an object")
                exports = (
                    self._json(export_path)
                    if export_path.is_file()
                    else result.get("artifacts", [])
                )
                if not exports:
                    exports = [
                        dict(a, target=t, clip_id=c["clip_id"])
                        for c in result.get("clips", [])
                        for t, a in c["targets"].items()
                    ]
                assignments = {
                    a["clip_id"]: a for a in self._json(directory / "assignment.json")
                }
                parent_path = directory / "parent_result.json"
                parent = self._json(parent_path) if parent_path.is_file() else {}
                if not isinstance(parent, dict) or not isinstance(
                    parent.get("validation", {}), dict
                ):
                    raise ValueError("invalid parent verification record")
                verified = parent.get("validation", {}).get("errors") == 0
                for item in exports:
                    if not isinstance(item, dict):
                        raise ValueError("export entries must be objects")
                    clip_id, target = item["clip_id"], item["target"]
                    if clip_id not in self.clips:
                        continue
                    assignment = assignments.get(clip_id, {})
                    errors = []
                    if (
                        assignment.get("manifest_sha256")
                        != self.clips[clip_id].manifest_sha
                    ):
                        errors.append("下書きのmanifestハッシュが一致しません")
                    if not item.get("annotation_sha256"):
                        errors.append("下書きに注釈ハッシュがありません")
                    self._version(
                        clip_id,
                        target,
                        Path(item["annotation"]),
                        "draft",
                        generation=generation,
                        expected_sha=item.get("annotation_sha256"),
                        parent_verified=verified,
                        provenance_errors=errors,
                    )
            except (OSError, ValueError, TypeError, KeyError) as error:
                self.diagnostics.append(f"{directory.name}: {error}")

    def _submission_versions(self) -> None:
        raw_dir = self.root / "annotated" / "raw"
        records: dict[str, dict[str, Any]] = {}
        for path in sorted((self.root / "annotated" / "processing").glob("*.json")):
            try:
                record = self._json(path)
                if not isinstance(record, dict):
                    raise ValueError("processing record must be an object")
                records[path.stem] = record
                for member in record.get("members", []):
                    if not isinstance(member, dict):
                        raise ValueError("processing members must be objects")
                    clip_id, target = member.get("clip_id"), member.get("target")
                    if clip_id not in self.clips or target not in TARGETS:
                        continue
                    if member.get("decision") in {
                        "accepted",
                        "duplicate",
                    } and member.get("output_sha256"):
                        for version in self.clips[clip_id].versions[target]:
                            if version.origin == "accepted":
                                expected = member["output_sha256"]
                                if (
                                    version.expected_sha
                                    and version.expected_sha != expected
                                ):
                                    version.provenance_errors.append(
                                        "採用記録に異なる出力ハッシュが複数あります"
                                    )
                                version.expected_sha = expected
                    if (
                        member.get("decision") != "held"
                        or member.get("disposition") == "superseded_campaign_draft"
                    ):
                        continue
                    self.clips[clip_id].warnings.append(
                        f"{target}: 未採用の提出物あり ({path.stem[:8]})"
                    )
                    self._version(
                        clip_id,
                        target,
                        raw_dir / f"{path.stem}.zip",
                        "raw",
                        member=member.get("member"),
                        expected_sha=member.get("output_sha256"),
                    )
            except (OSError, ValueError, TypeError, KeyError) as error:
                self.diagnostics.append(f"processing {path.name}: {error}")
        for path in sorted(raw_dir.glob("*.zip")):
            try:
                self._track(path)
                if records.get(path.stem, {}).get("state") == "completed":
                    continue
                if path.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("ZIP size limit exceeded")
                with zipfile.ZipFile(self.safe(path)) as archive:
                    members = archive.infolist()
                    if not 1 <= len(members) <= 256 or len(
                        {m.filename for m in members}
                    ) != len(members):
                        raise ValueError("ZIP has too many or duplicate members")
                    if sum(m.file_size for m in archive.infolist()) > MAX_JSON_BYTES:
                        raise ValueError("expanded ZIP size limit exceeded")
                    for member in archive.infolist():
                        if member.flag_bits & 1 or member.compress_type not in {
                            zipfile.ZIP_STORED,
                            zipfile.ZIP_DEFLATED,
                        }:
                            raise ValueError("encrypted or unsupported ZIP member")
                        data = loads_json(archive.read(member))
                        if not isinstance(data, dict):
                            raise ValueError("submitted JSON must be an object")
                        schema = data.get("schema_version")
                        if not isinstance(schema, str):
                            raise ValueError("submitted JSON has no schema version")
                        target = {
                            "tennis_chat_ball_annotation.v1": "ball",
                            "tennis_chat_player_annotation.v1": "player",
                        }.get(schema)
                        clip_id = data.get("clip_id")
                        if clip_id in self.clips and target:
                            self.clips[clip_id].warnings.append(
                                f"{target}: 受領済み・未整理 ({path.stem[:8]})"
                            )
                            self._version(
                                clip_id, target, path, "raw", member=member.filename
                            )
                        else:
                            self.diagnostics.append(
                                f"raw {path.name}: manifestに対応しないclip/target: {clip_id}"
                            )
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                zipfile.BadZipFile,
            ) as error:
                self.diagnostics.append(f"raw {path.name}: {error}")

    def read_version(
        self, version: Version
    ) -> tuple[SupportedAnnotation, dict[str, Any], list[dict[str, Any]]]:
        path = self.safe(version.path)
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != version.file_signature:
            raise RevisionConflict("注釈ファイルが更新されています。再集計してください")
        if version.member is None:
            if path.stat().st_size > MAX_JSON_BYTES:
                raise ValueError("JSON size limit exceeded")
            content = path.read_bytes()
        else:
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("ZIP size limit exceeded")
            if hashlib.sha256(path.read_bytes()).hexdigest() != path.stem:
                raise ValueError("提出ZIPのハッシュが一致しません")
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo(version.member)
                if info.file_size > MAX_JSON_BYTES:
                    raise ValueError("expanded JSON size limit exceeded")
                if info.flag_bits & 1 or info.compress_type not in {
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                }:
                    raise ValueError("encrypted or unsupported ZIP member")
                content = archive.read(version.member)
        digest = hashlib.sha256(content).hexdigest()
        if version.expected_sha and digest != version.expected_sha:
            raise ValueError("注釈ハッシュが出典記録と一致しません")
        if version.sha256 and digest != version.sha256:
            raise RevisionConflict("注釈が更新されています。一覧を再読込してください")
        annotation = parse_annotation(loads_json(content))
        summary, timeline = inspect_annotation(
            annotation, self.clips[version.clip_id].manifest, version.target
        )
        if version.provenance_errors:
            summary["errors"].extend(version.provenance_errors)
            summary["state"] = "invalid"
        version.sha256 = digest
        return annotation, summary, timeline

    def inspect(self, version: Version) -> dict[str, Any]:
        if version.summary is None:
            try:
                _, summary, _ = self.read_version(version)
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                zipfile.BadZipFile,
            ) as error:
                summary = missing_summary(
                    len(self.clips[version.clip_id].manifest.frames)
                )
                summary.update(state="invalid", errors=[str(error)])
            version.summary = summary
        return version.summary

    def default_version(self, clip: Clip, target: str, view: str) -> Version | None:
        versions = clip.versions[target]
        accepted = next((v for v in versions if v.origin == "accepted"), None)
        if accepted:
            self.inspect(accepted)
            return accepted  # Invalid accepted data must never silently fall back.
        if view == "working":
            drafts = sorted(
                (v for v in versions if v.origin == "draft" and v.parent_verified),
                key=lambda v: v.generation or 0,
                reverse=True,
            )
            if drafts:
                self.inspect(drafts[0])
                return drafts[0]  # A corrupt latest draft is exposed, not skipped.
        return None

    def version_payload(
        self, version: Version, *, statistics: bool = True
    ) -> dict[str, Any]:
        summary = self.inspect(version) if statistics else None
        return {
            "id": version.id,
            "origin": version.origin,
            "generation": version.generation,
            "path": self.relative(version.path),
            "member": version.member,
            "parent_verified": version.parent_verified,
            "label": "採用済み"
            if version.origin == "accepted"
            else f"下書き g{version.generation:03d}"
            if version.origin == "draft"
            else f"未採用ZIP {version.path.stem[:8]}",
            "sha256": version.sha256,
            **({"statistics": summary} if statistics else {}),
        }

    def snapshot(self, view: str = "working") -> dict[str, Any]:
        with self.lock:
            rows: list[dict[str, Any]] = []
            for clip in self.clips.values():
                targets = {}
                for target in TARGETS:
                    selected = self.default_version(clip, target, view)
                    accepted = self.default_version(clip, target, "accepted")
                    stats = (
                        self.inspect(selected)
                        if selected
                        else missing_summary(len(clip.manifest.frames))
                    )
                    targets[target] = {
                        **stats,
                        "selected_version": selected.id if selected else None,
                        "origin": selected.origin if selected else None,
                        "accepted_state": self.inspect(accepted)["state"]
                        if accepted
                        else "missing",
                        "versions": [
                            self.version_payload(v, statistics=False)
                            for v in clip.versions[target]
                        ],
                        "selection_required": selected is None
                        and bool(clip.versions[target]),
                    }
                states = [t["state"] for t in targets.values()]
                needs_selection = any(t["selection_required"] for t in targets.values())
                state = (
                    "invalid"
                    if clip.errors or "invalid" in states
                    else "selection_required"
                    if needs_selection
                    else "missing"
                    if all(v == "missing" for v in states)
                    else "pending"
                    if any(
                        v in {"missing", "unreviewed", "in_progress"} for v in states
                    )
                    else "completed"
                    if all(v == "completed" for v in states)
                    else "reviewed_partial"
                )
                manifest = clip.manifest
                rows.append(
                    {
                        "id": clip.id,
                        "source_id": manifest.source.source_id,
                        "title": manifest.source.title,
                        "source_url": manifest.source.url,
                        "frames": len(manifest.frames),
                        "width": manifest.width,
                        "height": manifest.height,
                        "fps": manifest.nominal_fps,
                        "duration_seconds": float(
                            (
                                manifest.frames[-1].clip_pts
                                + manifest.frames[-1].duration_pts
                            )
                            * Fraction(manifest.time_base)
                        ),
                        "location": clip.location,
                        "video_available": clip.video is not None and not clip.errors,
                        "state": state,
                        "targets": targets,
                        "active": clip.active,
                        "errors": clip.errors,
                        "warnings": sorted(set(clip.warnings)),
                        "delegatable": state in {"missing", "pending"}
                        and not clip.active
                        and not clip.warnings,
                    }
                )
            counts = Counter(row["state"] for row in rows)
            return {
                "revision": self.revision,
                "view": view,
                "root": str(self.root),
                "campaign": self.campaign,
                "diagnostics": self.diagnostics,
                "summary": {
                    "clips": len(rows),
                    "sources": len({row["source_id"] for row in rows}),
                    "states": dict(counts),
                    "reviewed_target_frames": sum(
                        t["reviewed"] for row in rows for t in row["targets"].values()
                    ),
                    "target_frames": sum(row["frames"] * 2 for row in rows),
                    "accepted_annotations": sum(
                        t["accepted_state"] != "missing"
                        for row in rows
                        for t in row["targets"].values()
                    ),
                },
                "clips": rows,
            }

    def check_revision(self, revision: str) -> None:
        if revision != self.revision:
            raise RevisionConflict("一覧が更新されています。再読込してください")

    def check_clip_current(self, clip: Clip) -> None:
        if (
            hashlib.sha256(self.safe(clip.manifest_path).read_bytes()).hexdigest()
            != clip.manifest_sha
        ):
            raise RevisionConflict("manifestが変更されています。再集計してください")
        if clip.video is not None:
            path = self.safe(clip.video)
            if not path.is_file():
                raise RevisionConflict(
                    "動画が移動・削除されています。再集計してください"
                )
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != clip.video_signature:
                raise RevisionConflict("動画が変更されています。再集計してください")

    def select(
        self, clip_id: str, target: str, version_id: str | None
    ) -> Version | None:
        if clip_id not in self.clips or target not in TARGETS:
            raise KeyError("動画または対象が見つかりません")
        if version_id is None:
            return None
        version = self.versions.get(version_id)
        if version is None or version.clip_id != clip_id or version.target != target:
            raise KeyError("注釈の版が動画・対象と一致しません")
        return version
