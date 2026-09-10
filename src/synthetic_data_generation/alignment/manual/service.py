"""Revision-checked draft storage and reversible canonical owner publication."""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from src.synthetic_data_generation.alignment.heatmaps import validate_line_heatmaps
from src.synthetic_data_generation.alignment.manual.artifacts import (
    CONFIRMATION_FILE,
    load_manual_source,
    read_json,
    validate_manual_outputs,
    write_manual_outputs,
)
from src.synthetic_data_generation.alignment.manual.geometry import (
    build_manual_result,
    court_segments,
)
from src.synthetic_data_generation.alignment.manual.models import (
    ApplyRequest,
    EditRequest,
    LayoutEdit,
)
from src.synthetic_data_generation.alignment.manual.source import (
    file_digest,
    import_source,
    owner_digest,
    reconstruction_identity,
)
from src.synthetic_data_generation.pipeline.contracts import StageName, StageStatus
from src.synthetic_data_generation.pipeline.locking import scene_write_lock
from src.synthetic_data_generation.pipeline.publication import (
    exchange_owner_directories,
)
from src.synthetic_data_generation.pipeline.run_manifest import MutableRunManifest
from src.utils.io import save_json_atomic, utc_now_iso

DEPENDENT_OWNERS = {
    StageName.COURT_DATASET: Path("datasets/court"),
    StageName.BLCS_DATASET: Path("datasets/blcs"),
    StageName.PLCS_DATASET: Path("datasets/plcs"),
    StageName.REPORT: Path("report"),
}


class AlignmentEditor:
    """An editor is explicitly bound to one scene, with no client-supplied paths."""

    def __init__(self, scene_root: Path, *, recover_ground_frame: bool = False) -> None:
        self.root = scene_root.resolve(strict=True)
        self.owner = self.root / "alignment"
        self.editor_root = self.root / "alignment-editor"
        self.token = secrets.token_urlsafe(32)
        self.mutex = RLock()
        self.recover_ground_frame = recover_ground_frame
        self._reload()

    def _reload(self) -> None:
        """Refresh the source after an explicit browser reload, under the writer lock."""
        with scene_write_lock(self.root):
            if (self.owner / CONFIRMATION_FILE).exists():
                validate_manual_outputs(self.owner)
                self.source = load_manual_source(self.owner)
                self.heatmaps = validate_line_heatmaps(self.owner / "line-heatmaps")
                self.layout = LayoutEdit.model_validate(
                    read_json(self.owner / CONFIRMATION_FILE)["layout"]
                )
            else:
                self.source, self.heatmaps = import_source(
                    self.root, recover_ground_frame=self.recover_ground_frame
                )
                self.layout = self.source.initial_edit()
            if self.source.reconstruction != reconstruction_identity(self.root):
                raise ValueError(
                    "Reconstruction has changed since the source alignment was imported."
                )
            self.revision = self.current_revision()
        self.camera_export = self.root / "reconstruction" / "export"
        camera_document = read_json(self.camera_export / "cameras.json")
        self.cameras = camera_document["cameras"]

    def current_revision(self) -> str:
        payload = {
            "alignment": owner_digest(self.owner),
            "reconstruction": reconstruction_identity(self.root),
            "run": file_digest(self.root / "run.json"),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def check_revision(self, revision: str) -> None:
        if revision != self.revision or self.current_revision() != self.revision:
            raise RuntimeError(
                "Scene changed since it was opened. Reload the editor before saving or applying."
            )

    def state(self) -> dict[str, Any]:
        with self.mutex:
            if self.current_revision() != self.revision:
                self._reload()
            draft_path = self.editor_root / "draft.json"
            draft = read_json(draft_path) if draft_path.exists() else None
            if draft is not None:
                if (
                    set(draft) != {"schema", "revision", "layout", "saved_at"}
                    or draft["schema"] != "manual_alignment_draft_v1"
                ):
                    raise ValueError("Invalid saved editor draft.")
                LayoutEdit.model_validate(draft["layout"])
            current_draft = (
                draft if draft and draft["revision"] == self.revision else None
            )
            return {
                "scene_id": self.root.name,
                "revision": self.revision,
                "token": self.token,
                "layout": current_draft["layout"]
                if current_draft
                else self.layout.model_dump(),
                "initial_layout": self.source.initial_edit().model_dump(),
                "has_draft": current_draft is not None,
                "stale_draft": bool(draft and current_draft is None),
                "bounds": self.heatmaps.bounds_uv,
                "grid_spacing": self.heatmaps.grid_spacing,
                "raster_shape": self.heatmaps.raster_shape,
                "segments": court_segments(),
                "provenance": self.source.provenance,
                "applied_manually": (self.owner / CONFIRMATION_FILE).exists(),
                "cameras": [
                    {
                        "index": i,
                        "camera_id": c["camera_id"],
                        "width": c["width"],
                        "height": c["height"],
                    }
                    for i, c in enumerate(self.cameras)
                ],
            }

    def save_draft(self, request: EditRequest) -> dict[str, str]:
        with self.mutex, scene_write_lock(self.root):
            self.check_revision(request.revision)
            self.editor_root.mkdir(exist_ok=True)
            saved_at = utc_now_iso()
            save_json_atomic(
                {
                    "schema": "manual_alignment_draft_v1",
                    "revision": self.revision,
                    "layout": request.layout.model_dump(),
                    "saved_at": saved_at,
                },
                self.editor_root / "draft.json",
            )
            return {"saved_at": saved_at}

    def preview(self, edit: LayoutEdit, camera_index: int) -> dict[str, Any]:
        import numpy as np

        from src.synthetic_data_generation.alignment.manual.geometry import (
            court_segments,
        )

        result = build_manual_result(self.source, self.heatmaps, edit)
        camera = self.cameras[camera_index]
        intrinsic = np.asarray(camera["intrinsics"]["matrix"])
        camera_from_nht = np.linalg.inv(np.asarray(camera["camera_to_scene"]))
        segments = np.asarray(court_segments())
        local = np.column_stack((segments.reshape(-1, 2), np.zeros(segments.size // 2)))
        projected = []
        for court in result.layout.courts:
            nht = result.metric_adapter.nht_from_metric_points(
                court.scene_from_court.apply(local)
            )
            points = nht @ camera_from_nht[:3, :3].T + camera_from_nht[:3, 3]
            lines = []
            for pair in points.reshape(-1, 2, 3):
                # Clip at positive camera depth before perspective division.
                pair = pair.copy()
                if np.all(pair[:, 2] <= 1e-5):
                    continue
                if np.any(pair[:, 2] <= 1e-5):
                    behind = int(np.argmin(pair[:, 2]))
                    front = 1 - behind
                    fraction = (1e-5 - pair[behind, 2]) / (
                        pair[front, 2] - pair[behind, 2]
                    )
                    pair[behind] += fraction * (pair[front] - pair[behind])
                pixels = pair @ intrinsic.T
                lines.append((pixels[:, :2] / pixels[:, 2:]).tolist())
            projected.append({"court_id": court.court_instance_id, "segments": lines})
        return {
            "courts": projected,
            "diagnostics": [
                {
                    "court_id": c.court_instance_id,
                    "fit": c.fit.to_dict(),
                    "holdout": c.holdout.to_dict(),
                }
                for c in result.candidates
            ],
        }

    def apply(self, request: ApplyRequest) -> dict[str, Any]:
        if not request.human_confirmed:
            raise ValueError("Explicit human confirmation is required.")
        with self.mutex, scene_write_lock(self.root):
            self.check_revision(request.revision)
            manifest = MutableRunManifest.load(self.root / "run.json")
            if manifest.scene_id != self.root.name or any(
                record.status is StageStatus.RUNNING
                for record in manifest.stages.values()
            ):
                raise RuntimeError(
                    "Scene manifest is mismatched or a stage is running. Resolve it before applying."
                )
            if (
                manifest.stages[StageName.RECONSTRUCTION].status
                is not StageStatus.COMPLETED
            ):
                raise RuntimeError(
                    "Reconstruction must be completed before applying alignment."
                )
            if not request.layout.courts:
                raise ValueError("Add at least one court before applying.")
            history = self.editor_root / "history" / uuid4().hex
            snapshot = history / "alignment"
            snapshot.mkdir(parents=True)
            moved: list[tuple[Path, Path]] = []
            exchanged = False
            mutated = False
            original_manifest = (self.root / "run.json").read_bytes()
            try:
                result = write_manual_outputs(
                    snapshot,
                    source_owner=self.owner,
                    source=self.source,
                    edit=request.layout,
                    confirmed_at=utc_now_iso(),
                    revision=self.revision,
                )
                (history / "run.json").write_bytes(original_manifest)
                # All descendants lose their completed claims before geometry changes.
                for stage in (*DEPENDENT_OWNERS, StageName.ALIGNMENT):
                    manifest.invalidate(stage)
                manifest.begin(StageName.ALIGNMENT)
                manifest.save(self.root / "run.json")
                mutated = True
                for relative in (*DEPENDENT_OWNERS.values(), Path("publication")):
                    origin = self.root / relative
                    if origin.exists():
                        if origin.is_symlink() or not origin.is_dir():
                            raise ValueError(
                                f"Dependent owner must be an ordinary directory: {origin}"
                            )
                        destination = history / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        origin.rename(destination)
                        moved.append((origin, destination))
                exchange_owner_directories(snapshot, self.owner)
                exchanged = True
                manifest.complete(
                    StageName.ALIGNMENT,
                    {
                        "authority": "human_confirmed",
                        "accepted_court_count": len(result.layout.courts),
                        "evaluated_court_count": len(result.candidates),
                        "fit_camera_count": len(result.partitions.fit_camera_ids),
                        "holdout_camera_count": len(
                            result.partitions.holdout_camera_ids
                        ),
                        "primary_court_instance_id": result.layout.primary_court_instance_id,
                    },
                )
                manifest.save(self.root / "run.json")
            except Exception:
                if exchanged:
                    exchange_owner_directories(snapshot, self.owner)
                for origin, destination in reversed(moved):
                    destination.rename(origin)
                if mutated:
                    rollback_path = self.root / "run.json.rollback"
                    rollback_path.write_bytes(original_manifest)
                    rollback_path.replace(self.root / "run.json")
                shutil.rmtree(history)
                raise
            self.layout = request.layout
            self.revision = self.current_revision()
            (self.editor_root / "draft.json").unlink(missing_ok=True)
            return {
                "revision": self.revision,
                "court_count": len(result.layout.courts),
                "history": str(history),
                "message": "手動確定した配置を適用しました。下流データは再生成待ちです。",
            }
