"""Atomic top-level run manifest for the current scene state."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from .stages import BY_STAGE, ORDER, Stage, Target, descendants
from .workspace import SceneWorkspace


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


class SceneRunState:
    """Mutable state machine; filesystem paths are relative to one workspace."""

    def __init__(self, workspace: SceneWorkspace, payload: dict[str, Any]):
        self.workspace = workspace
        self.path = workspace.path("run.json")
        self.payload = payload

    @classmethod
    def create_or_load(cls, workspace: SceneWorkspace) -> Self:
        path = workspace.path("run.json")
        if path.exists():
            loaded_payload = json.loads(path.read_text())
            if loaded_payload.get("scene_id") != workspace.scene_id:
                raise ValueError(
                    f"Workspace belongs to {loaded_payload.get('scene_id')!r}, "
                    f"not {workspace.scene_id!r}"
                )
            return cls(workspace, loaded_payload)
        payload: dict[str, Any] = {
            "schema": "tennis_scene_pipeline_run_v1",
            "scene_id": workspace.scene_id,
            "status": "pending",
            "source_video": None,
            "resolved_config": "resolved-config.yaml",
            "seed": None,
            "requested_from_stage": None,
            "nht_from_stage": None,
            "targets": [],
            "created_at_utc": _now(),
            "updated_at_utc": _now(),
            "stages": {
                stage.value: {
                    "status": "pending",
                    "attempts": 0,
                    "output": str(BY_STAGE[stage].owned_path),
                    "external_run": (
                        "reconstruction/run.json"
                        if stage is Stage.RECONSTRUCTION
                        else None
                    ),
                    "summary": None,
                    "error": None,
                    "started_at_utc": None,
                    "finished_at_utc": None,
                }
                for stage in ORDER
            },
        }
        state = cls(workspace, payload)
        state.save()
        return state

    def save(self) -> None:
        self.payload["updated_at_utc"] = _now()
        _atomic_json(self.path, self.payload)

    def request(
        self,
        from_stage: Stage,
        targets: tuple[Target, ...],
        seed: int,
        nht_from_stage: str,
    ) -> None:
        self.payload.update(
            {
                "status": "running",
                "seed": seed,
                "requested_from_stage": from_stage.value,
                "nht_from_stage": nht_from_stage,
                "targets": [target.value for target in targets],
            }
        )
        for stage in descendants(from_stage, include_self=True):
            record = self.payload["stages"][stage.value]
            record.update(
                {
                    "status": "invalidated",
                    "summary": None,
                    "error": None,
                    "started_at_utc": None,
                    "finished_at_utc": None,
                }
            )
        self.save()

    def recover_interrupted(self) -> tuple[Stage, ...]:
        recovered = []
        for stage in ORDER:
            record = self.payload["stages"][stage.value]
            if record["status"] == "running":
                record.update(
                    {
                        "status": "failed",
                        "error": {
                            "category": "process_interrupted",
                            "message": "Previous process ended while stage was running",
                        },
                        "finished_at_utc": _now(),
                    }
                )
                recovered.append(stage)
        if recovered:
            self.payload["status"] = "failed"
            self.save()
        return tuple(recovered)

    def running(self, stage: Stage) -> None:
        record = self.payload["stages"][stage.value]
        record.update(
            {
                "status": "running",
                "attempts": int(record["attempts"]) + 1,
                "summary": None,
                "error": None,
                "started_at_utc": _now(),
                "finished_at_utc": None,
            }
        )
        self.payload["status"] = "running"
        self.save()

    def completed(self, stage: Stage, summary: dict[str, Any]) -> None:
        record = self.payload["stages"][stage.value]
        record.update(
            {
                "status": "completed",
                "summary": summary,
                "error": None,
                "finished_at_utc": _now(),
            }
        )
        self.save()

    def failed(self, stage: Stage, error: BaseException) -> None:
        message = str(error)
        category = "process_signal" if "signal" in message.lower() else "stage_failure"
        self.payload["stages"][stage.value].update(
            {
                "status": "failed",
                "error": {"category": category, "message": message},
                "finished_at_utc": _now(),
            }
        )
        self.payload["status"] = "failed"
        self.save()

    def finish(self, executed: tuple[Stage, ...]) -> None:
        if all(
            self.payload["stages"][stage.value]["status"] == "completed"
            for stage in executed
        ):
            self.payload["status"] = "completed"
        self.save()
