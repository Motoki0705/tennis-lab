"""Single background worker; sync proposals never silently modify the project."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.tennis_scene.clip_studio.audio_sync import estimate_audio_offsets
from src.tennis_scene.clip_studio.export import (
    ClipExportPlan,
    ExportSettings,
    plan_clip_export,
    verify_existing_export,
)
from src.tennis_scene.clip_studio.web.exporting import (
    ExportCancelled,
    export_interruptibly,
)
from src.tennis_scene.clip_studio.web.service import Editor
from src.tennis_scene.configuration import AudioSyncRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import register_exported_clip

LOGGER = logging.getLogger(__name__)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=0)
    kind: Literal["sync", "export"]
    reference: int = Field(default=0, ge=0)
    clip_names: list[str] | None = None


class Jobs:
    """At most one running job; encoding runs in a cancellable child process."""

    def __init__(
        self, editor: Editor, export: ExportSettings, audio: AudioSyncRuntimeConfig
    ) -> None:
        self.editor = editor
        self.export = export
        self.audio = audio
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="clip-studio-job"
        )
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._state: dict[str, Any] = {"status": "idle"}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def start(self, request: JobRequest) -> dict[str, Any]:
        with self._lock, self.editor.lock:
            if self._state["status"] == "running":
                raise ValueError("処理中です。完了またはキャンセルを待ってください。")
            self.editor.check_revision(request.revision)
            project = deepcopy(self.editor.project)
            if request.reference >= len(project.sources):
                raise ValueError("基準カメラが存在しません。")
            self._cancel.clear()
            self._state = {
                "kind": request.kind,
                "status": "running",
                "revision": request.revision,
                "completed": 0,
                "skipped": [],
                "message": "準備中",
            }

            def run() -> None:
                try:
                    if request.kind == "sync":
                        self._update(
                            message="音声を解析しています（キャンセル時は結果を破棄します）"
                        )
                        result = estimate_audio_offsets(
                            [source.path for source in project.sources],
                            reference_index=request.reference,
                            reference_offset_sec=project.sources[
                                request.reference
                            ].offset_sec,
                            sample_rate=self.audio.sample_rate,
                            envelope_rate=self.audio.envelope_rate,
                            max_seconds=self.audio.max_seconds,
                        )
                        if not self._cancel.is_set():
                            self._update(
                                offsets_sec=result.offsets_sec,
                                confidences=result.confidences,
                            )
                    else:
                        names = request.clip_names
                        clips = (
                            project.clips
                            if names is None
                            else [
                                project.clips[project.clip_index_by_name(n)]
                                for n in names
                            ]
                        )
                        if not clips or len({clip.name for clip in clips}) != len(
                            clips
                        ):
                            raise ValueError(
                                "書き出すクリップを重複なく選択してください。"
                            )
                        # Validate the entire batch before the first file is written.
                        plans = [
                            plan_clip_export(
                                project, self.editor.infos, clip, self.export
                            )
                            for clip in clips
                        ]
                        pending = []
                        skipped = []
                        for plan in plans:
                            if self._cancel.is_set():
                                raise ExportCancelled()
                            if not self.export.overwrite and verify_existing_export(
                                plan, self.export
                            ):
                                skipped.append(plan.clip_name)
                            else:
                                pending.append(plan)
                        # Existing media were verified against the current edit. Repair
                        # a missing index record only after the whole batch passes.
                        for name in skipped:
                            register_exported_clip(
                                self.export.output_dir,
                                self.export.output_dir
                                / "clips"
                                / project.recording_id
                                / name
                                / "clip.json",
                            )
                        self._update(
                            total=len(pending),
                            skipped=skipped,
                            frames_completed=0,
                            frames_total=0,
                        )
                        for index, plan in enumerate(pending):
                            if self._cancel.is_set():
                                raise ExportCancelled()
                            self._update(
                                message=f"{plan.clip_name} を準備しています ({index + 1}/{len(pending)})",
                                frames_completed=0,
                                frames_total=plan.num_frames * len(plan.cameras),
                            )

                            def progress(
                                camera: str,
                                completed: int,
                                total: int,
                                plan: ClipExportPlan = plan,
                                index: int = index,
                            ) -> None:
                                camera_index = next(
                                    i
                                    for i, item in enumerate(plan.cameras)
                                    if item.camera_id == camera
                                )
                                self._update(
                                    message=f"{plan.clip_name} / {camera}: {completed}/{total} フレーム ({index + 1}/{len(pending)} クリップ)",
                                    frames_completed=camera_index * total + completed,
                                )

                            export_interruptibly(
                                plan, self.export, self._cancel, progress
                            )
                            self._update(completed=index + 1)
                    self._update(
                        status="cancelled" if self._cancel.is_set() else "done"
                    )
                except ExportCancelled:
                    self._update(
                        status="cancelled",
                        message="書き出しを停止し、途中ファイルを削除しました。",
                    )
                except Exception as error:
                    LOGGER.exception("Clip studio job failed")
                    self._update(status="failed", message=str(error))

            self._executor.submit(run)
            return self.snapshot()

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._state["status"] == "running":
                self._cancel.set()
                self._state["message"] = (
                    "キャンセル要求済み。エンコード停止と途中ファイルの削除を行っています。"
                    if self._state["kind"] == "export"
                    else "キャンセル要求済み。音声解析の終了後に結果を破棄します。"
                )
            return self.snapshot()

    def close(self) -> None:
        self._cancel.set()
        self._executor.shutdown(wait=True)
