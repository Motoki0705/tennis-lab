"""Archive reviewed candidates and publish accepted JSON with recoverable decisions."""

from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from ..artifacts.store import ArtifactStore
from ..runtime.contracts import BallAnnotation
from ..runtime.validation import validate_annotation
from .campaign_state import locked_state, log_event, processed_path, read_state
from .common import atomic_write_json, load_annotation, load_manifest, utc_now
from .configuration import file_sha256, json_object, paths
from .path_contracts import campaign_resolver, validate_command_paths

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.chat_annotation.local_agent",
    fields=(BoundaryPathField("campaign", PathRole.OUTPUT, PathDirection.INPUT, PathKind.DIRECTORY,
                              must_exist=True, allow_role_root=True),),
)



def final_attempt(task: dict[str, Any]) -> dict[str, Any]:
    for record in reversed(task["attempts"]):
        if record.get("kind") == "done":
            return dict(record)
    raise ValueError("task has no finished attempt")


@contextmanager
def publication_lock() -> Iterator[None]:
    paths().annotated.mkdir(parents=True, exist_ok=True)
    with (paths().annotated / ".raw-processing.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def json_only_zip(member: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, payload)
    return buffer.getvalue()


def publish_new(path: Path, payload: bytes) -> None:
    """Atomic create-only publication, also used for immutable history copies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def replace_bytes(path: Path, payload: bytes) -> None:
    """Publish one fully validated file atomically."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def candidate(
    task_id: str, task: dict[str, Any]
) -> tuple[Path, bytes, str, dict[str, Any]]:
    record = final_attempt(task)
    directory = Path(record["dir"])
    source = directory / f"annotation_{task['clip_id']}.json"
    result = json_object(directory / "result.json")
    digest = file_sha256(source)
    if (
        result["task_id"] != task_id
        or result["attempt"] != record["n"]
        or result["annotation_sha256"] != digest
        or result["clip_id"] != task["clip_id"]
        or result["target"] != task["target"]
    ):
        raise ValueError("finished result does not match this task/attempt/annotation")
    annotation = load_annotation(source)
    if not isinstance(annotation, BallAnnotation):
        raise ValueError("ball campaigns cannot adopt another target schema")
    report = validate_annotation(annotation, load_manifest(Path(task["manifest"])))
    if report.errors or report.reviewed_frames != report.target_frames:
        raise ValueError(
            f"candidate is not adoptable: {report.errors[:3]}, reviewed={report.reviewed_frames}/{report.target_frames}"
        )
    if (
        result["outcome"] not in ("completed", "partial")
        or annotation.status != result["outcome"]
    ):
        raise ValueError("candidate outcome/status is not a finished annotation")
    validation = {
        "validation_status": report.status,
        "reviewed_frames": report.reviewed_frames,
        "target_frames": report.target_frames,
    }
    return source, source.read_bytes(), digest, validation


def processing_path(artifact_id: str) -> Path:
    return paths().annotated / "processing" / artifact_id.replace(".zip", ".json")


def adopt(task_id: str, note: str) -> dict[str, Any]:
    if not note.strip():
        raise ValueError("adoption requires a recorded QA note")
    with publication_lock(), locked_state() as state:
        task = state["tasks"][task_id]
        if task["status"] not in ("review", "held", "adopted"):
            raise ValueError(f"cannot adopt from {task['status']}")
        source, payload, digest, validation = candidate(task_id, task)
        saved = ArtifactStore(paths().annotated / "raw").save(
            f"local_{task['clip_id']}.zip",
            json_only_zip(f"annotation_{task['clip_id']}.json", payload),
        )
        artifact_id = saved["artifact_id"]
        output = processed_path(task["target"], task["clip_id"])
        record_path = processing_path(artifact_id)
        processing = json_object(record_path) if record_path.exists() else None
        if processing is not None:
            prior = processing["members"][0]
            if prior["clip_id"] != task["clip_id"]:
                raise ValueError("archive record belongs to a different clip")
            if prior["decision"] == "accepted" and (
                not output.exists() or file_sha256(output) != prior["output_sha256"]
            ):
                raise ValueError(
                    "published annotation differs from its completed processing record"
                )
        if task["status"] == "adopted":
            if (
                not processing
                or processing["members"][0]["decision"] not in ("accepted", "duplicate")
                or file_sha256(output) != digest
            ):
                raise ValueError("adopted task publication is inconsistent")
            return {
                "task": task_id,
                "decision": task["adoption"]["decision"],
                "artifact_id": artifact_id,
            }
        if not output.exists():
            publish_new(output, payload)
            decision = "accepted"
        elif file_sha256(output) == digest:
            decision = (
                "accepted"
                if processing and processing["members"][0]["decision"] == "accepted"
                else "duplicate"
            )
        else:
            decision = "held"
        if processing is None:
            member = {
                "member": f"annotation_{task['clip_id']}.json",
                "clip_id": task["clip_id"],
                "target": task["target"],
                "decision": decision,
                "output": str(output.relative_to(paths().annotation_root))
                if decision != "held"
                else None,
                "output_sha256": digest if decision != "held" else None,
                "manifest": str(
                    Path(task["manifest"]).relative_to(paths().annotation_root)
                ),
                "manifest_sha256": file_sha256(Path(task["manifest"])),
                **validation,
                "reason": note,
            }
            processing = {
                "artifact_id": artifact_id,
                "updated_at": utc_now(),
                "state": "completed",
                "origin": "repository local_agent workflow; no external MCP transfer",
                "campaign_task": task_id,
                "campaign_attempt": final_attempt(task)["n"],
                "campaign_result": str(source.parent / "result.json"),
                "members": [member],
                "next_action": "review replacement against the existing annotation"
                if decision == "held"
                else "none",
            }
            atomic_write_json(record_path, processing)
        task["status"] = "held" if decision == "held" else "adopted"
        task["adoption"] = {
            "at": utc_now(),
            "artifact_id": artifact_id,
            "decision": decision,
            **validation,
            "note": note,
        }
    log_event("ADOPT", f"{task_id} decision={decision} artifact={artifact_id[:12]}")
    return {"task": task_id, "decision": decision, "artifact_id": artifact_id}


def replace(
    task_id: str, comparison_path: Path, note: str, presence_reviewed: bool
) -> dict[str, Any]:
    """Require a pinned comparison and a visual decision, including one-frame disagreements."""
    if not note.strip():
        raise ValueError("replacement requires a recorded visual QA note")
    comparison = json_object(comparison_path)
    if comparison["task_id"] != task_id:
        raise ValueError("comparison belongs to another task")
    with publication_lock(), locked_state() as state:
        task = state["tasks"][task_id]
        if task.get("phase") != 2 or task["status"] not in ("held", "adopted"):
            raise ValueError(
                "replacement is only available for archived phase-2 candidates"
            )
        _, payload, digest, _ = candidate(task_id, task)
        output = processed_path(task["target"], task["clip_id"])
        record_path = processing_path(task["adoption"]["artifact_id"])
        processing = json_object(record_path)
        member = processing["members"][0]
        if comparison["new_sha256"] != digest or comparison[
            "manifest_sha256"
        ] != file_sha256(Path(task["manifest"])):
            raise ValueError("candidate or manifest changed after comparison")
        old_digest = comparison["old_sha256"]
        current_digest = file_sha256(output)
        backup = (
            paths().annotated
            / "history"
            / task["target"]
            / task["clip_id"]
            / f"{old_digest}.json"
        )
        pending = processing.get("replacement_pending")
        completed = member.get("replaced")
        recovering = bool(
            (
                pending
                and pending.get("old_sha256") == old_digest
                and pending.get("new_sha256") == digest
            )
            or (
                completed
                and completed.get("sha256") == old_digest
                and member.get("output_sha256") == digest
            )
        )
        if current_digest != old_digest and not (
            recovering and current_digest == digest
        ):
            raise ValueError(
                "processed annotation changed after comparison; repeat the comparison and review"
            )
        old_source = backup if current_digest == digest else output
        if file_sha256(old_source) != old_digest:
            raise ValueError("replacement history is inconsistent")
        from .phase2 import compare_pair

        actual = compare_pair(
            load_annotation(old_source),
            load_annotation(
                Path(final_attempt(task)["dir"]) / f"annotation_{task['clip_id']}.json"
            ),
            load_manifest(Path(task["manifest"])),
        )
        old, new, distance = actual["old"], actual["new"], actual["centre_distance_px"]
        if new["unreviewed"] != 0 or new["unresolved_share"] > old["unresolved_share"]:
            raise ValueError(
                "replacement fails the unreviewed/unresolved ratio criteria"
            )
        if distance["median"] is None or distance["median"] > 2:
            raise ValueError(
                "replacement requires a defined centre-distance median <= 2 pixels"
            )
        if (
            actual["ball_only_in_old"]
            or actual["ball_only_in_new"]
            or actual["count_disagreement_frames"]
        ) and not presence_reviewed:
            raise ValueError(
                "all presence disagreements require explicit visual review, including short runs"
            )
        if not backup.exists():
            publish_new(backup, output.read_bytes())
        elif file_sha256(backup) != old_digest:
            raise ValueError("history backup hash mismatch")
        # Persist the approved intent before publishing; retries can finish either side of the rename.
        processing["replacement_pending"] = {
            "old_sha256": old_digest,
            "new_sha256": digest,
            "backup": str(backup.relative_to(paths().annotation_root)),
            "comparison": actual,
            "presence_reviewed": presence_reviewed,
            "qa_note": note,
        }
        processing.update(
            state="in_progress",
            next_action="finish approved replacement",
            updated_at=utc_now(),
        )
        atomic_write_json(record_path, processing)
        if current_digest != digest:
            replace_bytes(output, payload)
        member.update(
            decision="accepted",
            output=str(output.relative_to(paths().annotation_root)),
            output_sha256=digest,
            reason=note,
            replaced={
                "sha256": old_digest,
                "backup": str(backup.relative_to(paths().annotation_root)),
            },
            replacement_review={
                "comparison": actual,
                "presence_reviewed": presence_reviewed,
                "qa_note": note,
            },
        )
        processing.pop("replacement_pending", None)
        processing.update(state="completed", next_action="none", updated_at=utc_now())
        atomic_write_json(record_path, processing)
        task["status"] = "adopted"
        task["adoption"].update(
            decision="replaced",
            replaced_at=utc_now(),
            old_sha256=old_digest,
            output_sha256=digest,
        )
    log_event("REPLACE", f"{task_id} old={old_digest[:12]} new={digest[:12]}")
    return {
        "task": task_id,
        "decision": "replaced",
        "old_sha256": old_digest,
        "new_sha256": digest,
    }


def keep_old(task_id: str, reason: str) -> dict[str, Any]:
    if not reason.strip():
        raise ValueError("keeping the old annotation requires a reason")
    with publication_lock(), locked_state() as state:
        task = state["tasks"][task_id]
        if task.get("phase") != 2 or task["status"] not in ("held", "kept_old"):
            raise ValueError(
                "only archived phase-2 candidates can keep the old annotation"
            )
        record_path = processing_path(task["adoption"]["artifact_id"])
        processing = json_object(record_path)
        processing["members"][0]["reason"] = reason
        processing.update(state="completed", updated_at=utc_now(), next_action="none")
        atomic_write_json(record_path, processing)
        task["status"] = "kept_old"
        task["adoption"].update(
            decision="kept_old", decided_at=utc_now(), reason=reason
        )
    log_event("KEEP_OLD", f"{task_id}: {reason}")
    return {"task": task_id, "decision": "kept_old"}


def revise(task_id: str, notes: str, unreview: list[str]) -> dict[str, Any]:
    from .ct import parse_frames

    with locked_state() as state:
        task = state["tasks"][task_id]
        if task["status"] not in ("review", "held", "failed", "kept_old"):
            raise ValueError("task must be stopped before revision")
        count = len(load_manifest(Path(task["manifest"])).frames)
        frames = sorted(
            {index for spec in unreview for index in parse_frames(spec, count)}
        )
        task.update(parent_review=notes, unreview_frames=frames, status="continue")
    log_event("REVISE", task_id)
    return {"task": task_id, "decision": "continue", "unreview_frames": frames}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("adopt")
    p.add_argument("task_ids", nargs="+")
    p.add_argument("--note", required=True)
    p = sub.add_parser("replace")
    p.add_argument("task_id")
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--note", required=True)
    p.add_argument("--presence-reviewed", action="store_true")
    p = sub.add_parser("keep-old")
    p.add_argument("task_ids", nargs="+")
    p.add_argument("--reason", required=True)
    p = sub.add_parser("revise")
    p.add_argument("task_id")
    p.add_argument("--notes", required=True)
    p.add_argument("--unreview", nargs="*", default=[])
    p = sub.add_parser("list")
    p.add_argument("--status")
    args = parser.parse_args(argv)
    PATH_BOUNDARY.validate({"campaign": paths().campaign_dir}, resolver=campaign_resolver(paths()))
    validate_command_paths()
    if args.command == 'replace':
        args.comparison = validate_command_paths(comparison=args.comparison)['comparison']
    if args.command == "list":
        state = read_state()
        for task_id, task in state["tasks"].items():
            if args.status is None or task["status"] == args.status:
                print(task_id, task["status"], len(task["attempts"]))
        return 0
    if args.command == "replace":
        result = replace(
            args.task_id, args.comparison, args.note, args.presence_reviewed
        )
    elif args.command == "revise":
        result = revise(args.task_id, args.notes, args.unreview)
    else:
        results = [
            (
                adopt(tid, args.note)
                if args.command == "adopt"
                else keep_old(tid, args.reason)
            )
            for tid in args.task_ids
        ]
        print(json.dumps(results, ensure_ascii=False))
        return 0
    print(json.dumps(result, ensure_ascii=False))
    return 0
