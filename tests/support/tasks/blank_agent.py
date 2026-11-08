"""Real pinned local inputs and controlled worker attempts for blank-tool tests."""

from __future__ import annotations

import base64
import ctypes
import json
import os
import select
import time
from pathlib import Path
from typing import Any

from src.tasks.annotation.configuration import configure_local_videos
from src.tasks.annotation.local_videos import LocalVideoConfigurationReceipt
from src.tasks.ball_detection.annotation.video.agent.contracts import (
    AgentDecision,
    AgentTask,
    AgentView,
    EditQuery,
    ServiceEvent,
)
from src.tasks.ball_detection.annotation.video.agent.services import execute_service
from src.tasks.ball_detection.annotation.video.agent.store import VideoRevisionStore
from src.tasks.ball_detection.annotation.video.agent.worker import BallVideoWorker
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan
from src.tasks.base.annotation.telemetry.capture import capture
from src.tasks.base.annotation.workers.contracts import ExitReceipt, Launch, WorkRequest
from src.tasks.base.annotation.workers.ledger import WorkerLedger
from src.tasks.base.annotation.workers.pool import WorkerPool
from src.tasks.base.annotation.workers.services import (
    ServiceReply,
    ServiceRequest,
    publish_reply,
)
from tests.support.tasks.local_annotation import local_configuration


def drive(
    layout: AnnotationLayout,
    plan: LocalAnnotationPlan,
    pool: WorkerPool,
    request: WorkRequest,
    directory: Path,
    *,
    saved_only: bool = False,
) -> dict:
    # Wait on real inbox/result notifications, never a guessed completion sleep.
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "inotify unavailable")
    inbox = directory / "work/service-requests"
    inbox.mkdir(exist_ok=True)
    for path in (directory, directory / "work", inbox):
        if (
            libc.inotify_add_watch(descriptor, os.fsencode(path), 0x8 | 0x80 | 0x100)
            < 0
        ):
            os.close(descriptor)
            raise OSError(ctypes.get_errno(), "inotify watch failed")
    try:
        return _drive(
            layout, plan, pool, request, directory, descriptor, saved_only=saved_only
        )
    finally:
        os.close(descriptor)


def _drive(
    layout: AnnotationLayout,
    plan: LocalAnnotationPlan,
    pool: WorkerPool,
    request: WorkRequest,
    directory: Path,
    descriptor: int,
    *,
    saved_only: bool,
) -> dict[str, Any]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = pool.reconcile(request)
        if state["state"] not in {"running", "exited"}:
            return state
        for path in sorted((directory / "work/service-requests").glob("*.json")):
            reference = annotation_reference(layout, path)
            service = read_document(layout, reference, ServiceRequest)
            if (directory / "services" / service.sha256 / "reply.json").exists():
                continue
            result = execute_service(layout, plan, reference)
            publish_reply(
                layout,
                reference,
                ServiceReply(
                    request_digest=service.sha256,
                    status="succeeded",
                    result=result,
                    reason=None,
                ),
            )
        if saved_only and (directory / "work/saved.json").exists():
            return state
        if select.select([descriptor], [], [], max(0, deadline - time.monotonic()))[0]:
            os.read(descriptor, 65536)
    raise AssertionError(f"synthetic worker did not settle: {state}")


def synthetic_telemetry(
    layout, worker, request, launch, launch_ref, exit_ref, *, missing=False
):
    """Use the actual fake-worker service envelopes with explicitly synthetic usage."""
    directory = Path(launch.directory)
    execution = read_document(layout, exit_ref, ExitReceipt)

    def record(kind, **payload):
        return {"type": kind, "timestamp": "2026-11-09T00:00:00Z", "payload": payload}

    records = [
        record(
            "session_meta",
            id=execution.thread_id,
            cwd=str(directory / "work"),
            cli_version="synthetic",
            model_provider="openai",
        ),
        record("turn_context", model="gpt-6.1-sol"),
    ]
    envelopes = [
        json.loads(line)["envelope"]
        for line in (directory / "events.jsonl").read_text().splitlines()
        if json.loads(line).get("type") == "tool.synthetic"
    ]
    for index, envelope in enumerate(envelopes, 1):
        event = read_document(
            layout,
            annotation_reference(
                layout,
                directory / "services" / envelope["video_service"] / "event.json",
            ),
            ServiceEvent,
        )
        call_id = "service-" + str(index)
        records.extend(
            [
                record(
                    "response_item",
                    type="custom_tool_call",
                    call_id=call_id,
                    name="exec",
                    input="tools.py " + event.operation,
                ),
                record(
                    "response_item",
                    type="custom_tool_call_output",
                    call_id=call_id,
                    output=json.dumps(envelope),
                ),
            ]
        )
        if event.operation == "video_image":
            view = envelope["view"]
            image_id = "image-" + str(index)
            records.extend(
                [
                    record(
                        "response_item",
                        type="custom_tool_call",
                        call_id=image_id,
                        name="view_image",
                        input=json.dumps({"path": view["image_path"]}),
                    ),
                    record(
                        "event_msg",
                        type="item_completed",
                        call_id=image_id,
                        item={"type": "ImageView", "path": view["image_path"]},
                    ),
                    record(
                        "response_item",
                        type="custom_tool_call_output",
                        call_id=image_id,
                        output=[
                            {
                                "type": "input_image",
                                "detail": "original",
                                "image_url": "data:image/jpeg;base64,"
                                + base64.b64encode(
                                    Path(view["image_path"]).read_bytes()
                                ).decode(),
                            }
                        ],
                    ),
                ]
            )
        usage = (
            None
            if missing and index == 2
            else dict(
                input_tokens=100,
                cached_input_tokens=None if missing and index == 3 else 40,
                output_tokens=10,
                total_tokens=110,
            )
        )
        usage_record = record(
            "token_usage_record", response_id="response-" + str(index), usage=usage
        )
        records.append(usage_record)
        if index == 1:
            records.append(usage_record)
    rollout = directory / "synthetic-rollout.jsonl"
    rollout.write_text("".join(json.dumps(value) + "\n" for value in records))
    plan = read_document(layout, launch.plan, LocalAnnotationPlan)
    return capture(
        layout,
        launch_ref,
        exit_ref,
        worker.telemetry_images(layout, plan, request, directory),
        rollout=rollout,
        classifier=worker.classify_tool,
        classifier_id="ball_video_annotation.3",
        evidence_resolver=lambda category, text: worker.tool_evidence(
            layout, launch_ref, category, text
        ),
    )


class BlankSession:
    def __init__(self, root: Path) -> None:
        self.layout = AnnotationLayout.testing(root)
        self.configuration = local_configuration(self.layout)
        receipt = configure_local_videos(self.layout, self.configuration)
        self.receipt = read_document(
            self.layout, receipt, LocalVideoConfigurationReceipt
        )
        self.plan = read_document(self.layout, self.receipt.plan, LocalAnnotationPlan)
        self.request = read_document(
            self.layout, self.receipt.registrations[0].request, WorkRequest
        )
        self.worker = BallVideoWorker()
        self.prepare(1)

    def prepare(self, number: int) -> None:
        self.directory = self.layout.path(
            "campaigns",
            self.plan.campaign_id,
            "workers",
            self.request.work_id,
            f"attempt-{number:04d}",
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        prepared = self.worker.prepare(
            self.layout, self.plan, self.request, self.directory, timeout_seconds=10
        )
        self.task_ref = annotation_reference(
            self.layout, self.directory / "request.json"
        )
        task = read_document(self.layout, self.task_ref, AgentTask)
        self.store = VideoRevisionStore(self.layout, task.run)
        self.launch = Launch(
            work_id=self.request.work_id,
            attempt=f"{self.request.work_id}-{number:04d}",
            plan=self.receipt.plan,
            request=self.request,
            prepared=prepared,
            directory=str(self.directory),
            code_root=str(self.layout.repository),
            codex_binary="/usr/bin/true",
            codex_home=str(self.directory / "work/.codex"),
            timeout_seconds=10,
            test_command=("/usr/bin/true",),
            test_root=str(self.layout.root),
        )
        self.actor = write_document(
            self.layout, self.directory / "launch.json", self.launch
        )
        with WorkerLedger(self.layout).registry.transaction() as db:
            db.execute(
                "INSERT INTO worker_attempts VALUES (?,?,?,'running',?,NULL)",
                (
                    self.launch.attempt,
                    self.request.work_id,
                    number,
                    self.actor.model_dump_json(),
                ),
            )
            db.execute(
                "UPDATE worker_tasks SET current_attempt=?,state='running' WHERE id=?",
                (self.launch.attempt, self.request.work_id),
            )

    @property
    def camera(self) -> str:
        camera: str = self.store.expected.source.camera_id
        return camera

    def service_ref(self, operation: str, parameters: dict) -> ArtifactRef:
        service = ServiceRequest(
            task_request=self.task_ref, operation=operation, parameters=parameters
        )
        return write_document(
            self.layout,
            self.directory / "work/service-requests" / (service.sha256 + ".json"),
            service,
        )

    def call(self, operation: str, parameters: dict) -> AgentView:
        result = execute_service(
            self.layout, self.plan, self.service_ref(operation, parameters)
        )
        view: AgentView = read_document(self.layout, result, AgentView)
        return view

    def edit(self, rows: tuple) -> ArtifactRef:
        query = EditQuery(
            camera_id=self.camera,
            expected_revision=self.store.current()[0],
            rows=rows,
            reason="synthetic decision",
        )
        return self.call("video_edit", query.model_dump(mode="json")).revision

    def execution(self) -> ExitReceipt:
        return ExitReceipt(
            launch_digest=self.actor.digest,
            exit_code=0,
            status="succeeded",
            started_at="2026-11-09T00:00:00+00:00",
            ended_at="2026-11-09T00:00:01+00:00",
            elapsed_seconds=1,
            events=write_asset(self.layout, self.directory / "events.jsonl", b""),
            errors=write_asset(self.layout, self.directory / "stderr.log", b""),
            thread_id="00000000-0000-0000-0000-000000000001",
            reason=None,
        )

    def decide(self) -> ArtifactRef:
        return write_document(
            self.layout,
            self.directory / "result.json",
            AgentDecision(
                status="complete",
                revision=self.store.current()[0],
                reason="complete synthetic annotations",
            ),
        )


def fake_worker(
    *, pause_after_save: bool = False, probes: tuple[str, ...] = ()
) -> tuple[str, ...]:
    """Only the model is replaced; service inboxes, kernel sandbox and saves are real."""
    source = r"""
import json, subprocess, signal
from pathlib import Path
fixed = json.loads(Path('../input/tool-input.json').read_text())
camera = fixed['input']['source']['camera_id']
print(json.dumps({'type':'thread.started','thread_id':'00000000-0000-0000-0000-000000000001'}), flush=True)
for path in PROBES:
    try:
        Path(path).read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError('host file readable: ' + path)
    child = subprocess.run(['/usr/bin/cat', path], capture_output=True)
    assert child.returncode != 0 and b'Permission denied' in child.stderr
    link = Path('escape')
    link.unlink(missing_ok=True)
    link.symlink_to(path)
    try:
        link.read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError('symlink escape')
Path('boundary-checked.json').write_text(json.dumps(PROBES))
counter = 0
def service(operation, parameters):
    global counter
    counter += 1
    query = Path('query-' + str(counter) + '.json')
    query.write_text(json.dumps(parameters))
    reply = subprocess.run(['/usr/bin/python3','-I','-S','../input/tools.py',operation,'--parameters',str(query)],capture_output=True,text=True)
    assert reply.returncode == 0, reply.stderr
    envelope = json.loads(reply.stdout)
    print(json.dumps({'type':'tool.synthetic','envelope':envelope}), flush=True)
    return envelope['view']
handoff = service('video_handoff', {'camera_id':camera})
rows = fixed['input']['frames']
selected = rows[0]['frame_index']
if handoff['remaining']:
    selected = handoff['remaining'][-1]
    if fixed['policy']['policy_id'] == 'chronological':
        selected = handoff['remaining'][0]
width,height = fixed['input']['source']['width'],fixed['input']['source']['height']
query = {'camera_id':camera,'tiles':[{'frame_index':selected,'crop':{'left':0,'top':0,'right':width,'bottom':height}}], 'columns':1,'tile_width':width,'tile_height':max(32,height),'reason':'adaptive anchor'}
image = service('video_image', query)
query['reason'] = 'revisit the same camera frame'
image = service('video_image', query)
assert Path(image['image_path']).read_bytes()[:2] == b'\xff\xd8'
remaining = handoff['remaining'][:1] if PAUSE else handoff['remaining']
updates = []
for row in rows:
    if row['frame_index'] not in remaining:
        continue
    # Final unknown is intentional; subsequent attempts must not repeat it.
    state = 'unknown' if row['frame_index'] == rows[0]['frame_index'] else 'in_frame'
    updates.append(dict(row, state=state, occlusion='unknown' if state=='unknown' else 'not_occluded', position=None if state=='unknown' else {'x':12.0,'y':9.0,'provenance':{'kind':'estimated','support_frames':[selected]}}))
if updates:
    handoff = service('video_edit', {'camera_id':camera,'expected_revision':handoff['revision'],'rows':updates,'reason':'inference from same-camera anchor'})
Path('saved.json').write_text(json.dumps(handoff))
if PAUSE:
    signal.pause()
Path('result.json').write_text(json.dumps({'status':'complete','revision':handoff['revision'],'reason':'all target frames processed'}))
print(json.dumps({'type':'turn.completed'}), flush=True)
"""
    source = source.replace("PROBES", repr(probes)).replace(
        "PAUSE", repr(pause_after_save)
    )
    return ("/usr/bin/python3", "-I", "-S", "-c", source)
