"""Real HTTP/store/controller; explicit fake editor/rendering verdict boundaries.

Four UI scenarios use a small renderer fake. The managed scenario uses actual
v1/v2 compiler, renderer bytes, visual safety and image-unavailability policy.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Condition, Event, Thread
from types import SimpleNamespace
from typing import Any

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from tests.support.agents_ops.explain_html import (
    PAYLOADS,
    candidate_for,
    document_api,
    make_document,
    schema,
    store_api,
    validation_for,
)
from tests.support.agents_ops.explain_html_visuals import make_visual_bundle, visuals
from tests.support.agents_ops.explain_jobs import jobs, service

web: Any = importlib.import_module("explain_html.web")
managed: Any = importlib.import_module("explain_html.managed")
RENDERER = b"""(() => {
const input = document.querySelector('[data-figure-id="error-plot"] input');
if (input) input.addEventListener('input', () => {
document.querySelector('[data-figure-id="error-plot"] output').textContent = input.value;
});
})();"""
STYLE = b"""body{margin:0;padding:28px;font:18px/1.7 system-ui;color:#202c39;background:#fff}
section{padding:15px 10px;max-width:850px}h1,h2{line-height:1.3}img{width:160px;max-width:100%}
#overview{min-height:900px}#limits{min-height:950px}
input{max-width:100%}h2:focus-visible,figure:focus-visible{outline:3px solid #a94c00}
@media(max-width:500px){body{padding:16px}}"""
HTML = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src HASH; style-src 'self' 'unsafe-inline'; img-src 'self'">
<title>Two views, one position</title></head><body>
<section id="overview" data-content-id="overview"><link rel="stylesheet" href="assets/theme.css"><h1>Two views, one position</h1>
<p>A hypothetical example: two cameras locate a tennis ball.</p><p>Keep reading while an explanation is being revised.</p><a href="#geometry">Read the geometry</a></section>
<section id="geometry" data-content-id="geometry"><h2>How the rays meet</h2><p id="quote">😀 Intersect two rays. Then measure 7 m.</p>
<p data-evidence-id="baseline">Two synchronized cameras; baseline 7 m.</p>
<figure data-figure-id="court-diagram"><img data-asset-id="court" src="assets/court.svg" alt="Two calibrated camera rays meet at the ball"><figcaption>Two camera rays intersect.</figcaption></figure></section>
<section id="limits" data-content-id="limits"><h2>Explore the assumptions</h2><details><summary>Why calibration matters</summary><p>Calibration gives each image point a ray.</p></details>
<figure data-figure-id="error-plot"><figcaption>A hypothetical sensitivity control</figcaption><label>Error (px) <input type="range" min="-10" max="10" value="0"></label><output>0</output></figure>
<p>Adjust the example error while reading the explanation.</p><script data-asset-id="plot_renderer">RENDERER</script></section>
</body></html>"""


def initial_document(document_id: str) -> tuple[Any, dict[str, bytes]]:
    import base64

    payloads = {**PAYLOADS, "theme": STYLE, "plot_renderer": RENDERER}
    document = make_document()
    data = document.model_dump(mode="json")
    data["document_id"] = document_id
    digest = base64.b64encode(hashlib.sha256(RENDERER).digest()).decode("ascii")
    data["html"] = HTML.replace("HASH", f"'sha256-{digest}'").replace(
        "RENDERER", RENDERER.decode()
    )
    if document_id == "reading":
        data["html"] = (
            data["html"]
            .replace("<body>", '<body><article data-content-id="explanation">', 1)
            .replace("</body>", "</article></body>", 1)
            .replace(
                "<details>",
                '<details id="assumptions" data-content-id="assumptions">',
                1,
            )
        )
    for asset in data["assets"]:
        asset["sha256"] = hashlib.sha256(payloads[asset["asset_id"]]).hexdigest()
    return schema.parse(schema.Document, json.dumps(data)), payloads


class BrowserWorld:
    def __init__(self, root: Path):
        self.store = store_api.RevisionStore(root)
        self.initial: dict[str, Any] = {}
        self.calls: list[Any] = []
        self.gates: dict[str, tuple[Event, Event]] = {}
        self.condition = Condition()
        self.finishing = False
        self.hold_undo = False
        self.restore_calls: list[Any] = []
        self.resources = managed.ManagedResources()
        restore = self.store.restore

        def controlled_restore(request: Any) -> Any:
            self.restore_calls.append(request)
            if self.hold_undo:
                self.hold_undo = False
                entered, release = Event(), Event()
                with self.condition:
                    self.gates[request.request_id] = entered, release
                    if self.finishing:
                        release.set()
                    self.condition.notify_all()
                entered.set()
                assert release.wait(30), "Browser did not release the undo publication."
            return restore(request)

        self.store.restore = controlled_restore
        for name in ("reading", "recovery", "security", "removed"):
            document, payloads = initial_document(name)
            candidate = candidate_for(document)
            candidate = candidate.model_copy(
                update={
                    "request": candidate.request.model_copy(
                        update={"document_id": name}
                    ),
                    "changes": document_api.content_changes(None, document),
                }
            )
            self.initial[name] = self.store.import_initial(
                candidate, payloads, validation_for(candidate)
            )
        for version in ("1", "2"):
            name = f"managed-v{version}"
            document, payloads = make_visual_bundle(version)
            # Generated CSS must remain confined to the content iframe. If the
            # UI accidentally inserts parsed nodes, this hides its edit form.
            document = document.model_copy(
                update={
                    "document_id": name,
                    "html": document.html.replace(
                        "p{line-height:1.4}",
                        "p{line-height:1.4}#edit-form{display:none!important}",
                        1,
                    ),
                }
            )
            visuals.validate_visuals(document, payloads)
            candidate = candidate_for(document)
            candidate = candidate.model_copy(
                update={
                    "request": candidate.request.model_copy(
                        update={"document_id": name}
                    )
                }
            )
            self.initial[name] = self.store.import_initial(
                candidate, payloads, validation_for(candidate)
            )
        self.controller = jobs.JobController(
            self.store,
            lambda prepared: self,
            self.validate_bundle,
            prepare=self.resources.prepare,
        )

    @staticmethod
    def validate_bundle(candidate: Any, assets: Any, *, cancel: Event) -> Any:
        if candidate.document.document_id.startswith("managed-"):
            visuals.validate_visuals(candidate.document, dict(assets))
        # Publication's browser verdict is a test fake; the Node browser then
        # exercises the real renderer and CSP in the published reader view.
        return schema.CheckResult(status="passed")

    def edit(self, request: Any, current: Any, context: Any, *, cancel: Event) -> Any:
        entered, release = Event(), Event()
        with self.condition:
            self.calls.append(request)
            self.gates[request.request_id] = entered, release
            if self.finishing:
                release.set()
            self.condition.notify_all()
        entered.set()
        if request.instruction.startswith("hold"):
            assert release.wait(30), "Browser did not release the fake editor."
        if "fail" in request.instruction:
            raise schema.ContractError("generation_failed")
        if "image" in request.instruction:
            if request.document_id.startswith("managed-"):
                image = schema.Figure(
                    figure_id="requested-image",
                    content_id="geometry",
                    kind="image",
                    asset_ids=[],
                    caption="A new generated illustration of the camera rays.",
                    alt="Conceptual camera rays meeting at a ball.",
                    provenance=schema.Provenance(
                        kind="generated_concept", source="Requested during reading."
                    ),
                )
                intent = current.document.model_copy(
                    update={
                        "figures": [*current.document.figures, image],
                        "html": current.document.html.replace(
                            "</section>",
                            '<figure data-figure-id="requested-image"></figure></section>',
                            1,
                        ),
                    }
                )
                self.resources.compiler({})(intent)
                raise AssertionError(
                    "Managed browsing image policy admitted new generation."
                )
            raise schema.ContractError("image_generation_unavailable")
        if "interrupt" in request.instruction:
            raise schema.ContractError("interrupted")
        data = current.document.model_dump(mode="json")
        if "remove" in request.instruction:
            start, end = (
                data["html"].index('<section id="limits"'),
                data["html"].index("</body>"),
            )
            data["html"] = data["html"][:start] + data["html"][end:]
            data["figures"] = [
                figure for figure in data["figures"] if figure["content_id"] != "limits"
            ]
        else:
            answer = (
                '<p class="answer">Added explanation: calibrated rays give the '
                "ball position. Each camera supplies one direction; their intersection "
                "gives the shared position in space.</p>"
            )
            marker = (
                "<p>😀 Intersect"
                if request.document_id.startswith("managed-")
                else '<p data-evidence-id="baseline">'
            )
            assert marker in data["html"]
            data["html"] = data["html"].replace(marker, answer + marker, 1)
        document = schema.parse(schema.Document, json.dumps(data))
        if request.document_id.startswith("managed-"):
            document = self.resources.compiler({})(document)
        candidate = schema.Candidate(
            request=request,
            document=document,
            changes=document_api.content_changes(current.document, document),
            evidence_changes=[],
            change_summary="Remove the assumptions section."
            if "remove" in request.instruction
            else "Add an explanation beside the calibrated rays.",
        )
        return SimpleNamespace(
            candidate=candidate,
            validation=validation_for(candidate, rendering={"status": "not_run"}),
        )

    def finish(self) -> None:
        with self.condition:
            self.finishing = True
            for _, release in self.gates.values():
                release.set()
        self.controller.close()

    def release(self, request_id: str) -> None:
        with self.condition:
            assert self.condition.wait_for(lambda: request_id in self.gates, timeout=5)
            self.gates[request_id][1].set()

    def wait_entered(self, request_id: str) -> None:
        with self.condition:
            assert self.condition.wait_for(lambda: request_id in self.gates, timeout=5)
            entered = self.gates[request_id][0]
        assert entered.wait(5)


@contextmanager
def live_browser_server(root: Path) -> Iterator[Any]:
    world = BrowserWorld(root)
    ready = Event()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        app = service.create_app(world.controller, port=port, ui_files=web.files())

        @app.post("/__test__/control")
        async def control(request: Request) -> JSONResponse:
            if request.headers.get("X-Explain-Token") != app.state.session_token:
                return JSONResponse({}, status_code=401)
            body = await request.json()
            if body.get("action") == "release":
                await run_in_threadpool(world.release, body["request_id"])
            elif body.get("action") == "hold_undo":
                world.hold_undo = True
            elif body.get("action") == "wait_entered":
                await run_in_threadpool(world.wait_entered, body["request_id"])
            return JSONResponse(
                {
                    "calls": [call.model_dump(mode="json") for call in world.calls],
                    "restores": [
                        call.model_dump(mode="json") for call in world.restore_calls
                    ],
                    "revisions": {
                        name: world.store.current(name).revision_id
                        for name in world.initial
                    },
                }
            )

        class Server(uvicorn.Server):
            async def startup(self, sockets: list[socket.socket] | None = None) -> None:
                await super().startup(sockets=sockets)
                ready.set()

        server = Server(uvicorn.Config(app, log_config=None, access_log=False))
        thread = Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            assert ready.wait(10) and server.started
            yield SimpleNamespace(
                world=world,
                config={
                    "url": f"http://127.0.0.1:{port}",
                    "token": app.state.session_token,
                    "initial": {
                        name: revision.revision_id
                        for name, revision in world.initial.items()
                    },
                },
            )
        finally:
            world.finish()
            server.should_exit = True
            thread.join(10)
            assert not thread.is_alive()
