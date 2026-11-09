"""Six representative HTTP/CLI boundaries, with OS-assigned loopback ports.

Jobs' input/state combinations live in integration/test_jobs.py. These tests
exercise real HTTP and CLI processes, never external APIs or fixed sleeps.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import uvicorn

from tests.support.agents_ops.explain_editor import editor_env as editor_env
from tests.support.agents_ops.explain_html import PAYLOADS, schema
from tests.support.agents_ops.explain_jobs import Gate, JobWorld, service
from tests.support.agents_ops.explain_jobs import job_world as job_world

pytestmark = pytest.mark.e2e
REPO = Path(__file__).resolve().parents[4]


@contextmanager
def live_server(world: Any) -> Iterator[Any]:
    ready = Event()
    controller = world.controller()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        app = service.create_app(controller, port=port)

        class NotifyingServer(uvicorn.Server):
            async def startup(self, sockets: list[socket.socket] | None = None) -> None:
                await super().startup(sockets=sockets)
                ready.set()

        server = NotifyingServer(uvicorn.Config(app, log_config=None, access_log=False))
        thread = Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            assert ready.wait(5) and server.started
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=2
            ) as client:
                yield SimpleNamespace(
                    client=client,
                    controller=controller,
                    world=world,
                    port=port,
                    headers={"X-Explain-Token": app.state.session_token},
                    token=app.state.session_token,
                )
        finally:
            world.finish()
            server.should_exit = True
            thread.join(5)
            assert not thread.is_alive()


@pytest.fixture
def live(job_world: Any) -> Iterator[Any]:
    with live_server(job_world) as running:
        yield running


def test_async_edit_keeps_old_page_readable_and_status_does_not_restart(
    live: Any,
) -> None:
    world, client = live.world, live.client
    gate = world.editor.gates["edit"] = Gate()
    base = "/api/documents/ray-geometry"
    response = client.post(
        base + "/requests",
        headers=live.headers,
        json=world.request().model_dump(mode="json"),
        timeout=1,
    )
    assert response.status_code == 202
    assert gate.entered.wait(5)
    for _ in range(3):
        assert (
            client.get(base + "/requests/edit", headers=live.headers).json()["state"]
            == "running"
        )
    assert (
        client.get(base, headers=live.headers).json()["revision"]["revision_id"]
        == world.first.revision_id
    )
    content = base + f"/revisions/{world.first.revision_id}/content/index.html"
    page = client.get(content)
    assert page.text == world.first.document.html
    assert "sandbox allow-scripts" in page.headers["content-security-policy"]
    assert "allow-same-origin" not in page.headers["content-security-policy"]
    gate.release.set()
    done = live.controller.wait("ray-geometry", "edit", timeout=5)
    status = client.get(base + "/requests/edit", headers=live.headers).json()
    assert status["state"] == "succeeded" and status["revision_id"] == done.revision_id
    assert (
        client.post(
            base + "/requests",
            headers=live.headers,
            json=world.request().model_dump(mode="json"),
        ).json()
        == status
    )
    assert len(world.editor.calls) == len(world.commits) == 1
    assert client.get(content).content == page.content
    restored = client.post(
        base + "/undo",
        headers=live.headers,
        json={
            "document_id": "ray-geometry",
            "request_id": "undo",
            "base_revision": done.revision_id,
            "revision_id": world.first.revision_id,
        },
    )
    assert restored.status_code == 200
    assert restored.json()["revision"]["document"]["html"] == page.text
    assert (
        len(client.get(base + "/revisions", headers=live.headers).json()["revisions"])
        == 3
    )


def test_host_origin_token_route_and_body_rejections_do_not_generate(live: Any) -> None:
    client, world = live.client, live.world
    path = "/api/documents/ray-geometry/requests"
    payload = world.request().model_dump(mode="json")
    cases = [
        ({**live.headers, "Host": "localhost.evil.example"}, "invalid_host"),
        ({**live.headers, "Origin": "https://evil.example"}, "invalid_origin"),
        ({**live.headers, "Origin": "null"}, "invalid_origin"),
        ({"X-Explain-Token": "incorrect"}, "unauthorized"),
        ({}, "unauthorized"),
    ]
    for headers, code in cases:
        response = client.post(path, headers=headers, json=payload)
        assert response.status_code == schema.ERRORS[code][0]
        assert response.json()["error"]["code"] == code
    for changed in (
        {**payload, "document_id": "other"},
        {**payload, "validation": {"passed": True}},
    ):
        assert (
            client.post(path, headers=live.headers, json=changed).json()["error"][
                "code"
            ]
            == "invalid_request"
        )
    assert (
        client.get(path + "/missing", headers=live.headers).json()["error"]["code"]
        == "request_not_found"
    )
    assert (
        client.post(
            path + "/edit/cancel",
            headers=live.headers,
            json={"document_id": "ray-geometry", "request_id": "different"},
        ).json()["error"]["code"]
        == "invalid_request"
    )
    assert (
        client.get(
            "/api/documents/ray-geometry?token=private-secret", headers=live.headers
        ).json()["error"]["code"]
        == "unsafe_path"
    )
    assert world.editor.calls == [] and world.commits == []


def test_session_tokens_are_independent_and_isolated_content_gets_no_authority(
    live: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_SESSION_ID", "unregistered-parent-session")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "ambiguous-parent-session")
    with live_server(JobWorld(tmp_path / "other-runtime")) as other:
        assert live.token != other.token
        path = "/api/documents/ray-geometry"
        assert other.client.get(path, headers=live.headers).status_code == 401
        assert other.client.get(path, headers=other.headers).status_code == 200
        assert (
            other.client.get(
                path, headers={**other.headers, "Origin": "null"}
            ).status_code
            == 403
        )
        assert (
            other.client.get(
                path,
                headers={**other.headers, "Origin": f"http://localhost:{other.port}"},
            ).status_code
            == 200
        )
        public = other.client.get(
            path + f"/revisions/{other.world.first.revision_id}/content/index.html",
            headers={"Origin": "null"},
        )
        assert public.status_code == 200
        for page in (public, other.client.get("/"), other.client.get("/ui/index.html")):
            assert other.token not in page.text and live.token not in page.text
            assert "set-cookie" not in page.headers
            assert page.headers["referrer-policy"] == "no-referrer"
            assert page.headers["cache-control"] == "no-store"
            assert "access-control-allow-origin" not in page.headers


def test_revision_assets_exports_and_private_files_use_closed_allowlists(
    live: Any,
) -> None:
    client, world = live.client, live.world
    (world.root / "private/prompt.txt").write_text("private prompt and API credentials")
    base = "/api/documents/ray-geometry"
    revision = base + f"/revisions/{world.first.revision_id}"
    asset = client.get(revision + "/content/assets/court.svg")
    assert (
        asset.content == PAYLOADS["court"]
        and asset.headers["content-type"] == "image/svg+xml"
    )
    exported = client.post(
        base + "/exports",
        headers=live.headers,
        json={"document_id": "ray-geometry", "revision_id": world.first.revision_id},
    )
    assert exported.status_code == 201
    snapshot = exported.json()
    assert set(snapshot["files"]) == {
        "index.html",
        "evidence.json",
        "figures.json",
        "manifest.json",
        "assets/court.svg",
        "assets/theme.css",
        "assets/plot_renderer.mjs",
    }
    export = f"/api/exports/{snapshot['snapshot_id']}"
    assert client.get(export + "/index.html").text == world.first.document.html
    assert client.get(export + "/assets/court.svg").content == PAYLOADS["court"]
    for path in (
        "/private/prompt.txt",
        revision + "/content/record.json",
        revision + "/content/assets/missing.svg",
        export + "/private/prompt.txt",
        export + "/%2e%2e/private/prompt.txt",
        revision + "/content/assets/%2fcourt.svg",
    ):
        rejected = client.get(path, headers=live.headers)
        assert rejected.status_code >= 400
        assert "private prompt" not in rejected.text


def test_unexpected_editor_and_storage_errors_have_only_public_messages(
    live: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "PRIVATE PROMPT credential=must-never-reach-http"

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(secret)

    monkeypatch.setattr(live.world.editor, "edit", fail)
    base = "/api/documents/ray-geometry"
    assert (
        live.client.post(
            base + "/requests",
            headers=live.headers,
            json=live.world.request().model_dump(mode="json"),
        ).status_code
        == 202
    )
    failed = live.controller.wait("ray-geometry", "edit", timeout=5)
    assert failed.state == "failed" and failed.error.code == "generation_failed"
    status = live.client.get(base + "/requests/edit", headers=live.headers)
    assert secret not in status.text
    monkeypatch.setattr(live.world.store, "current", fail)
    response = live.client.get(base, headers=live.headers)
    assert (
        response.status_code == 500
        and response.json()["error"]["code"] == "storage_error"
    )
    assert secret not in response.text


def test_public_cli_start_stop_restart_and_offline_export(editor_env: Any) -> None:
    env = editor_env
    process_env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO / ".agents/ops"), str(REPO)]),
    }

    def cli(command: str, *args: str) -> Any:
        result = subprocess.run(
            [
                str(REPO / ".venv/bin/python"),
                "-m",
                "explain_html",
                command,
                "--root",
                str(env.store.root),
                *args,
            ],
            cwd=REPO,
            env=process_env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)

    def start() -> Any:
        return cli(
            "start",
            "--port",
            "0",
            "--document",
            "geometry",
            "--agy",
            str(env.root / "agy"),
            "--validator",
            "tests.support.agents_ops.explain_jobs:fake_rendering",
            "--compiler",
            "tests.support.agents_ops.explain_jobs:identity_compiler",
        )

    first = start()
    try:
        launched = urlsplit(first["url"])
        token = parse_qs(launched.fragment)["token"][0]
        assert launched.hostname == "localhost" and launched.port == first["port"]
        assert parse_qs(launched.fragment)["document_id"] == ["geometry"]
        assert cli("status")["running"]
        with httpx.Client(
            base_url=f"http://127.0.0.1:{first['port']}", trust_env=False
        ) as client:
            base = "/api/documents/geometry"
            headers = {"X-Explain-Token": token}
            assert (
                client.post(
                    base + "/requests",
                    headers=headers,
                    json=env.request.model_dump(mode="json"),
                ).status_code
                == 202
            )
            # Exercise the supported polling protocol with a deadline, not a
            # guessed sleep. Each read must be side-effect free.
            deadline = time.monotonic() + 5
            while True:
                job = client.get(base + "/requests/explain", headers=headers).json()
                if job["state"] not in {"queued", "running", "validating"}:
                    break
                assert time.monotonic() < deadline
            assert job["state"] == "succeeded", job
        assert len([call for call in env.calls() if call["prompt"] is not None]) == 2
    finally:
        assert cli("stop")["stopped"]
    assert not cli("status")["running"]
    exported = cli("export", "--document", "geometry")
    assert (
        Path(exported["directory"]) / "index.html"
    ).read_text() == env.proposed.document.html
    assert token not in (Path(exported["directory"]) / "index.html").read_text()
    second = start()
    try:
        assert second["url"] != first["url"]
        with httpx.Client(
            base_url=f"http://127.0.0.1:{second['port']}", trust_env=False
        ) as client:
            assert client.get(base, headers=headers).status_code == 401
            new_token = parse_qs(urlsplit(second["url"]).fragment)["token"][0]
            new_headers = {"X-Explain-Token": new_token}
            assert (
                client.get(base + "/requests/explain", headers=new_headers).json()[
                    "state"
                ]
                == "succeeded"
            )
            assert (
                client.get(base, headers=new_headers).json()["revision"]["revision_id"]
                == job["revision_id"]
            )
        assert len([call for call in env.calls() if call["prompt"] is not None]) == 2
    finally:
        cli("stop")
    assert (env.store.root / "private/session.json").stat().st_mode & 0o077 == 0
