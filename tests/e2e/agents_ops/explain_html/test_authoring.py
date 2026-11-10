"""One public skill CLI workflow with fake agy and real Chromium/store/HTTP.

Malformed output, races and security permutations remain in the upstream suites.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx2 as httpx
import pytest

from tests.support.agents_ops.explain_editor import candidate, schema
from tests.support.agents_ops.explain_editor import editor_env as editor_env
from tests.support.agents_ops.explain_html_visuals import images, png

REPO = Path(__file__).resolve().parents[4]
SCRIPTS = REPO / ".agents/skills/explain-html/scripts"
sys.path.insert(0, str(SCRIPTS))
author: Any = importlib.import_module("explain_html_author")


def cli(root: Path, command: str, *options: str, success: bool = True) -> Any:
    completed = subprocess.run(
        [
            str(REPO / ".venv/bin/python"),
            str(SCRIPTS / "explain_html_author.py"),
            command,
            "--root",
            str(root),
            *options,
        ],
        cwd=REPO,
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (completed.returncode == 0) == success, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


def review_for(created: Any) -> Any:
    """Explicit test double for the operator's review, not a prose evaluator."""
    candidate = schema.parse(
        schema.Candidate,
        (Path(created["review_directory"]) / "candidate.json").read_bytes(),
    )
    evidence = [item.evidence_id for item in candidate.document.evidence]
    reasoning = author.ReasoningReview(
        claim="Fixture only: no autonomous reasoning is simulated.",
        evidence_ids=evidence,
        assessment="The real Codex operator supplies this assessment in production.",
    )
    return author.CodexReview(
        reviewer="codex",
        candidate_digest=created["candidate_digest"],
        check=schema.CheckResult(status="passed"),
        evidence_checks=[
            author.EvidenceReview(evidence_id=identifier, assessment="Test fixture.")
            for identifier in evidence
        ],
        asynchronous_observations=reasoning,
        sensitivity_fixed_quantities=reasoning,
        conditional_generalizations=reasoning,
    )


@pytest.mark.e2e
def test_brief_create_open_update_reopen_stop_and_export(
    editor_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = editor_env
    monkeypatch.setenv("FAKE_EXPLAIN_BIND_REQUEST", "1")
    root = env.root / "author-runtime"
    image = images.image_from_bytes(
        png(),
        asset_id="scene",
        caption="試験用の概念画像。",
        alt="概念画像。",
        provenance=schema.Provenance(kind="generated_concept", source="Test fixture."),
    )
    image_path = env.root / "scene.png"
    image_path.write_bytes(image.content)
    figure = schema.Figure(
        figure_id="scene-image",
        content_id="geometry",
        kind="image",
        asset_ids=[image.asset.asset_id],
        caption=image.asset.caption,
        alt=image.asset.alt,
        provenance=image.asset.provenance,
    )
    document = env.current.document.model_copy(
        update={
            "html": env.current.document.html.replace(
                "<html><body>",
                '<html lang="ja"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                "<title>三角測量</title></head><body>",
            ).replace(
                "</section>",
                '<figure data-figure-id="scene-image"></figure></section>',
                1,
            ),
            "assets": [image.asset],
            "figures": [figure],
        }
    )
    brief = author.Brief(
        document_id=document.document_id,
        instruction="説明を日本語で書く。",
        context=env.context.model_copy(
            update={"assets": [image.asset], "figures": [figure]}
        ),
        asset_files={"scene": "scene.png"},
    )
    brief_path = env.root / "brief.json"
    brief_path.write_text(brief.model_dump_json(), encoding="utf-8")
    env.outputs(candidate(document))

    def run(command: str, *options: str, success: bool = True) -> Any:
        return cli(root, command, *options, success=success)

    created = run("create", "--brief", str(brief_path), "--agy", str(env.root / "agy"))
    assert created["document_id"] == "geometry"
    assert created["state"] == "awaiting_codex_review"
    store = author.RevisionStore(root)
    with pytest.raises(schema.ContractError, match="document_not_found"):
        store.current("geometry")
    assert (
        run("open", "--document", "geometry", success=False)["error"]["code"]
        == "document_not_found"
    )
    review_path = env.root / "codex-review.json"
    review = review_for(created)
    review_path.write_text(
        review.model_copy(update={"candidate_digest": "0" * 64}).model_dump_json()
    )
    assert (
        run(
            "publish",
            "--stage",
            created["stage_id"],
            "--review-file",
            str(review_path),
            success=False,
        )["error"]["code"]
        == "validation_mismatch"
    )
    review_path.write_text(
        review.model_copy(
            update={"evidence_checks": review.evidence_checks[:-1]}
        ).model_dump_json()
    )
    assert (
        run(
            "publish",
            "--stage",
            created["stage_id"],
            "--review-file",
            str(review_path),
            success=False,
        )["error"]["code"]
        == "invalid_request"
    )
    review_path.write_text(review.model_dump_json())
    published = run(
        "publish", "--stage", created["stage_id"], "--review-file", str(review_path)
    )
    assert (
        run(
            "publish", "--stage", created["stage_id"], "--review-file", str(review_path)
        )
        == published
    )
    first = store.current("geometry")
    assert first.revision_id == published["revision_id"]
    assert "間隔は7 m" in first.document.html
    assert store.asset_bytes("geometry", first.revision_id, "scene") == image.content
    calls = len(env.calls())
    assert (
        run("create", "--brief", str(brief_path), success=False)["error"]["code"]
        == "document_exists"
    )
    assert len(env.calls()) == calls  # no duplicate generation on create or open

    launched = run("open", "--document", "geometry", "--agy", str(env.root / "agy"))
    try:
        assert run("open", "--document", "geometry")["url"] == launched["url"]
        assert len(env.calls()) == calls
        parsed = urlsplit(launched["url"])
        token = parse_qs(parsed.fragment)["token"][0]
        headers = {"X-Explain-Token": token}
        base = "/api/documents/geometry"
        with httpx.Client(
            base_url=f"http://127.0.0.1:{parsed.port}", trust_env=False
        ) as client:
            assert 'id="edit-form"' in client.get("/").text

            def edit(identifier: str, after: Any) -> Any:
                current = store.current("geometry")
                request = schema.EditRequest(
                    document_id="geometry",
                    base_revision=current.revision_id,
                    request_id=identifier,
                    target=schema.ContentTarget(content_id="geometry"),
                    instruction="この図の読み方を追記して。",
                )
                env.outputs(candidate(after, current=current, request=request))
                response = client.post(
                    base + "/requests",
                    headers=headers,
                    json=request.model_dump(mode="json"),
                )
                assert response.status_code == 202
                deadline = time.monotonic() + 30
                while True:
                    job = client.get(
                        base + "/requests/" + identifier, headers=headers
                    ).json()
                    if job["state"] not in {"queued", "running", "validating"}:
                        return job
                    assert time.monotonic() < deadline

            changed = first.document.model_copy(
                update={
                    "html": first.document.html.replace(
                        "間隔は", "2方向から奥行きを決めます。間隔は"
                    )
                }
            )
            updated = edit("question", changed)
            assert updated["state"] == "succeeded", updated
            current = store.current("geometry")
            assert "2方向から奥行き" in current.document.html
            # The new glue must reject an actually invisible rendering rather
            # than converting a structural/semantic fake verdict into approval.
            hidden = current.document.model_copy(
                update={
                    "html": current.document.html.replace(
                        "</head>", "<style>body{display:none}</style></head>"
                    )
                }
            )
            rejected = edit("hidden", hidden)
            assert rejected["state"] == "failed", rejected
            assert rejected["error"]["code"] == "validation_failed"
            assert store.current("geometry").revision_id == current.revision_id
            restored = client.post(
                base + "/undo",
                headers=headers,
                json={
                    "document_id": "geometry",
                    "request_id": "undo",
                    "base_revision": current.revision_id,
                    "revision_id": first.revision_id,
                },
            )
            assert restored.status_code == 200
            assert store.current("geometry").document.html == first.document.html
    finally:
        run("stop")
    assert not run("status")["running"]
    exported = run("export", "--document", "geometry")
    html = (Path(exported["directory"]) / "index.html").read_text()
    assert html == first.document.html and token not in html
    assert (
        Path(exported["directory"]) / image.asset.path
    ).read_bytes() == image.content
    assert not any("private" in name for name in exported["files"])
    assert (
        author.validate_bundle(
            candidate(first.document), {"scene": image.content}, cancel=author.Event()
        ).status
        == "passed"
    )

    restarted = run("open", "--document", "geometry", "--agy", str(env.root / "agy"))
    try:
        assert restarted["url"] != launched["url"]
        assert len(store.history("geometry")) == 3
    finally:
        run("stop")


@pytest.mark.e2e
def test_failed_codex_review_requires_new_writer_and_all_checks(
    editor_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = editor_env
    monkeypatch.setenv("FAKE_EXPLAIN_BIND_REQUEST", "1")
    root = env.root / "review-runtime"
    document = env.current.document.model_copy(
        update={
            "html": env.current.document.html.replace(
                "<html><body>", "<html><head><title>Review</title></head><body>"
            )
        }
    )
    brief = author.Brief(
        document_id="geometry", instruction="初版を書く。", context=env.context
    )
    brief_path = env.root / "brief.json"
    brief_path.write_text(brief.model_dump_json())
    env.outputs(candidate(document))
    first = cli(
        root, "create", "--brief", str(brief_path), "--agy", str(env.root / "agy")
    )
    review = review_for(first)
    failed = review.model_copy(
        update={
            "check": schema.CheckResult(
                status="failed",
                issues=[
                    schema.ValidationIssue(
                        rule="causal_claim",
                        message="The test reviewer found an unsupported generalization.",
                    )
                ],
            )
        }
    )
    review_path = env.root / "review.json"
    review_path.write_text(failed.model_dump_json())
    assert (
        cli(
            root,
            "publish",
            "--stage",
            first["stage_id"],
            "--review-file",
            str(review_path),
            success=False,
        )["error"]["code"]
        == "validation_failed"
    )
    review_path.write_text(review.model_dump_json())
    assert (
        cli(
            root,
            "publish",
            "--stage",
            first["stage_id"],
            "--review-file",
            str(review_path),
            success=False,
        )["error"]["code"]
        == "request_conflict"
    )
    with pytest.raises(schema.ContractError, match="document_not_found"):
        author.RevisionStore(root).current("geometry")

    corrected = document.model_copy(
        update={
            "html": document.html.replace(
                "間隔は", "同時刻・校正済みという条件を確認します。間隔は"
            )
        }
    )
    brief = brief.model_copy(
        update={
            "instruction": "指摘した一般化を条件付きの説明へ修正し、初版を作り直す。"
        }
    )
    brief_path.write_text(brief.model_dump_json())
    env.outputs(candidate(corrected))
    second = cli(
        root, "create", "--brief", str(brief_path), "--agy", str(env.root / "agy")
    )
    assert second["stage_id"] != first["stage_id"]
    review_path.write_text(review_for(second).model_dump_json())
    cli(
        root,
        "publish",
        "--stage",
        second["stage_id"],
        "--review-file",
        str(review_path),
    )
    assert (
        "同時刻・校正済み"
        in author.RevisionStore(root).current("geometry").document.html
    )
    assert len([call for call in env.calls() if call["prompt"] is not None]) == 4
