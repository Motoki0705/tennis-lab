"""Scale CLI validates each subcommand's inputs before reading or rendering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.synthetic_data_generation.scripts import generate_court_scale
from src.utils.configuration.errors import PathContractError


def test_plan_rejects_saved_train_method_before_loading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("Read-only trajectory method reached scene loading.")

    monkeypatch.setattr(generate_court_scale, "load_scene_inputs", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_court_scale",
            "plan",
            "--scenes-root",
            "unused-scenes",
            "--scenes",
            "B00",
            "--methods",
            "saved_train_trajectories",
            "--output-root",
            "unused-output",
            "--generation-seed",
            "42",
            "--trajectory-count",
            "1",
            "--density-level",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as error:
        generate_court_scale.main()
    assert error.value.code == 2
    assert "invalid choice: 'saved_train_trajectories'" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["plan", "render", "render-colab", "validate"])
@pytest.mark.parametrize("input_kind", ["missing", "directory"])
def test_invalid_file_inputs_fail_before_loading_or_rendering(
    command: str,
    input_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = tmp_path / "invalid.json"
    if input_kind == "directory":
        invalid.mkdir()
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    quality = tmp_path / "quality.json"
    quality.write_text("{}")
    output = tmp_path / "output"
    options = {
        "plan": [
            "--scenes-root",
            str(scenes),
            "--scenes",
            "B00",
            "--methods",
            "stratified_counter",
            "--output-root",
            str(output),
            "--recipe",
            str(invalid),
            "--quality",
            str(quality),
            "--generation-seed",
            "42",
            "--trajectory-count",
            "1",
            "--density-level",
            "0",
        ],
        "render": ["--output-root", str(output), "--request", str(invalid)],
        "render-colab": [
            "--output-root",
            str(output),
            "--request",
            str(invalid),
            "--drive-path",
            "data/synthetic_data_generation/fixture",
        ],
        "validate": ["--dataset-version", str(invalid)],
    }[command]

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid paths reached loading, planning or rendering.")

    for name in (
        "read_json",
        "load_scene_inputs",
        "renderer_fingerprint",
        "LocalNHTRenderer",
        "load_dataset_version",
        "persist_candidate_plan",
    ):
        monkeypatch.setattr(generate_court_scale, name, unexpected)
    monkeypatch.setattr(sys, "argv", ["generate_court_scale", command, *options])

    with pytest.raises(PathContractError):
        generate_court_scale.main()

    assert not output.exists()


def test_render_passes_all_validated_requests_and_publishes_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    requests = [output / f"request-{i}.json" for i in range(2)]
    for path in requests:
        path.write_text('{"renderer": {"schema": "fixture_renderer_v1"}}')
    backend = object()
    consumed = []

    def renderer(root, *, fingerprint, batch_size):
        assert root == output
        assert fingerprint == {"schema": "fixture_renderer_v1"}
        assert batch_size == 2
        return backend

    def execute(root, request_path, render):
        assert root == output
        assert render is backend
        consumed.append(request_path)
        return request_path.with_suffix(".dataset.json")

    monkeypatch.setattr(generate_court_scale, "LocalNHTRenderer", renderer)
    monkeypatch.setattr(generate_court_scale, "execute_request", execute)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_court_scale",
            "render",
            "--output-root",
            "output",
            "--request",
            "output/request-0.json",
            "output/request-1.json",
            "--batch-size",
            "2",
        ],
    )

    generate_court_scale.main()

    assert consumed == requests
    catalog = json.loads((output / "review-catalog.json").read_text())
    assert catalog["dataset_versions"] == [
        str(path.with_suffix(".dataset.json")) for path in requests
    ]


def test_validate_passes_the_explicit_version_and_checks_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    version = tmp_path / "version.dataset.json"
    version.write_text("{}")

    def load(path, *, verify_images):
        assert path == version
        assert verify_images is True
        return {"dataset_version": "fixture"}

    monkeypatch.setattr(generate_court_scale, "load_dataset_version", load)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_court_scale", "validate", "--dataset-version", version.name],
    )

    generate_court_scale.main()

    assert capsys.readouterr().out.strip() == "verified fixture"


def test_fetch_passes_explicit_drive_path_and_request_to_cpu_recovery(
    tmp_path, monkeypatch, capsys
):
    remote = object()
    output = tmp_path / "recovered"
    drive_path = "data/synthetic_data_generation/court-scale/fixture"

    def drive(path):
        assert path == drive_path
        return remote

    def fetch(root, received, *, request_sha256):
        assert root == output and received is remote
        assert request_sha256 == "a" * 64
        return output / "fixture.dataset.json"

    monkeypatch.setattr(generate_court_scale, "SkillDriveFiles", drive)
    monkeypatch.setattr(generate_court_scale, "fetch_completed_dataset", fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_court_scale",
            "fetch",
            "--output-root",
            str(output),
            "--drive-path",
            drive_path,
            "--request-sha256",
            "a" * 64,
        ],
    )
    generate_court_scale.main()
    assert "verified Drive dataset" in capsys.readouterr().out
