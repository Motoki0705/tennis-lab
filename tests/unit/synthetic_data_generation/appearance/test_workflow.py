"""Data integrity, input-coordinate mapping, retries and resumability."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import pytest
import requests
import yaml
from PIL import Image

from src.synthetic_data_generation.appearance.contracts import (
    OpenAIImageConfig,
    VariantConfig,
)
from src.synthetic_data_generation.appearance.generation import (
    next_request,
    record_result,
)
from src.synthetic_data_generation.appearance.openai_api import (
    check_api_setup,
    generate_next,
)
from src.synthetic_data_generation.appearance.sampling import sample_indices
from src.synthetic_data_generation.appearance.validation import validate_ready
from src.synthetic_data_generation.appearance.workspace import (
    load_manifest,
    prepare,
    revise_generation_size,
    sha256,
)


@pytest.fixture
def variant(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "frames/images").mkdir(parents=True)
    (source / "sfm/model").mkdir(parents=True)
    for name in ("cameras", "images", "points3D", "rigs", "frames"):
        (source / "sfm/model" / f"{name}.bin").write_bytes(name.encode())
    (source / "run.json").write_text("{}")
    (source / "sfm/reconstruction.json").write_text("{}")
    (source / "resolved-config.yaml").write_text(
        yaml.safe_dump({"nht_training": {"test_every": 8}})
    )
    records = []
    for index in range(10):
        name = f"frame_{index:06d}.jpg"
        Image.new("RGB", (32, 18), (index, 90, 70)).save(
            source / "frames/images" / name
        )
        records.append({"filename": name, "accepted": True})
    (source / "frames/frames.json").write_text(json.dumps({"frames": records}))
    config = VariantConfig(
        source_workspace=source,
        output_root=tmp_path / "variant",
        last_index=9,
        sample_count=4,
        reference_index=4,
        nht_source_root=tmp_path / "nht",
        training_python=tmp_path / "python",
    )
    prepare(config)
    return config.output_root


def accept_next(root: Path, tmp_path: Path, size: tuple[int, int] = (32, 18)) -> str:
    request = next_request(root)
    assert request is not None
    path = tmp_path / f"{request.request_id}.png"
    Image.new("RGB", size, (170, 65, 25)).save(path)
    record_result(
        root,
        request.request_id,
        path,
        accepted=True,
        review_notes="Synthetic fixture: known dimensions and uniform pixels",
    )
    return request.target


def test_b00_sampling_preserves_original_split() -> None:
    indices = sample_indices(0, 248, 50)
    assert len(indices) == len(set(indices)) == 50
    assert indices[0] == 0 and indices[-1] == 248
    assert [index for index in indices if index % 8 == 0] == [
        0,
        40,
        56,
        96,
        152,
        192,
        208,
        248,
    ]


def test_expansion_keeps_paid_views_and_covers_interval() -> None:
    from src.synthetic_data_generation.appearance.sampling import extend_indices

    original = sample_indices(0, 248, 50)
    expanded = extend_indices(original, 100)
    assert len(expanded) == len(set(expanded)) == 100
    assert set(original).issubset(expanded)
    assert max(b - a for a, b in zip(expanded[:-1], expanded[1:], strict=True)) <= 3
    assert extend_indices(original, 50) == original
    assert extend_indices(original, 100) == expanded
    with pytest.raises(ValueError):
        extend_indices(original, 49)


def test_cli_config_resolves_checkout_and_explicit_external_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omegaconf import OmegaConf

    from src.utils import hydra as hydra_support

    checkout = tmp_path / "different-checkout"
    monkeypatch.setattr(hydra_support, "PROJECT_ROOT", checkout)
    monkeypatch.delenv("APPEARANCE_DATA_ROOT", raising=False)
    monkeypatch.delenv("NHT_TRAINING_PYTHON", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    path = (
        Path(__file__).resolve().parents[4]
        / "src/synthetic_data_generation/configs/run_appearance_variant.yaml"
    )
    config = OmegaConf.load(path)
    variant = VariantConfig.model_validate(
        OmegaConf.to_container(config.variant, resolve=True)
    )
    assert variant.nht_source_root == checkout / "third_party/nht"
    assert (
        variant.source_workspace
        == checkout / "data/synthetic_data_generation/scenes/B00/reconstruction"
    )
    external = tmp_path / "shared-data"
    monkeypatch.setenv("APPEARANCE_DATA_ROOT", str(external))
    monkeypatch.setenv("NHT_TRAINING_PYTHON", str(tmp_path / "trainer/bin/python"))
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(tmp_path / "private.env"))
    variant = VariantConfig.model_validate(
        OmegaConf.to_container(config.variant, resolve=True)
    )
    assert variant.source_workspace == external / "scenes/B00/reconstruction"
    assert variant.training_python == tmp_path / "trainer/bin/python"
    assert (
        variant.api is not None and variant.api.api_key_file == tmp_path / "private.env"
    )


def test_common_metrics_uses_names_and_excludes_extra_validation_views() -> None:
    from src.synthetic_data_generation.appearance.reporting import common_metrics

    def metrics(value: float) -> dict[str, float]:
        return {"psnr": value, "ssim": value / 100, "lpips": 1 - value / 100}

    first = {"a.jpg": metrics(20), "c.jpg": metrics(30)}
    second = {"extra.jpg": metrics(99), "c.jpg": metrics(40), "a.jpg": metrics(22)}
    result = common_metrics([first, second])
    assert result["image_names"] == ["a.jpg", "c.jpg"]
    assert result["means"][0]["psnr"] == 25
    assert result["means"][1]["psnr"] == 31
    with pytest.raises(ValueError, match="no common"):
        common_metrics([first, {"other.jpg": metrics(3)}])


def test_prepare_copies_and_resume_verifies_source(variant: Path) -> None:
    manifest = load_manifest(variant)
    source = manifest.config.source_workspace / "frames/images/frame_000000.jpg"
    snapshot = variant / "inputs/targets/frame_000000.jpg"
    assert source.stat().st_ino != snapshot.stat().st_ino
    assert prepare(manifest.config) == manifest
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Original source artifact changed"):
        prepare(manifest.config)


def test_request_resume_is_stable_and_prompts_are_locked(variant: Path) -> None:
    request = next_request(variant)
    assert next_request(variant) == request
    assert len((variant / "generation/requests.jsonl").read_text().splitlines()) == 1
    (variant / "prompts/transfer.txt").write_text("changed")
    with pytest.raises(ValueError, match="Fixed prompt changed"):
        next_request(variant)


def test_explicit_input_resize_and_inverse_mapping(
    variant: Path, tmp_path: Path
) -> None:
    accept_next(variant, tmp_path)
    revise_generation_size(
        variant, (31, 18), reason="User requested native generator dimensions"
    )
    request = next_request(variant)
    assert request is not None
    with Image.open(request.referenced_image_paths[1]) as image:
        assert image.size == (31, 18)
    output = tmp_path / "generated.png"
    Image.new("RGB", (31, 18), (120, 45, 20)).save(output)
    result = record_result(
        variant,
        request.request_id,
        output,
        accepted=True,
        review_notes="Fixture matches explicit generation coordinate system",
    )
    assert result.accepted and result.raw_size == (31, 18)
    assert result.normalized_size == (32, 18)
    assert result.raw_sha256 == sha256(output)
    assert (
        result.processing is not None and "inverse recorded input" in result.processing
    )
    assert (variant / "provenance/revisions/0001/manifest.json").is_file()


def test_dimension_mismatch_retained_and_retry_limit_enforced(
    variant: Path, tmp_path: Path
) -> None:
    accept_next(variant, tmp_path)
    for _ in range(3):
        request = next_request(variant)
        assert request is not None
        output = tmp_path / "wrong.png"
        Image.new("RGB", (32, 17)).save(output)
        result = record_result(
            variant,
            request.request_id,
            output,
            accepted=True,
            review_notes="Review cannot override coordinate mismatch",
        )
        assert not result.accepted
        assert (variant / result.raw_path).is_file()
    with pytest.raises(ValueError, match="Retry limit"):
        next_request(variant)
    assert not list((variant / "generation/accepted").iterdir())


def test_all_selected_frames_required_for_training(
    variant: Path, tmp_path: Path
) -> None:
    accept_next(variant, tmp_path)
    with pytest.raises(ValueError, match="Unaccepted"):
        validate_ready(variant, load_manifest(variant))
    for _ in range(4):
        accept_next(variant, tmp_path)
    assert next_request(variant) is None
    validate_ready(variant, load_manifest(variant))
    (variant / "generation/accepted/extra.jpg").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="exactly match"):
        validate_ready(variant, load_manifest(variant))


def test_reference_mutation_is_detected(variant: Path, tmp_path: Path) -> None:
    accept_next(variant, tmp_path)
    Image.new("RGB", (32, 18)).save(variant / "reference/clay.png")
    with pytest.raises(ValueError, match="Accepted image changed"):
        next_request(variant)


@pytest.fixture
def api_variant(variant: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    secret = tmp_path / "openai.env"
    secret.write_text("OPENAI_API_KEY=sk-test-never-log\n")
    config: VariantConfig = load_manifest(variant).config.model_copy(
        update={
            "output_root": variant.with_name("api-variant"),
            "generation_provider": "openai_api",
            "generation_size": (1536, 864),
            "api": OpenAIImageConfig(api_key_file=secret),
        }
    )
    prepare(config)
    return config.output_root


def api_response(size: tuple[int, int] = (1536, 864)) -> requests.Response:
    encoded = BytesIO()
    Image.new("RGB", size, (140, 60, 20)).save(encoded, format="PNG")
    response = requests.Response()
    response.status_code = 200
    response.headers["x-request-id"] = "req-fixture"
    response._content = json.dumps(
        {
            "created": 123,
            "data": [{"b64_json": base64.b64encode(encoded.getvalue()).decode()}],
            "usage": {"total_tokens": 10},
        }
    ).encode()
    return response


def test_api_key_missing_fails_before_claim_or_network(
    api_variant: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_manifest(api_variant).config
    assert config.api is not None
    config.api.api_key_file.write_text("OPENAI_API_KEY=\n")
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: pytest.fail(
            "Missing key must not send network requests"
        ),
    )
    assert check_api_setup(api_variant)["key_configured"] is False
    with pytest.raises(ValueError, match="currently missing"):
        generate_next(api_variant)
    assert load_manifest(api_variant).pending is None


def test_api_uses_exact_model_and_order_then_reuses_saved_response(
    api_variant: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = []

    def post(url: str, **kwargs: object) -> requests.Response:
        assert url == "https://api.openai.com/v1/images/edits"
        data = kwargs["data"]
        assert isinstance(data, dict)
        assert data["model"] == "gpt-image-2.5-sunburst-2026-09-08"
        assert data["size"] == "1536x864"
        assert "input_fidelity" not in data
        assert "seed" not in data
        files = kwargs["files"]
        assert isinstance(files, list)
        assert all(name == "image[]" for name, _ in files)
        paths = [Path(stream.name) for _, (_, stream, _) in files]
        captured.append(paths)
        return api_response()

    monkeypatch.setattr(requests, "post", post)
    result = generate_next(api_variant)
    assert len(captured[0]) == 1
    assert captured[0][0].name == "reference-source.png"
    assert generate_next(api_variant)["cached"] is True
    assert len(captured) == 1
    record_result(
        api_variant,
        result["request_id"],
        Path(result["path"]),
        accepted=True,
        review_notes="Synthetic API reference",
    )
    target = generate_next(api_variant)
    assert captured[1][0] == api_variant / "reference/clay.png"
    assert captured[1][1] == api_variant / "inputs/generation/frame_000000.png"
    assert target["metadata"]["api_request_id"] == "req-fixture"
    assert target["metadata"]["usage"] == {"total_tokens": 10}
    for path in api_variant.rglob("*.json"):
        assert "sk-test-never-log" not in path.read_text()


def test_api_error_is_redacted_and_requires_explicit_retry(
    api_variant: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = requests.Response()
    response.status_code = 401
    response._content = json.dumps(
        {"error": {"message": "Invalid sk-test-never-log"}}
    ).encode()
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match="request failed"):
        generate_next(api_variant)
    with pytest.raises(ValueError, match="explicit api_retry"):
        generate_next(api_variant)
    errors = list(api_variant.rglob("api-error.json"))
    assert len(errors) == 1 and "sk-test-never-log" not in errors[0].read_text()
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: api_response())
    result = generate_next(api_variant, retry_failed_request=True)
    assert result["cached"] is False
    assert len(list(api_variant.rglob("api-call-*.json"))) == 2


def test_api_wrong_dimensions_cannot_be_accepted(
    api_variant: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        requests, "post", lambda *args, **kwargs: api_response((1535, 864))
    )
    result = generate_next(api_variant)
    accepted = record_result(
        api_variant,
        result["request_id"],
        Path(result["path"]),
        accepted=True,
        review_notes="Dimension gate must overrule acceptance",
    )
    assert not accepted.accepted
    assert load_manifest(api_variant).reference_accepted_attempt is None


def test_api_output_tampering_cannot_trigger_regeneration(
    api_variant: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: api_response())
    result = generate_next(api_variant)
    Path(result["path"]).write_bytes(b"changed")
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: pytest.fail("Tampered cache must fail closed"),
    )
    with pytest.raises(ValueError, match="Saved API result"):
        generate_next(api_variant)


def test_finalize_uses_public_import_and_queue_uses_common_root(
    variant: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.synthetic_data_generation.appearance import nht

    for _ in range(5):
        accept_next(variant, tmp_path)
    manifest = load_manifest(variant)
    calls = []

    def invoke(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if command[0] == "nht-reconstruct":
            target = Path(command[command.index("--workspace") + 1])
            assert "--prepare-only" in command
            assert command[command.index("--from-stage") + 1] == "nht_training"
            shutil.copytree(
                manifest.config.source_workspace / "sfm/model", target / "sfm/model"
            )
            shutil.copytree(variant / "generation/accepted", target / "frames/images")
            (target / "import-provenance.json").write_text(
                json.dumps(
                    {
                        "image_names": [frame.name for frame in manifest.frames],
                        "validation_names": [
                            frame.name
                            for frame in manifest.frames
                            if frame.split == "validation"
                        ],
                    }
                )
            )
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[2] == "add":
            assert (
                "--resource" in command
                and command[command.index("--resource") + 1] == "all"
            )
            environment = kwargs["env"]
            assert isinstance(environment, dict)
            assert environment["TRAINING_QUEUE_DIR"] == str(
                tmp_path / "main/.training_queue"
            )
            return subprocess.CompletedProcess(command, 0, "queued: fixture.job\n", "")
        assert command[2] == "start"
        return subprocess.CompletedProcess(
            command, 1, "", "worker already running (PID 123)."
        )

    monkeypatch.setattr(subprocess, "run", invoke)
    assert nht.finalize(variant)["status"] == "finalized"
    assert nht.finalize(variant)["status"] == "finalized"
    assert sum(command[0] == "nht-reconstruct" for command, _ in calls) == 1
    monkeypatch.setattr(nht, "save_code_provenance", lambda *args: None)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: str(tmp_path / "main/.git") + "\n",
    )
    monkeypatch.setenv("CODEX_THREAD_ID", "test-session")
    nht.enqueue_training(variant)
    nht.enqueue_training(variant)
    assert (
        sum(command[0] == "bash" and command[2] == "add" for command, _ in calls) == 1
    )
    failed_job = tmp_path / "main/.training_queue/failed/fixture.job"
    failed_job.parent.mkdir(parents=True)
    failed_job.touch()
    with pytest.raises(ValueError, match="training_retry=true"):
        nht.enqueue_training(variant)
    nht.enqueue_training(variant, retry_failed_job=True)
    assert (
        sum(command[0] == "bash" and command[2] == "add" for command, _ in calls) == 2
    )
    assert (variant / "provenance/queue-attempts/fixture.job.json").is_file()
    monkeypatch.delenv("TENNIS_RUN_ID", raising=False)
    with pytest.raises(ValueError, match="shared training queue"):
        nht.execute_training(variant)


def test_model_comparison_sends_identical_inputs_and_reuses_results(
    api_variant: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.synthetic_data_generation.appearance.comparison import compare_models

    reference = tmp_path / "clay-reference.png"
    Image.new("RGB", (32, 18), (150, 60, 20)).save(reference)
    captured = []

    def post(url: str, **kwargs: object) -> requests.Response:
        data = kwargs["data"]
        files = kwargs["files"]
        assert isinstance(data, dict) and isinstance(files, list)
        captured.append((dict(data), [stream.read() for _, (_, stream, _) in files]))
        return api_response()

    monkeypatch.setattr(requests, "post", post)
    root = tmp_path / "comparison"
    config = load_manifest(api_variant).config
    result = compare_models(config, root, reference, 0)
    assert Path(result["comparison_image"]).is_file()
    assert len(captured) == 2

    first, second = captured
    assert {first[0]["model"], second[0]["model"]} == {
        "gpt-image-2.5-sunburst-2026-09-08",
        "gpt-image-2.5-flare-2026-09-08",
    }
    assert first[1] == second[1]
    assert {key: value for key, value in first[0].items() if key != "model"} == {
        key: value for key, value in second[0].items() if key != "model"
    }
    compare_models(config, root, reference, 0)
    assert len(captured) == 2


def test_batch_preserves_review_gate_and_resumes_without_new_calls(
    api_variant: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.synthetic_data_generation.appearance.batch import (
        generate_batch,
        import_reference,
    )

    reference = tmp_path / "approved-reference.png"
    Image.new("RGB", (1536, 864), (155, 65, 25)).save(reference)
    import_reference(api_variant, reference, review_notes="Known fixture reference")
    calls = []

    def post(*args: object, **kwargs: object) -> requests.Response:
        calls.append(1)
        return api_response()

    monkeypatch.setattr(requests, "post", post)
    generate_batch(api_variant, indices=[0, 9], start_interval_seconds=0)
    assert len(calls) == 2
    manifest = load_manifest(api_variant)
    assert manifest.pending is None and all(
        frame.accepted_attempt is None for frame in manifest.frames
    )
    cached = generate_next(api_variant)
    assert cached["cached"] is True and len(calls) == 2
    record_result(
        api_variant,
        cached["request_id"],
        Path(cached["path"]),
        accepted=True,
        review_notes="Known fixture geometry",
    )
    generate_batch(api_variant, indices=[0, 9], start_interval_seconds=0)
    assert len(calls) == 2
    Image.new("RGB", (1536, 864), "black").save(api_variant / "reference/clay.png")
    with pytest.raises(ValueError, match="Imported fixed reference changed"):
        generate_batch(api_variant, start_interval_seconds=0)


def test_reuse_comparison_requires_identical_model_prompt_and_inputs(
    api_variant: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.synthetic_data_generation.appearance.batch import (
        import_reference,
        reuse_comparison,
    )
    from src.synthetic_data_generation.appearance.comparison import compare_models

    reference = tmp_path / "reference-reuse.png"
    Image.new("RGB", (1536, 864), (150, 60, 25)).save(reference)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: api_response())
    comparison = tmp_path / "paid-comparison"
    compare_models(load_manifest(api_variant).config, comparison, reference, 0)
    import_reference(
        api_variant,
        comparison / "inputs/reference.png",
        review_notes="Reviewed comparison reference",
    )
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: pytest.fail("Reuse must not call the API"),
    )
    with pytest.raises(ValueError, match="identical inputs and generation parameters"):
        reuse_comparison(api_variant, comparison / "flare")
    result = reuse_comparison(api_variant, comparison / "sunburst")
    reviewed = record_result(
        api_variant,
        result["request_id"],
        Path(result["path"]),
        accepted=True,
        review_notes="Reviewed cached comparison",
    )
    assert reviewed.accepted
    assert reviewed.tool_metadata["new_api_call"] is False


def test_derived_variant_preserves_paid_results_and_parent(
    api_variant: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.synthetic_data_generation.appearance.derived import derive_variant

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: api_response())
    while load_manifest(api_variant).status != "ready":
        generated = generate_next(api_variant)
        record_result(
            api_variant,
            generated["request_id"],
            Path(generated["path"]),
            accepted=True,
            review_notes="Explicit parent fixture review",
        )
    parent_hash = sha256(api_variant / "manifest.json")
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: pytest.fail("Derivation must not call API"),
    )
    output = tmp_path / "derived"
    result = derive_variant(
        api_variant, output, scene_id="derived", sample_count=6, max_steps=30000
    )
    assert result["reused_images"] == 4 and result["new_images_required"] == 2
    manifest = load_manifest(output)
    assert manifest.status == "generating" and manifest.config.max_steps == 30000
    assert sum(frame.accepted_attempt is not None for frame in manifest.frames) == 4
    assert not (output / "reconstruction").exists()
    for frame in load_manifest(api_variant).frames:
        parent_image = api_variant / "generation/accepted" / frame.name
        child_image = output / "generation/accepted" / frame.name
        assert sha256(parent_image) == sha256(child_image)
        assert parent_image.stat().st_ino != child_image.stat().st_ino
    assert (
        derive_variant(
            api_variant, output, scene_id="derived", sample_count=6, max_steps=30000
        )
        == result
    )
    assert sha256(api_variant / "manifest.json") == parent_hash
    with pytest.raises(ValueError, match="different configuration"):
        derive_variant(
            api_variant, output, scene_id="derived", sample_count=7, max_steps=30000
        )
