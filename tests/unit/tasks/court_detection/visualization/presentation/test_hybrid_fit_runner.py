"""Exercise the CLI's real source reader and exporter with CPU stub predictions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from unittest.mock import Mock

import av
import pytest
from hydra import compose, initialize_config_dir
from PIL import Image

from src.tasks.court_detection.inference.predictor import CourtPredictor
from src.tasks.court_detection.visualization.presentation import (
    hybrid_fit_runner as runner,
)
from src.tasks.court_detection.visualization.presentation.hybrid_fit_config import (
    HybridFitConfig,
)
from src.utils.data.image_record_store import ImageRecordWriter, encode_rgb_jpeg
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def configured_run(tmp_path, hybrid_prediction, monkeypatch):
    prediction, image = hybrid_prediction
    root = tmp_path / "data/court_detection/tennis_court_detector-v1"
    with ImageRecordWriter(root) as writer:
        for sample_id, split, shift in [
            ("train", "train", 0),
            ("val_best", "val", 0),
            ("val_second", "val", 1),
            ("val_rejected", "val", 5),
        ]:
            writer.append(
                encode_rgb_jpeg(image),
                {
                    "id": sample_id,
                    "split": split,
                    "metric": None,
                    "width": 320,
                    "height": 240,
                    "kps": (prediction.homography.best.projected + [shift, 0]).tolist(),
                },
            )
        storage = writer.finish(
            {"source_schema": "tennis_court_detector_annotations_v1"}
        )
    (root / "dataset.json").write_text(
        json.dumps({"schema": "tennis_court_detector_store_v1", "storage": storage})
    )
    checkpoint = tmp_path / "ckpt/court_detection/multiscale_depth3/b863df1f01f0.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test-checkpoint")
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "src/tasks/court_detection/configs"),
    ):
        config = compose(
            config_name="visualize_hybrid_fit",
            overrides=[
                f"paths.data_root={tmp_path / 'data'}",
                f"paths.checkpoint_root={tmp_path / 'ckpt'}",
                f"paths.output_root={tmp_path / 'outputs'}",
                "source.excluded_sample_ids=[]",
                "output.width=320",
                "output.height=240",
                "output.fps=1",
                "output.gif_width=160",
                "output.gif_fps=1",
            ],
        )
    predictor = Mock(spec=CourtPredictor)
    predictor.predict.return_value = prediction
    predictor.checkpoint_identity = {"schema": "test"}
    loader = Mock(return_value=predictor)
    monkeypatch.setattr(runner.CourtPredictor, "load_from_checkpoint", loader)
    return config, predictor, loader


def test_search_writes_sorted_thresholded_ranking_and_contact_sheet(
    configured_run,
) -> None:
    config, predictor, loader = configured_run
    runner.run_hybrid_fit(config)
    output = HybridFitConfig.from_config(config).output_dir
    ranking = json.loads((output / "ranking.json").read_text())
    assert ranking["scanned_count"] == 3
    assert ranking["qualified_count"] == 2
    assert [candidate["sample_id"] for candidate in ranking["ranking"]] == [
        "val_best",
        "val_second",
    ]
    assert len(ranking["scanned"]) == 3
    assert predictor.predict.call_count == 3
    assert loader.call_count == 1
    assert ranking["ranking"][0]["hybrid_error"] < 1e-4
    assert len(ranking["ranking"][0]["jpeg_sha256"]) == 64
    with Image.open(output / "contact_sheet.png") as sheet:
        assert sheet.size == (960, 310)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["selection"] == "candidate_for_user_review"
    assert (
        manifest["checkpoint"]["sha256"]
        == hashlib.sha256(b"test-checkpoint").hexdigest()
    )
    assert len(manifest["input"]["annotations"]["sha256"]) == 64
    with pytest.raises(FileExistsError):
        runner.run_hybrid_fit(config)


def test_search_limit_and_top_k_are_explicit(configured_run) -> None:
    config, predictor, _ = configured_run
    config.search.max_samples = 1
    config.search.top_k = 1
    runner.run_hybrid_fit(config)
    ranking = json.loads(
        (HybridFitConfig.from_config(config).output_dir / "ranking.json").read_text()
    )
    assert ranking["scanned_count"] == 1
    assert len(ranking["ranking"]) == 1
    assert predictor.predict.call_count == 1


def test_no_match_keeps_original_criteria_and_records_empty_result(
    configured_run,
) -> None:
    config, _, _ = configured_run
    config.search.min_kp_only_error_px = 1000
    with pytest.raises(ValueError, match="No samples met"):
        runner.run_hybrid_fit(config)
    output = HybridFitConfig.from_config(config).output_dir
    ranking = json.loads((output / "ranking.json").read_text())
    assert ranking["ranking"] == []
    assert ranking["criteria"]["min_kp_only_error_px"] == 1000
    assert (output / "contact_sheet.png").is_file()


def test_render_exports_real_stages_manifest_and_decodable_video(
    configured_run,
) -> None:
    config, predictor, _ = configured_run
    config.mode, config.sample_id = "render", "val_best"
    runner.run_hybrid_fit(config)
    output = HybridFitConfig.from_config(config).output_dir
    assert (output / "hybrid_fit.gif").is_file()
    assert sorted(path.stem for path in (output / "stills").glob("*.png")) == [
        "01_input",
        "02_kp",
        "03_kp_only",
        "04_line",
        "05_seeds",
        "06_hybrid",
    ]
    with av.open(str(output / "hybrid_fit.mp4")) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (320, 240)
        assert len(list(container.decode(stream))) == 14
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["input"]["sample_id"] == "val_best"
    assert manifest["errors"]["hybrid_error"] < 1e-4
    assert [stage["name"] for stage in manifest["stages"]] == [
        "input",
        "kp",
        "kp_only",
        "line",
        "seeds",
        "hybrid",
        "hold",
    ]
    assert manifest["provenance"]["hybrid"] == "measured"
    assert manifest["provenance"]["timing_and_colors"] == "schematic"
    assert manifest["seed_scores"] == [0.2, 0.8]
    assert predictor.predict.call_count == 1


def test_render_unknown_sample_does_not_substitute_another(configured_run) -> None:
    config, predictor, _ = configured_run
    config.mode, config.sample_id = "render", "absent"
    with pytest.raises(ValueError, match="Expected one"):
        runner.run_hybrid_fit(config)
    predictor.predict.assert_not_called()
    assert not HybridFitConfig.from_config(config).output_dir.exists()


def test_render_image_has_no_fabricated_gt_errors(
    configured_run, tmp_path, hybrid_prediction
) -> None:
    config, _, _ = configured_run
    _, image = hybrid_prediction
    path = tmp_path / "data/frame.png"
    Image.fromarray(image).save(path)
    config.mode, config.input.kind, config.input.path = "render", "image", "frame.png"
    runner.run_hybrid_fit(config)
    manifest = json.loads(
        (HybridFitConfig.from_config(config).output_dir / "manifest.json").read_text()
    )
    assert manifest["errors"] is None
    assert manifest["input"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_rejected_render_produces_no_presentation(configured_run) -> None:
    config, predictor, _ = configured_run
    config.mode, config.sample_id = "render", "val_best"
    prediction = predictor.predict.return_value
    predictor.predict.return_value = replace(
        prediction,
        homography=replace(prediction.homography, status="ambiguous_candidates"),
    )
    with pytest.raises(ValueError, match="status='ok'"):
        runner.run_hybrid_fit(config)
    assert not HybridFitConfig.from_config(config).output_dir.exists()
