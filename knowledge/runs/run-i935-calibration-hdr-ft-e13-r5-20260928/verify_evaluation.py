"""CPU audit of the captured validation diagnostic; run with its checkout on PYTHONPATH."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache, select_clips
from src.tasks.ball_refiner.data.gaps import fixed_gap_mask, validation_partition
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.data.windows import LoadedClip
from src.tasks.ball_refiner.evaluation.configuration import EvaluationConfig
from src.tasks.ball_refiner.evaluation.hdr import highest_density_regions
from src.tasks.ball_refiner.evaluation.runner import _summarize
from src.tasks.ball_refiner.refiner_2d import build_ball_refiner_2d
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.training.configuration import PilotConfig
from src.tasks.ball_refiner.training.evaluation import (
    metric_rows,
    predict_clip,
    summarize_rows,
)
from src.utils.checksum import dual_sha256


def audit(directory: Path) -> dict[str, Any]:
    torch.set_num_threads(4)
    config = EvaluationConfig.from_config(OmegaConf.load(directory / "config.yaml"))
    training = PilotConfig.from_config(OmegaConf.load(config.training_run / "config.yaml"))
    manifest = json.loads((directory / "manifest.json").read_text())
    data = json.loads((config.training_run / "data_manifest.json").read_text())
    state = json.loads((directory / "run_state.json").read_text())
    assert state == {"status": "complete", "partition": "calibration", "camera_clips": 18, "prediction_files": 36}
    assert manifest["schema"] == "ball_refiner_validation_diagnostics.v1"
    assert manifest["partition"] == config.partition == "calibration"
    for filename, digest in manifest["input_sha256"].items():
        assert dual_sha256(Path(filename)) == digest, filename
    assert json.loads((directory / "progress.json").read_text())["completed"] == manifest["artifacts"]
    store = BallFrameStore(training.store)
    cache = EvidenceCache(training.evidence, store)
    records = {r.clip_id: r for r in select_clips(store, sources=("meiji",), splits=("val",))}
    partition = validation_partition(list(records.values()), training.partition_seed)
    assert partition == data["validation"]
    assert manifest["clip_ids"] == partition[config.partition]
    assert set(partition["calibration"]).isdisjoint(partition["selection"])
    expected = {(key, condition) for key in manifest["clip_ids"] for condition in ("observed", "evidence_gap")}
    assert len(manifest["artifacts"]) == len(expected)
    assert {(a["clip_id"], a["condition"]) for a in manifest["artifacts"]} == expected
    assert set((directory / "predictions").iterdir()) == {directory / a["path"] for a in manifest["artifacts"]}
    best = json.loads((config.training_run / "best.json").read_text())
    assert best == manifest["checkpoint"]
    checkpoint = torch.load(config.training_run / best["checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["data_manifest_sha256"] == dual_sha256(config.training_run / "data_manifest.json")
    pair = build_ball_refiner_2d(training.model)
    pair.model.load_state_dict(checkpoint["state_dict"], strict=True)
    pair.model.eval()
    replay_id = min(manifest["clip_ids"], key=lambda key: records[key].frame_count)
    rows: dict[str, list[tuple[str, dict[str, NDArray[np.generic]]]]] = {"observed": [], "evidence_gap": []}
    full_rows: dict[str, list[dict[str, NDArray[np.generic]]]] = {key: [] for key in rows}
    hashes, replay, hdr_replay = {}, {}, {}
    for artifact in manifest["artifacts"]:
        clip_id, condition = artifact["clip_id"], artifact["condition"]
        record = records[clip_id]
        clip = LoadedClip(record, cache.load(clip_id), project_store_targets(store, record))
        path = directory / artifact["path"]
        assert path.name == f"clip-{record.index:05d}-{condition}.npz"
        assert dual_sha256(path) == artifact["sha256"]
        hashes[artifact["path"]] = artifact["sha256"]
        with np.load(path, allow_pickle=False) as file:
            arrays = {key: file[key] for key in file.files}
        for key in ("frame_index", "pts", "timestamps_seconds"):
            np.testing.assert_array_equal(arrays[key], getattr(clip.evidence, key))
        np.testing.assert_array_equal(arrays["target_uv"], clip.targets.uv)
        np.testing.assert_array_equal(arrays["position_valid"], clip.targets.position_valid)
        assert record.frame_count == artifact["frames"]
        assert artifact["source_size_wh"] == [record.source_width, record.source_height]
        gap = fixed_gap_mask(record.frame_count, clip_id=clip_id, block_length=training.window_length,
                             lengths=training.training.gap_lengths, seed=training.partition_seed)
        mask = gap if condition == "evidence_gap" else np.zeros_like(gap)
        np.testing.assert_array_equal(arrays["gap_mask"], mask)
        selected = gap if condition == "evidence_gap" else np.ones_like(gap)
        located = selected & clip.targets.position_valid
        assert artifact["scored_frames"] == int(located.sum())
        np.testing.assert_array_equal(arrays["scored_frame_index"], clip.evidence.frame_index[located])
        lengths = np.zeros(record.frame_count, dtype=np.int32)
        edges = np.diff(np.r_[False, gap, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            lengths[start:stop] = stop - start
        np.testing.assert_array_equal(arrays["gap_length"], lengths[located])
        prediction = BallGMM2D(**{f.name: torch.from_numpy(arrays[f.name])[None] for f in fields(BallGMM2D)})
        metrics = metric_rows(prediction, clip, selected)
        full_rows[condition].append(metrics)
        row = {key: metrics[key] for key in ("nll_uv", "nll_px", "error_px", "variance_px2")}
        for key, value in row.items():
            np.testing.assert_allclose(arrays[key], value, rtol=1e-7, atol=1e-7)
        scale = np.array(artifact["source_size_wh"], dtype=np.float64) - 1
        detector_error = np.linalg.norm((clip.evidence.argmax_uv[located] - clip.targets.uv[located]) * scale, axis=-1)
        np.testing.assert_array_equal(arrays["unmasked_detector_error_px"], detector_error)
        mixture = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=prediction.mixture_logits[0, located].double()),
            torch.distributions.MultivariateNormal(prediction.means[0, located].double(),
                                                   scale_tril=prediction.scale_tril[0, located].double()))
        density = mixture.log_prob(torch.from_numpy(clip.targets.uv[located]).double()).numpy()
        np.testing.assert_allclose(-density, arrays["nll_uv"], rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(density[:, None] >= arrays["hdr_log_threshold_uv"], arrays["coverage"])
        for key in ("area_px2", "area_mc_standard_error_px2"):
            assert np.isfinite(arrays[key]).all() and (arrays[key] >= 0).all()
        assert (np.diff(arrays["area_px2"], axis=-1) >= 0).all()
        assert (np.diff(arrays["hdr_log_threshold_uv"], axis=-1) <= 0).all()
        seed = int.from_bytes(hashlib.sha256(f"{config.seed}:{clip_id}:{condition}".encode()).digest()[:8], "little") % (2**63 - 1)
        assert artifact["mc_seed"] == seed
        count = min(32, int(located.sum()))
        subset = BallGMM2D(**{f.name: getattr(prediction, f.name)[:, located][:, :count] for f in fields(prediction)})
        hdr = highest_density_regions(subset, torch.from_numpy(clip.targets.uv[located][:count])[None],
                                      levels=config.levels, samples=config.samples, seed=seed, chunk_size=7)
        for key, actual in (("hdr_log_threshold_uv", hdr.log_threshold[0].numpy()),
                            ("area_px2", hdr.area_uv2[0].numpy() * scale.prod()),
                            ("area_mc_standard_error_px2", hdr.area_mc_standard_error_uv2[0].numpy() * scale.prod())):
            np.testing.assert_allclose(actual, arrays[key][:count], atol=1e-7, rtol=1e-9)
        np.testing.assert_array_equal(hdr.covered[0].numpy(), arrays["coverage"][:count])
        hdr_replay[artifact["path"]] = {"scored_frames": count, "samples": config.samples, "cpu_chunk_size": 7}
        row.update({key: arrays[key] for key in ("coverage", "area_px2", "area_mc_standard_error_px2",
                                               "unmasked_detector_error_px", "gap_length")})
        rows[condition].append((clip_id.rsplit("/", 1)[0], row))
        if clip_id == replay_id:
            restored = predict_clip(pair, clip, training, device=torch.device("cpu"), gap=mask)
            differences = {}
            for field in fields(prediction):
                actual, saved = getattr(restored, field.name), getattr(prediction, field.name)
                tolerance = 3e-4 if field.name.endswith("logits") else 3e-5
                torch.testing.assert_close(actual, saved, atol=tolerance, rtol=3e-4)
                differences[field.name] = float((actual - saved).abs().max())
            max_px = float(((restored.means - prediction.means).abs() * torch.from_numpy(scale)).max())
            assert max_px < .5
            replay[condition] = {"clip_id": clip_id, "frames": record.frame_count,
                                 "max_abs_difference": differences, "max_mean_coordinate_difference_px": max_px}
    recomputed = {key: _summarize(value, config, observed=key == "observed") for key, value in rows.items()}
    for condition, values in rows.items():
        for length in training.training.gap_lengths:
            subset_rows = [(group, {key: value[row["gap_length"] == length] for key, value in row.items()})
                           for group, row in values]
            recomputed[f"{condition}_at_gap_length_{length}"] = _summarize(subset_rows, config, observed=condition == "observed")
    saved_report = json.loads((directory / "metrics.json").read_text())
    assert recomputed == saved_report
    for length in training.training.gap_lengths:
        assert recomputed[f"observed_at_gap_length_{length}"]["frames"] == recomputed[f"evidence_gap_at_gap_length_{length}"]["frames"]
    return {
        "status": "passed", "scope": manifest["scope"], "run_state": state,
        "input_sha256": manifest["input_sha256"], "prediction_sha256": hashes,
        "recomputed_metrics": recomputed, "presence_and_position_summaries": {k: summarize_rows(v) for k, v in full_rows.items()},
        "cpu_checkpoint_replay": replay, "cpu_hdr_replay": hdr_replay,
        "limitations": ["HDR replay uses the first 32 scored frames per artifact, not every MC estimate",
                        "six temporal groups in one validation video; not independent test",
                        "bootstrap holds MC thresholds fixed; area SE excludes threshold uncertainty",
                        "no detector density, RGB occlusion, real amodal GT, context ablation or pipeline check",
                        "no raw media/JPEG/annotation rehash"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), ensure_ascii=False, indent=2, allow_nan=False))
