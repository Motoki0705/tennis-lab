"""CPU calibration of immutable saved validation predictions, with OOF scoring."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tasks.ball_refiner.data.gaps import fixed_gap_mask, validation_partition
from src.tasks.ball_refiner.data.targets import ClipTargets, project_store_targets
from src.tasks.ball_refiner.evaluation.cached_comparison import (
    METRICS,
    paired_reference,
)
from src.tasks.ball_refiner.evaluation.calibration_fit import (
    cross_validate_clips,
    density_terms,
)
from src.tasks.ball_refiner.evaluation.detector_selection import validation_clips
from src.tasks.ball_refiner.evaluation.paired_metrics import (
    Array,
    gmm_rows,
    strata,
    summarize,
)
from src.tasks.ball_refiner.refiner_2d.calibration import (
    CovarianceCalibration,
    load_covariance_calibration,
)
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.training.configuration import PilotConfig
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256


@dataclass(frozen=True)
class SavedPrediction:
    record: ClipRecord
    condition: str
    target: ClipTargets
    arrays: dict[str, Array]
    distribution: BallGMM2D
    half: str | None
    entry: dict[str, Any]

    @property
    def group(self) -> str:
        return str(self.record.clip_id.rsplit("/", 1)[0])

    @property
    def scored(self) -> Array:
        active = self.arrays["gap_mask"] if self.condition == "evidence_gap" else np.ones(len(self.target.pts), bool)
        return self.target.position_valid & active

    @property
    def groups(self) -> list[str]:
        record = self.record
        result = [record.source]
        if record.camera_id is not None:
            result.append(f"{record.source}/{record.camera_id}")
        if self.half is not None:
            result.extend([f"meiji/{self.half}", f"meiji/{self.half}/{record.camera_id}"])
        return result


def _load_saved(source: Path, manifest: dict[str, Any], config: PilotConfig) -> list[SavedPrediction]:
    store = BallFrameStore(config.store)
    records = validation_clips(store)
    partition = validation_partition(tuple(r for r in records if r.source == "meiji"), config.partition_seed)
    if partition != manifest["partition"]:
        raise ValueError("Saved validation partition changed")
    entries = {(a["clip_id"], a["condition"]): a for a in manifest["artifacts"]}
    if (len(entries) != len(manifest["artifacts"]) or set(entries) !=
            {(r.clip_id, c) for r in records for c in ("observed", "evidence_gap")}):
        raise ValueError("Require exactly every validation clip and condition")
    saved = []
    for record in records:
        target = project_store_targets(store, record)
        gap = fixed_gap_mask(record.frame_count, clip_id=record.clip_id, block_length=config.window_length,
                             lengths=config.training.gap_lengths, seed=config.partition_seed)
        for condition in ("observed", "evidence_gap"):
            entry = entries[record.clip_id, condition]
            if (entry["source"] != record.source or entry["camera"] != record.camera_id
                    or entry["frames"] != record.frame_count):
                raise ValueError("Saved prediction metadata differs from validation store")
            fixed = {"frame_index": target.frame_index, "pts": target.pts, "target_uv": target.uv,
                     "target_reason": target.reason, "presence": target.presence,
                     "presence_valid": target.presence_valid,
                     "gap_mask": gap if condition == "evidence_gap" else np.zeros_like(gap)}
            arrays = paired_reference(source / entry["path"], entry["sha256"], fixed)
            prediction = BallGMM2D(**{f.name: torch.from_numpy(arrays[f.name])[None] for f in fields(BallGMM2D)})
            if prediction.means.shape[1] != record.frame_count:
                raise ValueError("Saved GMM frame axis differs")
            half = None
            if record.source == "meiji":
                half = "selection" if record.clip_id in partition["selection"] else "calibration"
            saved.append(SavedPrediction(record, condition, target, arrays, prediction, half, entry))
    return saved


def _check_hashes(hashes: dict[str, str]) -> None:
    for path, digest in hashes.items():
        if dual_sha256(Path(path)) != digest:
            raise ValueError(f"Calibration input changed: {path}")


def _table(metrics: dict[str, Any], output: Path) -> None:
    groups = ["meiji", "meiji/selection", "meiji/calibration",
              *[f"meiji/cam{i}" for i in range(3)],
              *[f"meiji/{half}/cam{i}" for half in ("selection", "calibration") for i in range(3)],
              "tracknet", "chat_annotation"]
    lines = ["# 共分散倍率の評価（before → after）", "",
             "calibration halfはclip単位OOF。selection half/TrackNet/chatはcalibration全6群fitの倍率。",
             "selection halfはcheckpoint選択済み。全Meiji行はこれらの合算で、独立test性能ではない。",
             "observed教師のみ。gap行は人工証拠欠損frameのみ。NLLはpx²密度、HDR面積はR²上のpx²。",
             "平均・weight・presenceは不変。その他の教師区分/存在NLLはmetrics.jsonを参照。", "",
             "| group/condition | n | NLL | HDR50 | area50 | HDR90 | area90 | HDR95 | area95 |",
             "|---|---:|---|---|---|---|---|---|---|"]
    columns = ["mean_nll_px", "coverage_0.5", "area_px2_0.5", "coverage_0.9", "area_px2_0.9", "coverage_0.95", "area_px2_0.95"]
    for group in groups:
        for condition in ("observed", "evidence_gap"):
            key = f"{group}/{condition}/observed"
            before, after = (metrics[f"{m}/{key}"] for m in ("before", "after"))
            pairs = [f"{before[c]:.5g} → {after[c]:.5g}" for c in columns]
            lines.append(f"| {group}/{condition} | {before['nll_px_frames']} | " + " | ".join(pairs) + " |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def run_covariance_calibration(
    source: Path, output: Path, calibration_path: Path, *,
    bounds: tuple[float, float], grid_points: int,
) -> Path:
    """Leave out whole calibration clips; export one full-fit scale for deployment.

    The bank adapter exports the full-fit scale on its fit frames and labels this
    reuse explicitly. It must never be used as the out-of-fold performance table.
    """
    if output.exists() or calibration_path.exists():
        raise FileExistsError("Calibration output and checkpoint sidecar must both be new")
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest["schema"] != "ball_refiner_cached_comparison.v1"
            or json.loads((source / "run_state.json").read_text())["status"] != "complete"):
        raise ValueError("Require a complete saved cached validation comparison")
    training = Path(manifest["training_run"])
    config = PilotConfig.from_config(OmegaConf.load(training / "config.yaml"))
    best = json.loads((training / "best.json").read_text())
    checkpoint = training / best["checkpoint"]
    if calibration_path.parent.resolve() != checkpoint.parent.resolve():
        raise ValueError("Calibration sidecar must be next to its exact checkpoint")
    hashes = {**manifest["input_sha256"],
              **{str(source / name): dual_sha256(source / name) for name in ("manifest.json", "run_state.json")},
              **{str(source / a["path"]): a["sha256"] for a in manifest["artifacts"]}}
    _check_hashes(hashes)
    if hashes[str(checkpoint)] != best["checkpoint_sha256"]:
        raise ValueError("Chosen checkpoint differs from saved predictions")
    saved = _load_saved(source, manifest, config)
    blocks = []
    for item in saved:
        if item.half == "calibration":
            selected = BallGMM2D(**{f.name: getattr(item.distribution, f.name)[:, item.scored] for f in fields(BallGMM2D)})
            blocks.append(density_terms(selected, torch.from_numpy(item.target.uv[item.scored])[None],
                                        group=item.group, condition=item.condition))
    full, folds = cross_validate_clips(blocks, bounds=bounds, grid_points=grid_points)
    output.mkdir(parents=True)
    write_json_atomic(output / "run_state.json", {"status": "evaluating"})
    fit = {"full": asdict(full), "folds": {g: asdict(f) for g, f in folds.items()},
           "bounds": bounds, "grid_points": grid_points,
           "objective": "equal temporal clips; observed/gap equal; pooled camera frames within group/condition; position NLL",
           "scope": "fit calibration half only; LOO whole clip/all cameras; selection half was used for checkpoint selection",
           "input_sha256": hashes}
    write_json_atomic(output / "fit.json", fit)
    artifact = {"schema": "ball_refiner_2d.covariance_calibration.v1",
                "covariance_multiplier": full.covariance_multiplier, "checkpoint_sha256": best["checkpoint_sha256"],
                "provenance": {"fit_groups": full.groups, "fit_report": str(output / "fit.json"),
                               "fit_report_sha256": dual_sha256(output / "fit.json"), "objective": fit["objective"],
                               "validation_manifest": str(source / "manifest.json"),
                               "validation_manifest_sha256": hashes[str(source / "manifest.json")],
                               "scope": "full calibration-half fit for deployment; OOF metrics use held-out clip scales"}}
    artifact_path = output / "covariance_calibration.json"
    write_json_atomic(artifact_path, artifact)
    artifact_hash = dual_sha256(artifact_path)
    full_calibration = load_covariance_calibration(artifact_path, expected_sha256=artifact_hash,
                                                 checkpoint_sha256=best["checkpoint_sha256"])
    print(json.dumps({"fit": asdict(full), "fold_multipliers": {k: v.covariance_multiplier for k, v in folds.items()}}), flush=True)
    rows: dict[str, list[dict[str, Array]]] = defaultdict(list)
    artifacts, bank_artifacts = [], []
    bank_source = output / "bank_input"
    bank_source.mkdir()
    (output / "predictions").mkdir()
    settings = {k: v for k, v in manifest["settings"].items() if k != "uniform_weight"}
    settings["levels"] = tuple(settings["levels"])
    for item in saved:
        multiplier = folds[item.group].covariance_multiplier if item.half == "calibration" else full.covariance_multiplier
        adjusted = CovarianceCalibration(multiplier, best["checkpoint_sha256"]).apply(item.distribution)
        for name in ("means", "mixture_logits", "presence_logits"):
            if not torch.equal(getattr(adjusted, name), getattr(item.distribution, name)):
                raise ValueError(f"Calibration changed {name}")
        values = gmm_rows(adjusted, item.target, (item.record.source_width, item.record.source_height),
                          **settings, clip_id=item.record.clip_id, condition=item.condition, device=torch.device("cpu"))
        for method, metrics in (("before", item.arrays), ("after", values)):
            for label, selected in strata(item.target, item.arrays["gap_mask"], item.condition).items():
                for group in item.groups:
                    rows[f"{method}/{group}/{item.condition}/{label}"].append({k: metrics[k][selected] for k in METRICS})
        path = output / "predictions" / item.entry["path"]
        with path.open("xb") as stream:
            np.savez_compressed(stream, **{**item.arrays, **values, "scale_tril": adjusted.scale_tril[0].numpy()})
        artifacts.append({**item.entry, "path": str(path.relative_to(output)), "sha256": dual_sha256(path),
                          "covariance_multiplier": multiplier,
                          "fit_excludes_this_clip": True, "half": item.half})
        if item.half == "calibration":
            deployed = full_calibration.apply(item.distribution)
            bank_path = bank_source / item.entry["path"]
            with bank_path.open("xb") as stream:
                np.savez_compressed(stream, **{f.name: getattr(deployed, f.name)[0].numpy() for f in fields(BallGMM2D)},
                                    target_uv=item.target.uv, position_valid=item.target.position_valid,
                                    scored_frame_index=item.target.frame_index[item.scored],
                                    frame_index=item.target.frame_index, pts=item.target.pts,
                                    timestamps_seconds=item.target.timestamps_seconds,
                                    segment_break=item.target.segment_break, gap_mask=item.arrays["gap_mask"])
            bank_artifacts.append({**item.entry, "sha256": dual_sha256(bank_path), "scored_frames": int(item.scored.sum()),
                                   "source_size_wh": [item.record.source_width, item.record.source_height]})
        write_json_atomic(output / "progress.json", {"artifacts": artifacts})
        print(json.dumps({"calibrated": item.record.clip_id, "condition": item.condition, "count": len(artifacts)}), flush=True)
    metrics = {key: summarize(parts, settings["levels"]) for key, parts in sorted(rows.items())}
    write_json_atomic(output / "metrics.json", metrics)
    _table(metrics, output)
    _check_hashes(hashes)
    write_json_atomic(bank_source / "manifest.json", {
        "schema": "ball_refiner_validation_diagnostics.v1", "partition": "calibration",
        "clip_ids": manifest["partition"]["calibration"], "artifacts": bank_artifacts,
        "checkpoint": best, "input_sha256": {**hashes, str(calibration_path): artifact_hash},
        "scope": "Full six-clip-fit multiplier on its fit frames, for empirical residual bootstrap only; NOT OOF scores",
        "covariance_multiplier": full.covariance_multiplier,
    })
    write_json_atomic(bank_source / "run_state.json", {"status": "complete"})
    write_json_atomic(output / "manifest.json", {"schema": "ball_refiner_2d.covariance_evaluation.v1",
                                               "input_sha256": hashes, "artifacts": artifacts,
                                               "calibration_path": str(calibration_path), "calibration_sha256": artifact_hash,
                                               "settings": settings, "scope": fit["scope"]})
    with calibration_path.open("xb") as stream:
        stream.write(artifact_path.read_bytes())
    load_covariance_calibration(calibration_path, expected_sha256=artifact_hash,
                                checkpoint_sha256=best["checkpoint_sha256"])
    write_json_atomic(output / "run_state.json", {"status": "complete", "artifacts": len(artifacts),
                                                "bank_input_artifacts": len(bank_artifacts)})
    return output
