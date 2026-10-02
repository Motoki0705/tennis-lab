"""CPU-only audit of the interrupted r16 run; never performs model inference."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("/home/kamimura/projects/tennis-lab")
RUN = ROOT / "outputs/ball_detection/train/i935_mixed_ft/s42-r16-20260929"
BUNDLE = Path(__file__).resolve().parent
JOB = "1790691583774854570_1206594_i935-mixed-ft-val-recall-s42-r16-20260929"
GROUPS = ("all", "chat_annotation", "meiji", "meiji/cam0", "meiji/cam1", "meiji/cam2", "tracknet")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(name: str, value: Any) -> None:
    (BUNDLE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    torch.set_num_threads(4)
    plt.switch_backend("Agg")
    inputs: list[dict[str, Any]] = []
    sources = [
        (RUN / "config.yaml", "resolved-config.yaml"),
        (RUN / "resource_usage.json", "resource_usage.json"),
        (ROOT / f".training_queue/logs/{JOB}.log", "queue.log"),
        (ROOT / f".training_queue/failed/{JOB}.job", "failed.job"),
    ]
    events = sorted((RUN / "logs/version_0").glob("events.out.tfevents.*"))
    assert len(events) == 1, events
    sources.append((events[0], events[0].name))
    for source, name in sources:
        before = digest(source)
        shutil.copyfile(source, BUNDLE / name)
        assert before == digest(source) == digest(BUNDLE / name)
        inputs.append({"path": str(source), "bundle_file": name, "sha256": before, "bytes": source.stat().st_size})
    write_json("source_artifacts.json", inputs)

    events_data = EventAccumulator(str(BUNDLE / events[0].name), size_guidance={"scalars": 0}).Reload()
    scalars = {
        tag: [{"step": e.step, "value": e.value, "wall_time": e.wall_time} for e in events_data.Scalars(tag)]
        for tag in events_data.Tags()["scalars"]
    }
    write_json("tensorboard_scalars.json", scalars)
    epoch_by_step: dict[int, int] = {}
    for event in scalars["epoch"]:
        step, epoch = int(event["step"]), int(event["value"])
        assert event["value"] == epoch
        if step in epoch_by_step:
            assert epoch_by_step[step] == epoch
        epoch_by_step[step] = epoch
    validation: dict[int, dict[str, float]] = {}
    for tag, samples in scalars.items():
        if not tag.startswith("val/"):
            continue
        assert len(samples) == 10, (tag, len(samples))
        for sample in samples:
            step = int(sample["step"])
            epoch = epoch_by_step[step]
            assert 0 <= epoch <= 9
            values = validation.setdefault(epoch, {"global_step": float(step)})
            assert tag not in values, (epoch, tag)
            values[tag] = sample["value"]
    assert sorted(validation) == list(range(10))
    rows: list[dict[str, Any]] = []
    expected_observed = {"all": 31167, "chat_annotation": 6622, "meiji": 23007,
                         "meiji/cam0": 7556, "meiji/cam1": 7999, "meiji/cam2": 7452, "tracknet": 1538}
    for epoch, values in sorted(validation.items()):
        for group in GROUPS:
            prefix = "val/" + ("" if group == "all" else group + "/") + "candidate_"
            row: dict[str, Any] = {"epoch": epoch, "group": group, "global_step": int(values["global_step"])}
            for name in ("frames", "observed", "recalled_at_1", "recalled_at_k", "not_in_candidates",
                         "wrong_ranked_above_true", "wrong_strictly_higher_score", "rank_only_due_to_tie"):
                value = values[prefix + name]
                assert value == int(value), (epoch, group, name)
                row[name] = int(value)
            assert row["observed"] == expected_observed[group]
            assert row["not_in_candidates"] + row["recalled_at_k"] == row["observed"]
            assert row["wrong_ranked_above_true"] + row["recalled_at_1"] == row["recalled_at_k"]
            for name, count in (("recall_at_8_20px", "recalled_at_k"), ("recall_at_1_20px", "recalled_at_1"),
                                ("not_in_candidates_rate", "not_in_candidates"),
                                ("wrong_ranked_above_true_rate", "wrong_ranked_above_true"),
                                ("wrong_strictly_higher_score_rate", "wrong_strictly_higher_score")):
                row[name] = row[count] / row["observed"]
                row[name + "_logged"] = values[prefix + name]
                assert math.isclose(row[name], row[name + "_logged"], abs_tol=3e-8)
            row["val_f1_all_sources"] = values["val/f1"]
            rows.append(row)
    for epoch in range(10):
        groups = {row["group"]: row for row in rows if row["epoch"] == epoch}
        for name in ("frames", "observed", "recalled_at_1", "recalled_at_k", "not_in_candidates",
                     "wrong_ranked_above_true", "wrong_strictly_higher_score", "rank_only_due_to_tie"):
            assert groups["all"][name] == sum(groups[g][name] for g in ("chat_annotation", "meiji", "tracknet"))
            assert groups["meiji"][name] == sum(groups[f"meiji/cam{i}"][name] for i in range(3))
    write_json("per_epoch.json", rows)
    with (BUNDLE / "per_epoch.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    checkpoints = []
    checkpoint_paths = sorted((RUN / "logs/version_0/checkpoints").glob("*.ckpt"))
    assert len(checkpoint_paths) == 11
    for path in checkpoint_paths:
        checksum = digest(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        epoch = checkpoint["epoch"]
        assert 0 <= epoch <= 9
        if path.name != "last.ckpt":
            assert path.name == f"ball-detection-epoch={epoch:02d}.ckpt"
        callbacks = [v for k, v in checkpoint["callbacks"].items()
                     if k.startswith("ModelCheckpoint") and v["monitor"] is not None]
        assert len(callbacks) == 1
        callback = callbacks[0]
        assert callback["monitor"] == "val/meiji/candidate_recall_at_8_20px"
        assert checkpoint["global_step"] == 1920 * (epoch + 1)
        checkpoints.append({
            "path": str(path), "sha256": checksum, "bytes": path.stat().st_size,
            "epoch": epoch, "global_step": checkpoint["global_step"],
            "monitor": callback["monitor"], "best_model_path": callback["best_model_path"],
            "best_model_score": float(callback["best_model_score"]),
        })
        assert checksum == digest(path)
        del checkpoint
    write_json("checkpoints.json", checkpoints)
    selected = max((r for r in rows if r["group"] == "meiji"), key=lambda r: (r["recall_at_8_20px"], -r["epoch"]))
    assert selected["epoch"] == 9
    selected_checkpoint = next(c for c in checkpoints if c["path"].endswith("ball-detection-epoch=09.ckpt"))
    assert selected_checkpoint["path"] == selected_checkpoint["best_model_path"]
    assert math.isclose(selected_checkpoint["best_model_score"], selected["recall_at_8_20px"], abs_tol=3e-8)
    write_json("selection.json", {
        "queue_job": JOB, "status": "selected_from_completed_epochs_of_failed_run",
        "rule": "maximum Meiji val recall@8 within 20 source px; exact tie chooses earlier epoch",
        "completed_epochs": list(range(10)), "last_logged_training_epoch": max(epoch_by_step.values()),
        "checkpoint": selected_checkpoint, "meiji_validation": selected,
        "test_used_for_selection": False, "resume_or_extension": False,
        "selection_precision": "bf16-mixed Lightning validation; cache inference is float32",
    })
    lines = ["# r16: 全epochのvalidation候補指標", "",
             "K=8 / NMS=5 / patch=5 / subpixel / T=8 / stride=4 / 距離≤20 source px。",
             "bf16-mixedでのLightning validation。率はobserved分母の整数件数から再計算。",
             "allは3 source合算。Meiji camera以外はcamera IDがなく、camera別値を作らない。",
             "誤候補上位は正解候補が存在しtop-1が誤りのframe。strict-score版・同点件数はJSON/CSVに保存。", "",
             "| epoch | source / camera | observed | hit@8 | hit@1 | recall@8 | recall@1 | 候補外 | 誤候補上位 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['epoch']} | {row['group']} | {row['observed']} | {row['recalled_at_k']} | "
                     f"{row['recalled_at_1']} | {row['recall_at_8_20px']:.6f} | {row['recall_at_1_20px']:.6f} | "
                     f"{row['not_in_candidates_rate']:.6f} | {row['wrong_ranked_above_true_rate']:.6f} |")
    lines += ["", "## 従来の閾値F1（全source validation）", "", "| epoch | val F1 |", "|---:|---:|"]
    lines += [f"| {epoch} | {values['val/f1']:.6f} |" for epoch, values in sorted(validation.items())]
    (BUNDLE / "per_epoch.md").write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for group in GROUPS:
        series = [r for r in rows if r["group"] == group]
        epochs = [r["epoch"] for r in series]
        axis = axes[0, 1] if group.startswith("meiji/") else axes[0, 0]
        axis.plot(epochs, [r["recall_at_8_20px"] for r in series], marker="o", label=group)
        axes[1, 0].plot(epochs, [r["recall_at_1_20px"] for r in series], marker="o", label=group)
    axes[1, 1].plot(sorted(validation), [validation[e]["val/f1"] for e in sorted(validation)], marker="o", label="all-source F1")
    for axis, title in zip(axes.flat, ("Source recall@8", "Meiji camera recall@8", "Recall@1", "Threshold F1 (not selection)"), strict=True):
        axis.set(title=title, xlabel="Completed epoch", ylabel="Rate", xticks=list(range(10)))
        axis.axvline(9, color="black", alpha=0.3, linestyle="--")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    figure.suptitle("r16: selected epoch 9; CUDA failure during epoch 10; no test selection")
    figure.savefig(BUNDLE / "per_epoch.png", dpi=150)
    plt.close(figure)
    print(json.dumps({"rows": len(rows), "selected": selected_checkpoint, "meiji": selected}, indent=2))


if __name__ == "__main__":
    main()
