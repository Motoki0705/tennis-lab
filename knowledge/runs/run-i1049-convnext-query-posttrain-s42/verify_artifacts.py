"""CPU-only verification of the completed run; takes launch JSON and report path."""

import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from PIL import Image

launch = json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0, launch["code_root"])

# Load the frozen training implementation specified by the launch record.
from src.tasks.ball_detection.models.mdd_pretrain import (  # noqa: E402
    DeepMDDQueryDetector,
    MDDPretrainConfig,
)
from src.tasks.ball_detection.training.posttraining.checkpoint import (  # noqa: E402
    completed_pretraining,
)
from src.utils.checksum import dual_sha256  # noqa: E402

torch.set_num_threads(2)
run = Path(launch["output"])


def read(name):
    return json.loads((run / name).read_text())


config, best, complete, stress = [
    read(n) for n in ("config.json", "best.json", "COMPLETED.json", "stress.json")
]
metrics = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
train = [json.loads(s) for s in (run / "train.jsonl").read_text().splitlines()]
queue = Path(launch["queue_directory"])
assert (queue / "done" / launch["job"]).is_file(), "Queue not done"
assert all(
    not (queue / state / launch["job"]).exists()
    for state in ("running", "jobs", "failed", "cancelled")
)
assert complete["stage"] == "query_posttraining" and complete["global_step"] == 60000
assert [r["epoch"] for r in metrics] == list(range(10))
assert [r["global_step"] for r in metrics] == list(range(6000, 60001, 6000))
assert [r["global_step"] for r in train] == list(range(50, 60001, 50))
assert all(r["encoder_frozen"] == (r["epoch"] == 0) for r in train)
assert all(
    r["encoder_learning_rate"] == 0
    if r["epoch"] == 0
    else math.isclose(r["encoder_learning_rate"], r["learning_rate"] * 0.1)
    for r in train
)
assert all(
    math.isfinite(r[k])
    for r in train
    for k in ("train_loss", "grad_norm", "learning_rate", "encoder_learning_rate")
)
selected = min(metrics, key=lambda r: r["scopes"]["common"]["macro_mean_error_px"])
assert best["epoch"] == complete["best_epoch"] == selected["epoch"] == 9
assert best["global_step"] == selected["global_step"] == 60000
assert (
    best["selection_error_px"]
    == selected["scopes"]["common"]["macro_mean_error_px"]
    == complete["best_error_px"]
)
assert best["scopes"] == selected["scopes"]
recipe = config["recipe"]
assert recipe["runtime"]["precision"] == "bf16" and recipe["test_usage"] == "none"
assert (recipe["epochs"], recipe["freeze_epochs"], recipe["windows_per_epoch"]) == (
    10,
    1,
    6000,
)
assert recipe["selection_scope"] == "common" and recipe["selection_profile"] == "clean"
assert recipe["model"]["encoder_variant"] == "convnext_v2"
assert tuple(recipe["model"][k] for k in ("dim", "layers", "heads", "ffn_dim")) == (
    256,
    4,
    8,
    704,
)
assert (
    recipe["manifest_sha256"]
    == dual_sha256(launch["manifest"])
    == dual_sha256(run / "data_manifest.json")
)
assert config["code"]["base_commit"] == launch["code_commit"]
for filename, digest in config["code"]["source_sha256"].items():
    assert dual_sha256(Path(launch["code_root"]) / filename) == digest, filename
parent_path, parent = completed_pretraining(
    Path(launch["pretraining_run"]), Path(launch["manifest"])
)
assert (
    recipe["parent"]["sha256"]
    == launch["pretrained_sha256"]
    == complete["pretrained_checkpoint"]
)
assert (
    str(parent_path) == launch["pretrained_checkpoint"]
    and parent["global_step"] == 48000
)
best_hash = dual_sha256(run / best["checkpoint"])
assert best_hash == best["sha256"]
saved = torch.load(run / best["checkpoint"], map_location="cpu", weights_only=True)
assert (
    saved["schema"] == "mdd_query_posttraining.v1"
    and saved["stage"] == "query_posttraining"
)
assert saved["epoch"] == best["epoch"] and saved["global_step"] == 60000
assert (
    saved["best_epoch"] == best["epoch"]
    and saved["best_score"] == best["selection_error_px"]
)
# torch keeps tuples; JSON represents the same configuration as lists.
assert json.loads(json.dumps(saved["recipe"])) == config["recipe"]
assert (
    saved["code"] == config["code"] and saved["validation"]["scopes"] == best["scopes"]
)
counts = Counter()


def finite(value, location):
    if isinstance(value, torch.Tensor):
        counts[location.split(".")[0]] += 1
        assert not value.is_floating_point() or torch.isfinite(value).all().item(), (
            location
        )
    elif isinstance(value, dict):
        for k, v in value.items():
            finite(v, f"{location}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            finite(v, f"{location}.{i}")
    elif isinstance(value, float):
        assert math.isfinite(value), location


finite(saved["state_dict"], "model")
finite(saved["optimizer"], "optimizer")
model = DeepMDDQueryDetector(MDDPretrainConfig(**saved["model_config"]))
model.load_state_dict(saved["state_dict"], strict=True)
assert not any(k.startswith("decoder.") for k in saved["state_dict"])
assert set(stress) == {"camera", "occlusion", "combined"}
finite(stress, "stress")
for profile, report in {"clean": best, **stress}.items():
    assert report["profile"] == profile and report["precision"] == "bf16"
    assert set(report["scopes"]) == {"common", "full"}
    for entry in report["scopes"].values():
        fps = entry["by_frame_step"]
        assert set(fps) == {"1", "2", "4"}
        assert all(x["available"] and x["observed_frames"] > 0 for x in fps.values())
        assert math.isclose(
            entry["macro_mean_error_px"],
            sum(x["mean_error_px"] for x in fps.values()) / 3,
        )
    if profile in ("occlusion", "combined"):
        assert report["artificially_occluded"]["scopes"]["full"]["frame_fps_pairs"] > 0
previews = []
for path in sorted(run.rglob("*.gif")):
    with Image.open(path) as im:
        assert im.n_frames == 32, (path, im.n_frames)
        size = im.size
        for index in range(im.n_frames):
            im.seek(index)
            im.load()
        previews.append(
            dict(
                path=str(path.relative_to(run)),
                frames=im.n_frames,
                size=list(size),
                sha256=dual_sha256(path),
            )
        )
assert len(previews) == 49
assert sum(p["path"].startswith("previews/train-epoch-") for p in previews) == 10
assert sum(p["path"].startswith("previews/val-epoch-") for p in previews) == 30
assert all(
    sum(p["path"].startswith(f"previews/{profile}/") for p in previews) == 3
    for profile in stress
)
report = dict(
    schema="posttraining_completion_verification.v1",
    at=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
    status="passed",
    job=launch["job"],
    queue_state="done",
    run=str(run),
    code_commit=launch["code_commit"],
    verified_source_files=len(config["code"]["source_sha256"]),
    manifest_sha256=recipe["manifest_sha256"],
    parent_checkpoint_sha256=launch["pretrained_sha256"],
    best_checkpoint=best["checkpoint"],
    best_checkpoint_sha256=best_hash,
    best_epoch=best["epoch"],
    global_step=60000,
    validation_epochs=len(metrics),
    train_records=len(train),
    checkpoint_count=len(list(run.glob("epoch-*.pt"))),
    frozen_train_records=sum(r["encoder_frozen"] for r in train),
    joint_train_records=sum(not r["encoder_frozen"] for r in train),
    finite_tensors=dict(counts),
    strict_cpu_model_load=True,
    no_gpu_execution=True,
    test_usage=recipe["test_usage"],
    metrics={
        p: {s: r["macro_mean_error_px"] for s, r in d["scopes"].items()}
        for p, d in {"clean": best, **stress}.items()
    },
    previews=previews,
    artifact_sha256={
        n: dual_sha256(run / n)
        for n in (
            "COMPLETED.json",
            "best.json",
            "config.json",
            "metrics.jsonl",
            "stress.json",
            "train.jsonl",
        )
    },
)
Path(sys.argv[2]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in report.items() if k != "previews"},
        ensure_ascii=False,
        indent=2,
    )
)
