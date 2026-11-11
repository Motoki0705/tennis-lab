"""Aggregate the fixed untrained benchmark; no validation accuracy is inferred."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "tests/benchmarks/ball_detection"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from efficiency import quantiles
from protocol import ROOT, read, write
from serial import index_store, snapshot_store

from src.utils.artifact_store import RcloneSnapshotStore


def aggregate():
    root = ROOT / "benchmarks"
    order = read(root / "order.json")
    if len(order) != 3 or any(set(block) != set(order[0]) for block in order):
        raise RuntimeError("Three complete reordered blocks are required")
    summaries = {}
    profiler = {}
    for key in order[0]:
        runs = [read(root / f"block-{block}" / (key + ".json")) for block in range(3)]
        if any(
            run["kind"] != "untrained benchmark"
            or run["warmup"] < 10
            or run["repetitions"] != 30
            for run in runs
        ):
            raise RuntimeError("Measurement protocol differs")
        inference = [row for run in runs for row in run["raw"]["inference"]]
        training = [row for run in runs for row in run["raw"]["training"]]
        frames = runs[0]["frames_per_inference"]
        summary = dict(
            spec=runs[0]["spec"],
            samples_per_mode=len(inference),
            frames_per_inference=frames,
            frames_per_optimizer_update=runs[0]["frames_per_optimizer_update"],
            inference_cuda_ms=quantiles(row["cuda_ms"] for row in inference),
            inference_wall_ms=quantiles(row["wall_ms"] for row in inference),
            inference_ms_per_frame=quantiles(
                row["cuda_ms"] / frames for row in inference
            ),
            update_cuda_ms=quantiles(row["total"]["cuda_ms"] for row in training),
            update_wall_ms=quantiles(row["total"]["wall_ms"] for row in training),
            phases={
                name: {
                    clock: quantiles(row["phases"][name][clock] for row in training)
                    for clock in ("cuda_ms", "wall_ms")
                }
                for name in training[0]["phases"]
            },
            backward_forward_ratio=quantiles(
                row["backward_forward_ratio"] for row in training
            ),
            peak_allocated=max(row["peak_allocated"] for row in training),
            peak_reserved=max(row["peak_reserved"] for row in training),
            inference_peak_allocated=max(
                run["inference_peak"]["allocated"] for run in runs
            ),
            inference_peak_reserved=max(
                run["inference_peak"]["reserved"] for run in runs
            ),
            parameter_bytes=training[-1]["parameter_bytes"],
            gradient_bytes=training[-1]["gradient_bytes"],
            optimizer_bytes=training[-1]["optimizer_bytes"],
            parameters=runs[0]["parameters"],
            trainable_parameters=runs[0]["trainable_parameters"],
            macs=runs[0]["macs"],
            compilation=[run["compilation"] for run in runs],
            block_inference_p50=[
                run["summary"]["inference_cuda_ms"]["p50"] for run in runs
            ],
            block_update_p50=[run["summary"]["update_cuda_ms"]["p50"] for run in runs],
            cold_start=[run["cold_start"] for run in runs],
        )
        summaries[key] = summary
        operators = read(root / "profiles" / (key + "-ops.json"))
        profiler[key] = [
            row
            for row in operators
            if any(
                token in row["name"].lower()
                for token in (
                    "cast",
                    "to_copy",
                    "copy_",
                    "permute",
                    "contiguous",
                    "deform",
                    "benchmark_",
                )
            )
        ]
    for key, item in summaries.items():
        case = item["spec"]["case"]
        if key.startswith("full-") or key.startswith("encoder-"):
            baseline = ("full-" if key.startswith("full-") else "encoder-") + (
                "CA" if case in ("CA", "T1", "T2") else "C0"
            )
            ref = summaries[baseline]
            item["baseline"] = baseline
            item["relative_to_baseline"] = {
                "inference_latency": item["inference_cuda_ms"]["p50"]
                / ref["inference_cuda_ms"]["p50"],
                "update_latency": item["update_cuda_ms"]["p50"]
                / ref["update_cuda_ms"]["p50"],
                "allocated_vram": item["peak_allocated"] / ref["peak_allocated"],
                "counted_macs": item["macs"]["conv_linear_attention_macs"]
                / ref["macs"]["conv_linear_attention_macs"],
            }
    output = dict(
        kind="untrained benchmark",
        accuracy=None,
        cases=summaries,
        scope="L4, fixed shapes, BF16 + official MSDA FP32, compile; synthetic resident input; phase-synchronized wall time",
        limitations=[
            "N=90 samples in three blocks on one GPU; samples are not independent machines",
            "Isolated component MSE gradients do not quantify the exact end-to-end runtime share",
            "MAC terms exclude normalization, activation, MDD, mHC mixing, memory traffic and backward; MSDA sampling bound is separate",
            "No accuracy/Pareto adoption decision before the scientific training validation",
        ],
    )
    write(root / "summary.json", output)
    write(root / "profiler-boundary-ops.json", profiler)
    columns = [
        "case",
        "MAC_G_per_frame",
        "infer_p50_ms",
        "infer_p95_ms",
        "update_p50_ms",
        "update_p95_ms",
        "backward_forward",
        "allocated_GiB",
        "reserved_GiB",
    ]
    rows = ["\t".join(columns)]
    for case in ("C0", "C1", "C2", "C3", "CA", "T1", "T2"):
        x = summaries["full-" + case]
        values = [
            case,
            x["macs"]["conv_linear_attention_macs_per_frame"] / 1e9,
            x["inference_cuda_ms"]["p50"],
            x["inference_cuda_ms"]["p95"],
            x["update_cuda_ms"]["p50"],
            x["update_cuda_ms"]["p95"],
            x["backward_forward_ratio"]["p50"],
            x["peak_allocated"] / 2**30,
            x["peak_reserved"] / 2**30,
        ]
        rows.append(
            "\t".join(str(v) if isinstance(v, str) else f"{v:.6f}" for v in values)
        )
    (root / "full-models.tsv").write_text("\n".join(rows) + "\n")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    names = ("C0", "C1", "C2", "C3", "CA", "T1", "T2")
    for case in names:
        x = summaries["full-" + case]
        axes[0].scatter(
            x["macs"]["conv_linear_attention_macs_per_frame"] / 1e9,
            x["inference_ms_per_frame"]["p50"],
        )
        axes[0].annotate(
            case,
            (
                x["macs"]["conv_linear_attention_macs_per_frame"] / 1e9,
                x["inference_ms_per_frame"]["p50"],
            ),
            xytext=(3, 3),
            textcoords="offset points",
        )
        axes[1].scatter(x["peak_allocated"] / 2**30, x["update_cuda_ms"]["p50"] / 1000)
        axes[1].annotate(
            case,
            (x["peak_allocated"] / 2**30, x["update_cuda_ms"]["p50"] / 1000),
            xytext=(3, 3),
            textcoords="offset points",
        )
    axes[0].set(
        xlabel="Counted forward GMAC/frame",
        ylabel="Inference ms/frame",
        title="MAC and inference latency",
    )
    axes[1].set(
        xlabel="Peak allocated GiB",
        ylabel="Seconds / 128-frame update",
        title="Training latency and memory",
    )
    bottom = np.zeros(len(names))
    for phase in ("forward", "loss", "backward", "gradient_clip", "optimizer"):
        values = np.array(
            [
                summaries["full-" + c]["phases"][phase]["cuda_ms"]["p50"] / 1000
                for c in names
            ]
        )
        axes[2].bar(names, values, bottom=bottom, label=phase)
        bottom += values
    axes[2].set(ylabel="Seconds / 128-frame update", title="Median synchronized phases")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.2)
    figure.suptitle(
        "Untrained benchmark — NVIDIA L4 / fixed shapes / no accuracy result"
    )
    figure.savefig(root / "efficiency.png", dpi=140)
    plt.close(figure)
    return output


def publish():
    import tempfile
    import time

    output = ROOT / "benchmarks"
    store = snapshot_store(output)
    receipt = store.publish("summary-" + str(time.time_ns()))
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        restored = RcloneSnapshotStore(
            Path(temporary) / "verified",
            remote="mounted_drive",
            remote_root=store.remote_root,
            config_path=store.config_path,
        )
        readback = restored.restore(
            receipt["generation"], manifest_sha256=receipt["manifest_sha256"]
        )
    write(
        ROOT / "benchmark-summary-receipt.json",
        dict(receipt=receipt, readback=readback),
    )
    index_store().publish_file(ROOT / "benchmark-summary-receipt.json")


if __name__ == "__main__":
    aggregate()
    if "--publish" in sys.argv:
        publish()
