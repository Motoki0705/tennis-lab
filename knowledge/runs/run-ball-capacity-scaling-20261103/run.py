"""Reproducible #1675 campaign. GPU entry points must use the shared queue."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from torch.utils.data import DataLoader

from src.tasks.ball_detection.checkpoints.provenance import source_identity
from src.tasks.ball_detection.checkpoints.temporal import save_json
from src.tasks.ball_detection.data.augmentation.video import (
    AugmentationConfig,
    VideoAugmenter,
)
from src.tasks.ball_detection.data.windows.dataset import (
    FrozenBallPackDataset,
    collate_pack_samples,
)
from src.tasks.ball_detection.model_io.mdd_temporal import (
    build_temporal_model,
    temporal_forward,
)
from src.tasks.ball_detection.models.mdd.config import (
    MDDHeatmapConfig,
    MDDMHCQueryConfig,
)
from src.tasks.ball_detection.training.runtime.compilation import (
    compile_coordinate_model,
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.runtime.execution import (
    coordinate_worker_init,
)
from src.tasks.ball_detection.training.runtime.images import (
    coordinate_batches,
    image_input,
)
from src.tasks.ball_detection.training.temporal.packs import (
    PackSampler,
    augment_pack,
    microbatch,
    supervised_loss_sum,
)
from src.tasks.ball_detection.training.temporal.runner import train_temporal_model
from src.utils.artifact_store import RcloneArtifactStore, RcloneSnapshotStore
from src.utils.checksum import dual_sha256
from src.utils.models.components.ffn_layers import default_ffn_dim

PRIOR = Path("knowledge/runs/run-ball-sequential-length-20261025/run.py")
spec = importlib.util.spec_from_file_location("sequential_recipe", PRIOR)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
CONDITIONS = [(256, 4), (384, 4), (512, 4), (768, 4), (1024, 4), (512, 6), (512, 8)]
PREPARED = Path(
    "outputs/ball_detection/precompute/sequential_length/observed-packs-20261025"
)
DOWNLOAD = Path(
    "outputs/ball_detection/precompute/sequential_length/drive-download-20261025"
)


def configuration(architecture, dim=256, layers=4):
    if architecture == "dpt":
        return replace(
            MDDHeatmapConfig.load(
                Path("src/tasks/ball_detection/configs/model/mdd_dpt_heatmap.yaml")
            ),
            stem_channels=(16, 32, 64, 128),
            mixed_channels=(256, 512),
            residual_blocks=(0, 1, 2, 3, 3, 2),
            encoder_variant="x3d_csn",
            temporal_mixing="depthwise3d",
            heads=4,
            frames=32,
            frame_chunk_size=None,
            activation_checkpointing=False,
        )
    base = MDDMHCQueryConfig.load(
        Path("src/tasks/ball_detection/configs/model/mdd_mhc_query.yaml")
    )
    return replace(
        base,
        frames=64,
        dim=dim,
        layers=layers,
        heads=dim // 64,
        ffn_dim=default_ffn_dim(dim),
        frame_chunk_size=None,
        activation_checkpointing=False,
    )


def budget(architecture, microbatch=1):
    # The first post-common stage of #1514, irrespective of our new T64 input.
    # Keep its LR horizon=12000 and offset=0; no second T32/T128 stage is added.
    return prior.budget(
        architecture=architecture,
        frames=32,
        common=architecture == "dpt",
        microbatch=microbatch,
    )


def runtime():
    return prior.runtime()


def inputs(args):
    receipt = json.loads((args.workspace_root / DOWNLOAD / "progress.json").read_text())
    if not receipt["complete"] or any(
        receipt["verified_" + u] != receipt["expected_" + u] for u in ("files", "bytes")
    ):
        raise ValueError(
            "Full downloaded file membership, size and dual SHA must pass first"
        )
    prepared = args.workspace_root / PREPARED
    audit = json.loads((prepared / "audit.json").read_text())
    for t in ("32", "64"):
        if audit["audit"][t]["accepted_packs"] != 30478:
            raise ValueError("The fixed observed-pack training set changed")
    return prepared, receipt


def archive(args, root, names):
    store = RcloneArtifactStore(
        root,
        remote="mounted_drive",
        remote_root="outputs/ball_detection/train/capacity_scaling/" + args.run_id,
        config_path=args.rclone_config,
    )
    for name in names:
        store.publish_file(root / name)


def benchmark_case(args, architecture, dim, layers, microbatch_windows):
    prepared, _ = inputs(args)
    cfg = configuration(architecture, dim, layers)
    rt = runtime()
    rt.configure(torch.device("cuda"))
    torch.manual_seed(42)
    model = build_temporal_model(cfg).cuda()
    model.train()
    compile_coordinate_model(
        model, mode=rt.compile_mode, recompile_limit=rt.compile_recompile_limit
    )
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": enc}, {"params": head}], lr=1e-4, weight_decay=0.01
    )
    packs = FrozenBallPackDataset(
        prepared / f"train_T{cfg.frames}.json",
        split="train",
        jpeg_decoder=rt.jpeg_decoder,
    )
    sampler = PackSampler(
        len(packs),
        seed=42,
        position=0 if architecture == "dpt" else 10000,
        count=args.benchmark_updates + 1,
    )
    loader = DataLoader(
        packs,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_pack_samples,
        num_workers=rt.num_workers,
        pin_memory=True,
        prefetch_factor=rt.prefetch_factor,
        worker_init_fn=coordinate_worker_init,
        generator=torch.Generator().manual_seed(42),
    )
    iterator = iter(coordinate_batches(loader, torch.device("cuda"), prefetch=True))
    augmenter = VideoAugmenter(
        AugmentationConfig.load(
            Path("src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml")
        ),
        42,
    )
    records = []
    # Query: additionally compile and check the frozen CNN path before joint training.
    phases = ([True] if architecture == "query" else []) + [
        False
    ] * args.benchmark_updates
    try:
        for i, frozen in enumerate(phases):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            model.train()
            if architecture == "query":
                model.freeze_encoder(frozen)
            waiting = time.perf_counter()
            batch = next(iterator)
            rgb = image_input(batch, torch.device("cuda"))
            data_wait_seconds = time.perf_counter() - waiting
            rgb, batch, _ = augment_pack(
                batch,
                rgb,
                augmenter,
                position=(0 if architecture == "dpt" else 10000) + i,
            )
            optimizer.zero_grad(set_to_none=True)
            teachers = 0
            loss_value = 0.0
            with coordinate_compile_scope(model):
                for first in range(0, rgb.shape[0], microbatch_windows):
                    last = min(first + microbatch_windows, rgb.shape[0])
                    part = microbatch(batch, first, last)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = temporal_forward(
                            model, rgb[first:last], part["timestamps"].cuda()
                        )
                        loss, n = supervised_loss_sum(
                            prediction,
                            part,
                            sigma_ratio=0.012,
                            gamma=2.0,
                            coordinates=architecture == "query",
                        )
                    loss.backward()
                    teachers += n
                    loss_value += float(loss.detach())
                if teachers < 1:
                    raise ValueError("Benchmark update lacks teachers")
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(teachers)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
            torch.cuda.synchronize()
            record = dict(
                index=i,
                frozen=frozen,
                seconds=time.perf_counter() - started,
                data_wait_seconds=data_wait_seconds,
                input_frames=rgb.shape[0] * rgb.shape[1],
                valid_teachers=teachers,
                loss=loss_value / teachers,
                gradient_norm=float(norm),
                peak_vram_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),
            )
            if record["input_frames"] != 128:
                raise ValueError("Benchmark must use complete 128-frame update")
            records.append(record)
            print(
                json.dumps(
                    dict(
                        event="benchmark_update",
                        architecture=architecture,
                        dim=dim,
                        layers=layers,
                        microbatch_windows=microbatch_windows,
                        **record,
                    )
                ),
                flush=True,
            )
    finally:
        iterator.close()
    joint = [r for r in records if not r["frozen"]]
    steady = joint[5:]
    if not steady:
        raise ValueError("Benchmark requires >5 measured joint updates")
    return dict(
        architecture=architecture,
        dim=dim,
        layers=layers,
        frames=cfg.frames,
        microbatch_windows=microbatch_windows,
        accumulation_microbatches=128 // cfg.frames // microbatch_windows,
        success=True,
        model_config=asdict(cfg),
        parameters=sum(p.numel() for p in model.parameters()),
        median_update_seconds=statistics.median(r["seconds"] for r in steady),
        mean_update_seconds=statistics.mean(r["seconds"] for r in steady),
        peak_vram_bytes=max(r["peak_vram_bytes"] for r in records),
        peak_reserved_vram_bytes=max(r["peak_reserved_vram_bytes"] for r in records),
        records=records,
        runtime=asdict(rt),
        scope="Real frozen dataset packs, augmentation, BF16 and compile; 128 frames/update. Independent random diagnostic weights. Steady timing excludes first five joint updates, includes per-update input wait, augmentation, forward/backward and optimizer; no convergence claim.",
    )


def benchmark(args):
    root = (
        args.workspace_root
        / "outputs/ball_detection/train/capacity_scaling"
        / args.run_id
    )
    root.mkdir(parents=True, exist_ok=False)
    cases = [("dpt", 256, 4, 1), ("dpt", 256, 4, 2)] + [
        ("query", d, depth, 1) for d, depth in CONDITIONS
    ]
    results = []
    for architecture, dim, layers, micro in cases:
        name = f"{architecture}-d{dim}-l{layers}-b{micro}"
        started = time.perf_counter()
        try:
            result = benchmark_case(args, architecture, dim, layers, micro)
        except torch.OutOfMemoryError:
            result = dict(
                architecture=architecture,
                dim=dim,
                layers=layers,
                microbatch_windows=micro,
                success=False,
                error="CUDA_OOM",
                traceback=traceback.format_exc(),
                memory_controls_changed=False,
            )
        except Exception:
            save_json(
                root / (name + ".failed.json"), dict(error=traceback.format_exc())
            )
            raise
        result["wall_seconds"] = time.perf_counter() - started
        results.append(result)
        save_json(root / (name + ".json"), result)
        archive(args, root, [name + ".json"])
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    dpt = [r for r in results if r["architecture"] == "dpt" and r["success"]]
    queries = [r for r in results if r["architecture"] == "query" and r["success"]]
    ready = bool(dpt) and len(queries) == 7
    selected = min(dpt, key=lambda r: r["median_update_seconds"]) if dpt else None
    summary = dict(
        schema="ball_capacity_benchmark.v1",
        controller_sha256=dual_sha256(Path(__file__)),
        gpu_name=torch.cuda.get_device_name(0),
        complete=ready,
        cases=results,
        selected_dpt_microbatch=selected["microbatch_windows"] if selected else None,
        selected_query_microbatch=1,
        activation_checkpointing=False,
        frame_chunk_size=None,
        estimated_compute_hours=(
            10000 * selected["median_update_seconds"]
            + 6000 * sum(r["median_update_seconds"] for r in queries)
        )
        / 3600
        if ready
        else None,
        estimate_excludes="full validation, dataset verification, compilation and verified Drive snapshots",
        code=source_identity(),
    )
    save_json(root / "benchmark.json", summary)
    archive(args, root, [p.name for p in root.glob("*.json")])
    if not ready:
        raise RuntimeError(
            "Not all eight model configurations fit. Consult user before enabling memory controls."
        )


def stage_definitions(dpt_micro):
    return [("common-dpt-T32", "dpt", 256, 4, dpt_micro)] + [
        (f"query-d{d}-l{depth}-T64", "query", d, depth, 1) for d, depth in CONDITIONS
    ]


def train(args):
    prepared, download = inputs(args)
    measured = json.loads(args.benchmark.read_text())
    if (
        measured.get("schema") != "ball_capacity_benchmark.v1"
        or not measured["complete"]
    ):
        raise ValueError("All eight GPU forward/backward preflights must pass")
    if measured["controller_sha256"] != dual_sha256(Path(__file__)):
        raise ValueError("Campaign controller changed after benchmark")
    expected = [("dpt", 256, 4, measured["selected_dpt_microbatch"])]
    expected += [("query", dim, depth, 1) for dim, depth in CONDITIONS]
    for architecture, dim, depth, micro in expected:
        cases = [
            r
            for r in measured["cases"]
            if r["success"]
            and (r["architecture"], r["dim"], r["layers"], r["microbatch_windows"])
            == (architecture, dim, depth, micro)
        ]
        if len(cases) != 1:
            raise ValueError("Benchmark condition is missing or duplicated")
        cfg = json.loads(json.dumps(asdict(configuration(architecture, dim, depth))))
        if cases[0]["model_config"] != cfg or cases[0]["runtime"] != asdict(runtime()):
            raise ValueError(
                "Production model/runtime differs from the measured configuration"
            )
        if any(r["input_frames"] != 128 for r in cases[0]["records"]):
            raise ValueError("Benchmark did not retain the physical input-frame budget")
    if measured["code"]["source_sha256"] != source_identity()["source_sha256"]:
        raise ValueError("Model source changed after benchmark")
    root = (
        args.workspace_root
        / "outputs/ball_detection/train/capacity_scaling"
        / args.run_id
    )
    definitions = stage_definitions(measured["selected_dpt_microbatch"])
    plan = dict(
        schema="ball_capacity_campaign.v1",
        issue=1675,
        seed=42,
        optimizer_updates=52000,
        physical_input_frames=52000 * 128,
        benchmark_sha256=dual_sha256(args.benchmark),
        download_distribution_sha256=download["distribution_sha256"],
        controller_sha256=dual_sha256(Path(__file__)),
        theory_sha256=dual_sha256(Path(__file__).with_name("theory.json")),
        analysis_sha256=dual_sha256(Path(__file__).with_name("report.py")),
        prior_recipe_sha256=dual_sha256(PRIOR),
        primary_metric="mean_error_source_px",
        secondary_metrics=["median_error_px", "p95_error_px", "within_20px_rate"],
        no_seed_variance_estimate=True,
        evaluation_split="val",
        activation_checkpointing=False,
        frame_chunk_size=None,
        stages=[
            dict(
                name=n,
                architecture=a,
                model=asdict(configuration(a, d, depth)),
                budget=budget(a, b).payload(),
                parent=None if a == "dpt" else "common-dpt-T32",
            )
            for n, a, d, depth, b in definitions
        ],
        code=source_identity(),
        posttraining_recipe="First stage immediately after common DPT in #1514; T64 directly, LR horizon12000 offset0, 600 frozen updates, encoder LR ratio0.1, named encoder AdamW moments inherited.",
    )
    if root.exists():
        if not args.resume:
            raise ValueError("Existing campaign requires --resume")
        old = json.loads((root / "plan.json").read_text())
        comparable = dict(plan, code=old["code"])
        if (
            old != comparable
            or old["code"]["source_sha256"] != plan["code"]["source_sha256"]
        ):
            raise ValueError(
                "Resume requires the unchanged original campaign plan and source"
            )
        plan = old
    else:
        if args.resume:
            raise ValueError("No campaign exists to resume")
        root.mkdir(parents=True)
        save_json(root / "plan.json", plan)
        shutil.copyfile(args.benchmark, root / "benchmark.json")
        shutil.copyfile(
            args.workspace_root / DOWNLOAD / "progress.json", root / "download.json"
        )
        subprocess.run(
            ["git", "bundle", "create", str(root / "source.bundle"), "HEAD"], check=True
        )
        shutil.copyfile(Path(__file__).with_name("theory.json"), root / "theory.json")
        shutil.copytree(prepared, root / "prepared")
        include = [
            "plan.json",
            "benchmark.json",
            "download.json",
            "source.bundle",
            "theory.json",
        ]
        include += [
            p.relative_to(root).as_posix()
            for p in (root / "prepared").rglob("*")
            if p.is_file()
        ]
        receipt = RcloneSnapshotStore(
            root,
            remote="mounted_drive",
            remote_root="outputs/ball_detection/train/capacity_scaling/"
            + args.run_id
            + "/inputs",
            config_path=args.rclone_config,
        ).publish("inputs-v1", include=include)
        save_json(root / "inputs-receipt.json", receipt)
        archive(args, root, ["plan.json", "benchmark.json", "inputs-receipt.json"])

    for name, architecture, dim, layers, micro in definitions:
        output = root / name
        bud = budget(architecture, micro)
        cfg = configuration(architecture, dim, layers)
        options = {}
        if (output / "COMPLETED.json").exists():
            done = json.loads((output / "COMPLETED.json").read_text())
            latest = json.loads((output / "latest.json").read_text())
            saved = torch.load(
                output / latest["checkpoint"], map_location="cpu", weights_only=True
            )
            expected_model = json.loads(json.dumps(asdict(cfg)))
            if (
                not done["complete"]
                or done["updates_completed"] != bud.updates
                or latest["update"] != bud.updates
            ):
                raise ValueError("Invalid stage completion")
            if (
                dual_sha256(output / latest["checkpoint"]) != latest["sha256"]
                or json.loads(json.dumps(saved["recipe"]["model"])) != expected_model
                or saved["recipe"]["budget"] != bud.payload()
            ):
                raise ValueError("Completed checkpoint no longer matches this stage")
            if (
                saved["code"]["source_sha256"] != plan["code"]["source_sha256"]
                or saved["validation"] is None
            ):
                raise ValueError("Completed stage lacks frozen-code validation")
            continue
        if output.exists():
            if not args.resume or not (output / "latest.json").is_file():
                raise ValueError(
                    "An incomplete stage needs an explicit saved checkpoint to resume"
                )
            latest = json.loads((output / "latest.json").read_text())
            if dual_sha256(output / latest["checkpoint"]) != latest["sha256"]:
                raise ValueError("Latest checkpoint digest differs")
            options = {"resume": output / latest["checkpoint"]}
        elif architecture == "query":
            common = root / "common-dpt-T32"
            latest = json.loads((common / "latest.json").read_text())
            if latest["update"] != 10000:
                raise ValueError("Every query must branch from completed common10k")
            options = dict(
                initial_artifact=common / latest["checkpoint"],
                initial_artifact_sha256=latest["sha256"],
                parent_update=10000,
            )
        persist = prior.persist_callback(
            root,
            output,
            args.rclone_config,
            "outputs/ball_detection/train/capacity_scaling/" + args.run_id,
        )
        try:
            result = train_temporal_model(
                architecture=architecture,
                phase="pretrain" if architecture == "dpt" else "finetune",
                manifest=prepared / f"train_T{cfg.frames}.json",
                evaluation_manifest=prepared / f"evaluation_T{cfg.frames}.json",
                model_config=cfg,
                output=output,
                budget=bud,
                runtime=runtime(),
                device=torch.device("cuda"),
                augmentation=AugmentationConfig.load(
                    Path(
                        "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml"
                    )
                ),
                distance_px=20.0,
                persist=persist,
                **options,
            )
        except Exception:
            if output.exists():
                save_json(
                    output / "FAILED.json",
                    dict(complete=False, traceback=traceback.format_exc()),
                )
                persist(output, "failed")
            raise
        if not result["complete"]:
            raise ValueError("Stage returned without completing its full update budget")
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    save_json(
        root / "COMPLETED.json",
        dict(
            complete=True,
            optimizer_updates=52000,
            stages=[s[0] for s in definitions],
            plan=plan,
        ),
    )
    archive(args, root, ["COMPLETED.json"])
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("report.py")),
            "--campaign",
            str(root),
            "--theory",
            str(root / "theory.json"),
            "--output",
            str(root / "report"),
        ],
        check=True,
    )
    archive(
        args,
        root,
        [
            p.relative_to(root).as_posix()
            for p in (root / "report").iterdir()
            if p.is_file()
        ],
    )
    save_json(
        root / "ANALYSIS_COMPLETED.json",
        dict(complete=True, summary_sha256=dual_sha256(root / "report/summary.json")),
    )
    archive(args, root, ["ANALYSIS_COMPLETED.json"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["benchmark", "train"])
    p.add_argument("--workspace-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--rclone-config", type=Path, required=True)
    p.add_argument("--benchmark", type=Path)
    p.add_argument("--benchmark-updates", type=int, default=15)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.benchmark_updates < 6:
        p.error("Benchmark needs at least six joint updates")
    if args.action == "train" and args.benchmark is None:
        p.error("--benchmark required")
    globals()[args.action](args)


if __name__ == "__main__":
    main()
