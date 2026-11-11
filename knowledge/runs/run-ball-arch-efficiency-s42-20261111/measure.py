"""Only the confirmed architecture matrix, three reordered CUDA measurement blocks."""

from __future__ import annotations

import argparse
import gc
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from protocol import DRIVE, REMOTE, ROOT, model_config, read, write
from serial import index_store, snapshot_store
from torch import nn
from torch.nn import functional as F

from src.tasks.ball_detection.model_io.mdd_temporal import build_temporal_model
from src.tasks.ball_detection.models.mdd.multiscale_query import (
    DeepMDDMultiScaleQueryDetector,
    DeformableMHCQueryBlock,
    spatial_grid,
)
from src.tasks.ball_detection.training.heatmap_objective import heatmap_objective
from src.utils.checksum import dual_sha256
from tests.benchmarks.ball_detection.efficiency import (
    benchmark,
    environment,
    forward_macs,
)


class Full(nn.Module):
    def __init__(self, model, query):
        super().__init__()
        self.network = model
        self.query = query

    def forward(self, rgb, times):
        return self.network(rgb, times) if self.query else self.network(rgb)


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.head = model.head

    def forward(self, *features):
        b, t = features[0].shape[:2]
        value = self.decoder([x.flatten(0, 1) for x in features])
        return self.head(
            F.interpolate(value, size=(180, 320), mode="bilinear", align_corners=False)
        ).reshape(b, t, 180, 320)


class Query(nn.Module):
    def __init__(self, model, multiscale):
        super().__init__()
        self.config = model.config
        self.multiscale = multiscale
        for name in ("projection", "spatial_position", "blocks", "head", "ball_query"):
            setattr(self, name, getattr(model, name))
        if multiscale:
            self.level_projections = model.level_projections
            self.level_embedding = model.level_embedding

    def forward(self, times, *features):
        b, t = features[-1].shape[:2]
        if self.multiscale:
            patches, memory = DeepMDDMultiScaleQueryDetector.prepare_memory(
                self, features, image_size=(720, 1280)
            )
        else:
            feature = features[-1]
            h, w = feature.shape[-2:]
            patches = self.projection(feature.flatten(-2).transpose(-1, -2))
            patches = (
                patches
                + self.spatial_position(
                    spatial_grid(h, w, stride=64, image_size=(720, 1280), like=feature)
                )[None, None]
            )
        streams = self.ball_query.expand(b, t, *self.ball_query.shape)
        times = times - times[:, :1]
        for block in self.blocks:
            streams = block(
                streams,
                memory if isinstance(block, DeformableMHCQueryBlock) else patches,
                times,
            )
        return self.head(streams.mean(-2)).float().sigmoid()


class Spatial(nn.Module):
    def __init__(self, model, deformable, chunk):
        super().__init__()
        self.deformable = deformable
        self.chunk = chunk
        block = next(
            b
            for b in model.blocks
            if isinstance(b, DeformableMHCQueryBlock) == deformable
        )
        path = block.path if deformable else block
        self.normq, self.normv = path.norms[0], path.norms[1]
        self.cross = block.cross
        if deformable:
            self.references = block.reference_points

    def forward(self, q, v, shapes, starts, mask):
        q = self.normq(q)
        if not self.deformable:
            v = self.normv(v)
            return self.cross(q, v, v, need_weights=False)[0]
        with torch.autocast("cuda", enabled=False):
            references = (
                self.references(q.float())
                .sigmoid()[:, :, None]
                .expand(-1, -1, shapes.shape[0], -1)
            )
        outputs = []
        for first in range(0, q.shape[0], self.chunk):
            last = first + self.chunk
            outputs.append(
                self.cross(
                    q[first:last],
                    references[first:last],
                    self.normv(v[first:last]),
                    shapes,
                    starts,
                    mask[first:last],
                )
            )
        return torch.cat(outputs)


def descriptors():
    cases = [
        dict(key="full-" + c, kind="full", case=c)
        for c in ("C0", "C1", "C2", "C3", "CA", "T1", "T2")
    ]
    cases += [
        dict(key="encoder-" + c, kind="encoder", case=c)
        for c in ("C0", "C1", "C2", "C3")
    ]
    cases += [dict(key="DPT-decoder", kind="decoder", case="C0")]
    cases += [dict(key="query-" + c, kind="query", case=c) for c in ("CA", "T1", "T2")]
    seen = set()
    for c in ("C0", "C1", "C2", "C3"):
        cfg = model_config(c)
        kinds = getattr(cfg, "stage_blocks", ("ib",) * 5)
        expansions = getattr(cfg, "expansion_ratios", (2.0,) * 5)
        for stage in range(6):
            identity = (
                stage,
                kinds[stage - 1] if stage else "stem",
                expansions[stage - 1] if stage else 0.0,
            )
            if identity in seen:
                continue
            seen.add(identity)
            cases.append(
                dict(
                    key=f"stage-{stage}-{identity[1]}-e{identity[2]}",
                    kind="stage",
                    case=c,
                    stage=stage,
                )
            )
            if stage:
                cases.append(
                    dict(
                        key=f"block-{stage}-{identity[1]}-e{identity[2]}",
                        kind="block",
                        case=c,
                        stage=stage,
                    )
                )
    cases += [
        dict(key="spatial-CA", kind="spatial", case="CA"),
        dict(key="spatial-MSDA", kind="spatial", case="T1"),
    ]
    return cases


def feature_inputs(model, t):
    model.encoder.cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        features = model.encoder(torch.randn(1, 5, t, 720, 1280, device="cuda"))
    return tuple(x.detach().requires_grad_() for x in features)


def build(spec):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    chosen = tuple(read(ROOT / "level-decision.json")["levels"])
    cfg = model_config(spec["case"], levels=chosen)
    model = build_temporal_model(cfg)
    torch.manual_seed(42)
    kind = spec["kind"]
    t = cfg.frames
    query = spec["case"] in ("CA", "T1", "T2")
    fullgraph = spec["case"] not in ("T1", "T2")
    if kind == "full":
        module = Full(model, query)
        args = (
            torch.randint(256, (1, t, 3, 720, 1280), device="cuda", dtype=torch.uint8),
            torch.arange(t, device="cuda")[None].float() / 30,
        )
        target = dict(
            uv=torch.full((1, t, 2), 0.5, device="cuda"),
            position_valid=torch.ones(1, t, dtype=torch.bool, device="cuda"),
            heatmap_valid=torch.ones(1, t, dtype=torch.bool, device="cuda"),
        )
        loss = (
            (lambda x: F.smooth_l1_loss(x.float(), target["uv"], beta=0.01))
            if query
            else (
                lambda x: heatmap_objective(x, target, sigma_ratio=0.012, gamma=2.0)[0]
            )
        )
        micro = 128 // t
    elif kind == "encoder":
        module = model.encoder
        args = (torch.randn(1, 5, t, 720, 1280, device="cuda"),)
        micro = 128 // t

        def loss(values):
            return sum(x.float().square().mean() for x in values)
    elif kind in ("decoder", "query"):
        features = feature_inputs(model, t)
        if kind == "decoder":
            module = Decoder(model)
            args = features
        else:
            module = Query(model, spec["case"] != "CA")
            args = (torch.arange(t, device="cuda")[None].float() / 30, *features)

        def loss(x):
            return x.float().square().mean()

        micro = 128 // t
    elif kind in ("stage", "block"):
        model.encoder.cuda().eval()
        index = spec["stage"]
        x = torch.randn(1, 5, t, 720, 1280, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for stage in model.encoder.stages[:index]:
                x = stage(x)
            if kind == "block":
                for operation in list(model.encoder.stages[index])[:6]:
                    x = operation(x)
        module = (
            model.encoder.stages[index]
            if kind == "stage"
            else model.encoder.stages[index][6]
        )
        args = (x.detach().requires_grad_(),)

        def loss(x):
            return x.float().square().mean()

        micro = 1
    else:
        deformable = spec["case"] == "T1"
        module = Spatial(model, deformable, 4)
        shape = [
            (math.ceil(720 / s), math.ceil(1280 / s))
            for s in (chosen if deformable else (64,))
        ]
        shapes = torch.tensor(shape, device="cuda", dtype=torch.long)
        starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
        tokens = sum(h * w for h, w in shape)
        args = (
            torch.randn(t, 4, 512, device="cuda", requires_grad=True),
            torch.randn(
                t,
                tokens,
                512,
                device="cuda",
                dtype=torch.float32 if deformable else torch.bfloat16,
                requires_grad=True,
            ),
            shapes,
            starts,
            torch.zeros(t, tokens, device="cuda", dtype=torch.bool),
        )
        micro = 2

        def loss(x):
            return x.float().square().mean()

    module = module.cuda()
    return module, args, loss, t, micro, fullgraph


def one(spec, block):
    result_path = ROOT / "benchmarks" / f"block-{block}" / (spec["key"] + ".json")
    if result_path.exists():
        raise RuntimeError("Do not overwrite a completed measurement")
    phase = "preparation"
    try:
        start = time.perf_counter()
        module, args, loss, t, micro, fullgraph = build(spec)
        preparation = time.perf_counter() - start
        phase = "benchmark"
        result = benchmark(
            module,
            args,
            loss,
            frames=t,
            microbatches=micro,
            compile_mode="default",
            fullgraph=fullgraph,
            profiler_path=ROOT / "benchmarks/profiles" / (spec["key"] + ".json")
            if block == 0
            else None,
        )
        result.update(
            spec=spec,
            block=block,
            environment=environment(),
            preparation_seconds=preparation,
            input_shapes=[list(x.shape) for x in args],
            input_dtypes=[str(x.dtype) for x in args],
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            loss_contract="actual focal heatmap / smooth-L1 for full models; all-output MSE for isolated components",
            scope="synthetic resident inputs; excludes decode/augmentation; phase synchronization is included in wall time",
        )
        del module, args
        gc.collect()
        torch.cuda.empty_cache()
        if block == 0:
            phase = "mac_count"
            module, args, _, _, _, _ = build(spec)
            module.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result["macs"] = forward_macs(module, lambda: module(*args))
            result["macs"]["conv_linear_attention_macs_per_frame"] = (
                result["macs"]["conv_linear_attention_macs"] / t
            )
        write(result_path, result)
    except BaseException as error:
        write(
            result_path.with_suffix(".failed.json"),
            dict(
                spec=spec,
                block=block,
                phase=phase,
                error=str(error),
                traceback=traceback.format_exc(),
                allocated=torch.cuda.max_memory_allocated(),
                reserved=torch.cuda.max_memory_reserved(),
            ),
        )
        raise


def matrix():
    output = ROOT / "benchmarks"
    identity = dict(
        schema="ball_arch_benchmark.v1",
        sources={
            str(path): dual_sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("protocol.py"),
                Path("tests/benchmarks/ball_detection/efficiency.py"),
            )
        },
        scientific_source=read(ROOT / "plan.json")["source"],
        levels=read(ROOT / "level-decision.json")["levels"],
    )
    identity_path = output / "definition.json"
    if identity_path.exists() and read(identity_path) != identity:
        raise RuntimeError(
            "Benchmark definition changed; preserve and review the existing measurements"
        )
    write(identity_path, identity)
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for name in (
        "msda-build-manifest.json",
        "msda-compatibility.patch",
        "level-decision.json",
        "plan.json",
    ):
        shutil.copyfile(ROOT / name, provenance / name)
    (provenance / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "tests/integration/benchmarks/ball_detection/test_efficiency_cuda.py",
        ],
        check=True,
    )
    cases = descriptors()
    orders = [
        cases,
        list(reversed(cases)),
        cases[len(cases) // 2 :] + cases[: len(cases) // 2],
    ]
    write(
        ROOT / "benchmarks/order.json", [[s["key"] for s in order] for order in orders]
    )
    for block, order in enumerate(orders):
        for spec in order:
            result = ROOT / "benchmarks" / f"block-{block}" / (spec["key"] + ".json")
            remote = DRIVE / REMOTE / result.relative_to(ROOT)
            if not result.exists() and remote.exists():
                result.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(remote, result)
                if dual_sha256(result) != dual_sha256(remote):
                    raise RuntimeError("Benchmark recovery bytes differ")
            if result.exists():
                continue
            logs = ROOT / "benchmarks/logs"
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / (f"{block}-" + spec["key"] + ".log")).open(
                "ab", buffering=0
            ) as stream:
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "one",
                        "--case",
                        spec["key"],
                        "--block",
                        str(block),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            index_store().publish_file(result)
            if dual_sha256(result) != dual_sha256(remote):
                raise RuntimeError("Benchmark sample readback differs")
    store = snapshot_store(output)
    receipt = store.publish("completed-" + str(time.time_ns()))
    # The restore validates all bytes through an independent rclone process.
    import tempfile

    from src.utils.artifact_store import RcloneSnapshotStore

    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        RcloneSnapshotStore(
            Path(temporary) / "readback",
            remote="mounted_drive",
            remote_root=store.remote_root,
            config_path=store.config_path,
        ).restore(receipt["generation"], manifest_sha256=receipt["manifest_sha256"])
    write(ROOT / "benchmark-receipt.json", receipt)
    index_store().publish_file(ROOT / "benchmark-receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("matrix", "one"))
    parser.add_argument("--case")
    parser.add_argument("--block", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "matrix":
        matrix()
    else:
        one(next(s for s in descriptors() if s["key"] == args.case), args.block)
