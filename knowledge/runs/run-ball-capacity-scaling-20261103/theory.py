"""Forward MACs for actual model shapes; not a latency/FLOP estimator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from torch import nn

from src.tasks.ball_detection.model_io.mdd_temporal import build_temporal_model
from src.utils.models.components.mhc import ManifoldConstrainedHyperConnection


def load_run():
    s = importlib.util.spec_from_file_location(
        "capacity_recipe", Path(__file__).with_name("run.py")
    )
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def spatial_macs(model, height, width, decoder=False):
    values = {}
    handles = []

    def hook(name, module, inputs, output):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            macs = (
                output.numel()
                * (module.in_channels // module.groups)
                * math.prod(module.kernel_size)
            )
        else:
            macs = output.numel() * module.in_features
        values[name] = values.get(name, 0) + macs

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            handles.append(
                module.register_forward_hook(lambda m, i, o, n=name: hook(n, m, i, o))
            )
    try:
        with torch.no_grad():
            features = model.encoder(torch.empty(1, 5, 1, height, width, device="meta"))
            if decoder:
                from torch.nn import functional as F

                fused = model.decoder([x.flatten(0, 1) for x in features])
                model.head(
                    F.interpolate(
                        fused,
                        size=((height + 3) // 4, (width + 3) // 4),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
    finally:
        for h in handles:
            h.remove()
    return dict(
        total=sum(values.values()),
        by_module=values,
        feature_shapes=[list(x.shape) for x in features],
    )


def query_macs(model, height, width, frames):
    c = model.config.dim
    f = model.config.ffn_dim
    p = 4
    spatial = spatial_macs(model, height, width)
    shape = spatial["feature_shapes"][-1]
    memory = shape[-1] * shape[-2]
    channels = shape[2]
    projection = memory * channels * c + memory * 2 * c
    cross_projection = (2 * memory + 2 * p) * c * c
    cross_attention = 2 * p * memory * c
    temporal_projection = 4 * c * c
    temporal_attention = 2 * frames * c
    ffn = 3 * c * f
    layers = []
    for block in model.blocks:
        mhcs = [
            m
            for m in block.modules()
            if isinstance(m, ManifoldConstrainedHyperConnection)
        ]
        if len(mhcs) != 1:
            raise ValueError("Expected one mHC span per layer")
        mhc = mhcs[0]
        h = mhc.coefficient_dim
        if mhc.num_streams != 4:
            raise ValueError("Expected exactly four streams")
        occurrences = dict(
            stream_in=p,
            stream_out=p,
            pre_head=p,
            post_head=p,
            residual_query=p,
            residual_key=p,
            pair_in=p * p,
            pair_out=p * p,
        )
        coefficients = sum(
            getattr(mhc, n).weight.numel() * nuses for n, nuses in occurrences.items()
        )
        products = p * p * h + p * c + p * p * c
        parts = dict(
            cross_projection=cross_projection,
            cross_attention=cross_attention,
            temporal_projection=temporal_projection,
            temporal_attention=temporal_attention,
            swiglu=ffn,
            mhc_coefficients=coefficients,
            mhc_products=products,
        )
        layers.append(dict(total=sum(parts.values()), **parts))
    head = 2 * c
    transformer = projection + head + sum(x["total"] for x in layers)
    return dict(
        cnn_macs_per_frame=spatial["total"],
        transformer_macs_per_frame=transformer,
        total_macs_per_frame=spatial["total"] + transformer,
        projection_macs_per_frame=projection,
        readout_macs_per_frame=head,
        per_layer_macs=layers,
        cnn=spatial,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    a = p.parse_args()
    run = load_run()
    results = []
    for architecture, dim, layers in [("dpt", 256, 4)] + [
        ("query", d, depth) for d, depth in run.CONDITIONS
    ]:
        cfg = run.configuration(architecture, dim, layers)
        with torch.device("meta"):
            model = build_temporal_model(cfg)
        if architecture == "dpt":
            counts = spatial_macs(model, a.height, a.width, decoder=True)
            cnn = sum(
                v for k, v in counts["by_module"].items() if k.startswith("encoder.")
            )
            counts = dict(
                cnn_macs_per_frame=cnn,
                dpt_macs_per_frame=counts["total"] - cnn,
                total_macs_per_frame=counts["total"],
                by_module=counts["by_module"],
                feature_shapes=counts["feature_shapes"],
            )
        else:
            counts = query_macs(model, a.height, a.width, 64)
        parameters = sum(x.numel() for x in model.parameters())
        cnn_parameters = sum(x.numel() for x in model.encoder.parameters())
        results.append(
            dict(
                architecture=architecture,
                dim=dim,
                layers=layers,
                frames=cfg.frames,
                heads=cfg.heads,
                ffn_dim=cfg.ffn_dim,
                total_parameters=parameters,
                cnn_parameters=cnn_parameters,
                non_encoder_parameters=parameters - cnn_parameters,
                **counts,
            )
        )
    report = dict(
        schema="ball_capacity_forward_macs.v1",
        resolution=[a.height, a.width],
        cases=results,
        definition="One multiply-accumulate is one MAC. Conv/Linear and attention matrix products, padded positions included. Per input frame. Spatial Conv/DPT output shapes counted with meta forward; query/mHC matrix sizes counted analytically from actual config/weights.",
        excluded=[
            "normalization",
            "activation",
            "softmax/Sinkhorn elementwise reductions",
            "MDD",
            "bias/elementwise operations",
            "input decoding/transfer",
            "backward",
            "optimizer",
        ],
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    for r in results:
        print(
            r["architecture"],
            r["dim"],
            r["layers"],
            round(r["total_parameters"] / 1e6, 3),
            "M",
            round(r["total_macs_per_frame"] / 1e9, 4),
            "GMAC/frame",
        )


if __name__ == "__main__":
    main()
