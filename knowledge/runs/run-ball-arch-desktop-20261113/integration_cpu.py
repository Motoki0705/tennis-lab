"""CPU-only integration comparison; official MSDA reference is test instrumentation."""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--case", choices=("scene", "dpt", "query", "variants"), required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--expected", type=Path)
p.add_argument("--assets", type=Path, required=True)
p.add_argument("--input-rgb", type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.source.resolve()))
import torch

assert not torch.cuda.is_available()
torch.set_num_threads(2)
torch.manual_seed(1859)
from src.tasks.ball_detection.model_io.mdd_temporal import (
    build_temporal_model,
    restore_temporal_model,
    temporal_forward,
    temporal_model_payload,
)

start = time.monotonic()
rgb = torch.load(a.input_rgb, weights_only=True)
records = {}
states = {}
if a.case == "scene":
    from src.tasks.ball_detection.inference.x3d_dpt import X3DDPTPredictor

    checkpoint = a.assets / "scene.pt"
    model = X3DDPTPredictor.load_from_checkpoint(
        checkpoint,
        device="cpu",
        image_size=(720, 1280),
        frames=32,
        input_channels=5,
        normalize_imagenet=False,
        precision="fp32",
        subpixel_refine=True,
        compile_mode="off",
    )
    with torch.inference_mode():
        pred = model.predict(rgb)
    records = {k: getattr(pred, k) for k in ("coords", "confidence", "heatmaps")}
    for k in ("coords", "scores", "valid", "cells", "patches", "patch_valid"):
        if hasattr(pred.candidates, k):
            records["candidates_" + k] = getattr(pred.candidates, k)
    states = {"state_tensors": len(model.model.state_dict()), "strict_load": True}
else:
    if a.case in ("dpt", "query"):
        checkpoint = a.assets / ("dpt.pt" if a.case == "dpt" else "query.pt")
        full = torch.load(checkpoint, map_location="cpu", weights_only=True)
        artifact = full.get("model_artifact", full)
        models = {a.case: restore_temporal_model(artifact).eval()}
    else:
        from src.tasks.ball_detection.models.mdd.multiscale_config import (
            MDDMultiScaleQueryConfig,
        )
        from src.tasks.ball_detection.models.mdd.stage_config import (
            MDDStageHeatmapConfig,
        )
        from src.utils.models.components.ops.msda.api import OfficialMSDeformAttn
        from src.utils.models.components.ops.msda.vendor.functions.ms_deform_attn_func import (
            MSDeformAttnFunction,
            ms_deform_attn_core_pytorch,
        )

        # Explicit, verification-only substitution: production CUDA boundary stays unchanged.
        def cpu_test_forward(self, q, r, v, s, b, mask=None):
            return self.op(q.float(), r.float(), v.float(), s, b, mask)

        OfficialMSDeformAttn.forward = cpu_test_forward
        MSDeformAttnFunction.apply = staticmethod(
            lambda v, s, b, locations, w, step: ms_deform_attn_core_pytorch(
                v, s, locations, w
            )
        )
        dpt = torch.load(a.assets / "dpt.pt", map_location="cpu", weights_only=True)[
            "model_artifact"
        ]["model_config"]
        query = torch.load(
            a.assets / "query.pt", map_location="cpu", weights_only=True
        )["model_artifact"]["model_config"]
        models = {}
        for name, kinds, e in (
            ("C1", ("ib",) * 5, 1.5),
            ("C2", ("convnext",) * 2 + ("ffn",) * 3, 2.0),
            ("C3", ("convnext",) * 2 + ("ib",) * 3, 2.0),
        ):
            torch.manual_seed(42)
            models[name] = build_temporal_model(
                MDDStageHeatmapConfig(
                    **dpt, stage_blocks=kinds, expansion_ratios=(e,) * 5
                )
            ).eval()
        for name, layers in (
            ("T1", ("M",) * 3 + ("C",) * 3),
            ("T2", ("C",) * 3 + ("M",) * 3),
        ):
            torch.manual_seed(42)
            models[name] = build_temporal_model(
                MDDMultiScaleQueryConfig(
                    **query,
                    spatial_layers=layers,
                    msda_levels=(16, 32, 64),
                    msda_points=4,
                    msda_chunk_size=4,
                    msda_im2col_step=64,
                )
            ).eval()
    for name, model in models.items():
        if a.expected and a.case == "variants":
            original = torch.load(a.expected, map_location="cpu", weights_only=True)[
                "states"
            ][name]["payload"]
            assert asdict(model.config) == original["model_config"]
            model.load_state_dict(original["state_dict"], strict=True)
        payload = temporal_model_payload(model)
        restored = restore_temporal_model(
            payload
        ).eval()  # strict=True inside the public loader
        states[name] = dict(
            state_tensors=len(model.state_dict()),
            strict_load=True,
            config=payload["model_config"],
        )
        frames = model.config.frames
        data = torch.nn.functional.interpolate(
            rgb[0].float(),
            size=(65, 129) if a.case != "variants" else (17, 25),
            mode="bilinear",
            align_corners=False,
        ).to(torch.uint8)[None]
        data = data.repeat(1, (frames + 31) // 32, 1, 1, 1)[:, :frames]
        pts = torch.arange(frames, dtype=torch.float32)[None] / 30
        with torch.inference_mode():
            records[name] = temporal_forward(restored, data, pts)
        states[name]["payload"] = payload
report = dict(
    input_rgb_sha256=hashlib.file_digest(a.input_rgb.open("rb"), "sha256").hexdigest(),
    case=a.case,
    source=str(a.source),
    seconds=time.monotonic() - start,
    device="cpu",
    precision="fp32",
    strict_load=True,
    msda="official PyTorch reference explicitly substituted in verification process only"
    if a.case == "variants"
    else None,
    shapes={k: list(v.shape) for k, v in records.items()},
)
if a.expected:
    expected = torch.load(a.expected, map_location="cpu", weights_only=True)
    assert records.keys() == expected["records"].keys()
    errors = {}
    for key, value in records.items():
        other = expected["records"][key]
        assert torch.equal(value, other), (
            key,
            (value.float() - other.float()).abs().max().item(),
        )
        errors[key] = (value.float() - other.float()).abs().max().item()
    report["max_abs_error"] = errors
else:
    torch.save(dict(records=records, states=states), a.output)
a.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
