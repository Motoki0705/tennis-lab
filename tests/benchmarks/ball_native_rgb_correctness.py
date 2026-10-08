"""GPU native/compiled MDD numerical and gradient checks; queue execution only."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.tasks.ball_detection.data.coordinate_dataset import (
    CoordinateWindowDataset,
    collate_coordinate_windows,
)
from src.tasks.ball_detection.data.pose_windows import coordinate_loss
from src.tasks.ball_detection.models.mdd_pose import MDDPoseConfig, MDDQueryDetector
from src.tasks.ball_detection.preprocessing import (
    RGBToMDD,
    luminance_to_mdd,
    mdd_coefficients,
)
from src.tasks.ball_detection.training.coordinate_compilation import (
    coordinate_compilation_report,
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.coordinate_evaluation import predict_coordinates
from src.tasks.ball_detection.training.coordinate_runtime import CoordinateRuntime
from src.utils.checksum import dual_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new correctness report")
    device = torch.device("cuda")
    runtime = CoordinateRuntime(precision="bf16", compile_mode="default")
    runtime.configure(device)
    torch.manual_seed(42)
    data = CoordinateWindowDataset(args.manifest, split="train", requires_pose=False)
    selected: dict[tuple[str, int], int] = {}
    for index, (record, window) in enumerate(data.windows):
        selected.setdefault((data.records[record]["clip"]["source"], window.frame_step), index)
    transform = RGBToMDD().to(device)
    compiled_transform = RGBToMDD().to(device)
    compiled_transform.compile(backend="inductor", fullgraph=True, dynamic=False)
    differences = []
    for (source, step), index in sorted(selected.items()):
        sample = data[index]
        rgb = sample["rgb"][None].to(device)
        # Independent reproduction of the previous NumPy BGR/luminance reader.
        bgr = sample["rgb"].numpy().transpose(0, 2, 3, 1)[..., ::-1].astype(np.float32) / 255
        gray = .114 * bgr[..., 0] + .587 * bgr[..., 1] + .299 * bgr[..., 2]
        gain, offset = mdd_coefficients(.2, .15)
        reference = luminance_to_mdd(torch.from_numpy(gray)[None], gain=gain, offset=offset)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            eager = transform(rgb).cpu()
            compiled = compiled_transform(rgb).cpu()
        assert eager.dtype == compiled.dtype == torch.float32
        assert not compiled[:, :, 0].any()
        torch.testing.assert_close(eager, reference, rtol=1e-5, atol=3e-6)
        torch.testing.assert_close(compiled, reference, rtol=1e-5, atol=3e-6)
        row = dict(source=source, frame_step=step, clip_id=sample["clip_id"],
                   eager_max_abs=float((eager-reference).abs().max()),
                   compiled_max_abs=float((compiled-reference).abs().max()))
        differences.append(row)
        print(json.dumps(row), flush=True)
        del rgb, bgr, gray, reference, eager, compiled
    # Dropout is disabled only for this numerical comparison; the throughput
    # diagnostic uses the unmodified dropout=0.1 training configuration.
    config = replace(MDDPoseConfig.load(args.model_config), dropout=0.)
    eager_model = MDDQueryDetector(config).to(device).train()
    compiled_model = MDDQueryDetector(config).to(device).train()
    compiled_model.load_state_dict(eager_model.state_dict(), strict=True)
    names = set(eager_model.state_dict())
    runtime.configure_model(compiled_model)
    batch = collate_coordinate_windows([data[next(iter(selected.values()))]])
    gradients, predictions, losses = [], [], []
    for model in (eager_model, compiled_model):
        with coordinate_compile_scope(model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = predict_coordinates(model, batch, device)
                loss = coordinate_loss(prediction, batch["uv"].to(device), batch["position_valid"].to(device))
            loss.backward()
        gradient = torch.cat([p.grad.detach().float().flatten() for p in model.parameters() if p.grad is not None]).cpu()
        assert torch.isfinite(gradient).all()
        gradients.append(gradient)
        predictions.append(prediction.detach().cpu())
        losses.append(float(loss.detach()))
    relative = float((gradients[0]-gradients[1]).norm()/gradients[0].norm().clamp_min(1e-12))
    uv_delta = float((predictions[0]-predictions[1]).abs().max())
    assert relative < .05, relative
    assert uv_delta < .003, uv_delta
    assert abs(losses[0]-losses[1]) < .002, losses
    assert set(compiled_model.state_dict()) == names
    restored = MDDQueryDetector(config).to(device).eval()
    restored.load_state_dict(compiled_model.state_dict(), strict=True)
    compiled_model.eval()
    with torch.no_grad(), coordinate_compile_scope(compiled_model), torch.autocast("cuda", dtype=torch.bfloat16):
        first = predict_coordinates(compiled_model, batch, device).cpu()
        second = predict_coordinates(restored, batch, device).cpu()
    restore_delta = float((first-second).abs().max())
    assert restore_delta < .003, restore_delta
    compilation = coordinate_compilation_report(compiled_model)
    assert compilation["unique_graphs"] > 0 and not compilation["graph_breaks"]
    report = dict(status="ok", manifest_sha256=dual_sha256(args.manifest), input_contract=transform.input_contract(),
                  preprocessing=differences, gradient_relative_l2=relative, max_uv_delta=uv_delta,
                  loss_eager=losses[0], loss_compiled=losses[1], restore_max_uv_delta=restore_delta,
                  state_dict_names_unchanged=True, compilation=compilation,
                  tolerances=dict(mdd_atol=3e-6, mdd_rtol=1e-5, gradient_relative_l2=.05, uv_abs=.003, loss_abs=.002),
                  scope="train-only numerical diagnostic; no production training or test evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
