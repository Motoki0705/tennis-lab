"""Real official CUDA sampling and frame-chunk gradient equivalence (no fallback)."""
import os
from dataclasses import replace

import pytest
import torch

from src.tasks.ball_detection.model_io.mdd_temporal import (
    restore_temporal_model,
    temporal_model_payload,
)
from src.tasks.ball_detection.models.mdd.multiscale_query import (
    DeepMDDMultiScaleQueryDetector,
)
from src.utils.models.components.ops.msda.api import OfficialMSDeformAttn
from tests.unit.tasks.ball_detection.models.mdd.test_multiscale_query import (
    multiscale_config,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("TENNIS_LAB_DINO_EXTENSION"),
                                reason="Requires CUDA and an explicitly built official MSDA binary")


def test_hand_computed_interpolation_padding_edges_and_gradients() -> None:
    module = OfficialMSDeformAttn(64, 1, 1, 1).cuda()
    with torch.no_grad():
        for linear in (module.op.value_proj, module.op.output_proj):
            linear.weight.copy_(torch.eye(64, device="cuda"))
            linear.bias.zero_()
        module.op.sampling_offsets.weight.zero_()
        module.op.sampling_offsets.bias.zero_()
        module.op.attention_weights.weight.zero_()
        module.op.attention_weights.bias.zero_()
    values = torch.tensor([1., 3., 5., 7.], device="cuda")[None, :, None].expand(1, 4, 64).clone().requires_grad_()
    locations = torch.tensor([[[[.25, .25]], [[.5, .5]], [[0., 0.]], [[1., 1.]], [[-.5, .5]]]], device="cuda", requires_grad=True)
    query = torch.zeros(1, 5, 64, device="cuda")
    shapes, starts = torch.tensor([[2, 2]], device="cuda"), torch.tensor([0], device="cuda")
    result = module(query, locations, values, shapes, starts)
    torch.testing.assert_close(result[0, :, 0], torch.tensor([1., 4., .25, 1.75, 0.], device="cuda"), rtol=0, atol=0)
    result[:, 1].sum().backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, .25), rtol=0, atol=0)
    assert locations.grad is not None and bool(torch.isfinite(locations.grad).all())
    mask = torch.tensor([[True, False, False, False]], device="cuda")
    padded = module(query, locations, values, shapes, starts, mask)
    assert float(padded[0, 1, 0]) == 3.75


def test_bf16_boundary_executes_official_projections_and_kernel_in_fp32() -> None:
    module = OfficialMSDeformAttn(64, 1, 1, 1).cuda()
    observed = []
    handle = module.op.register_forward_pre_hook(lambda _m, args: observed.append(tuple(x.dtype for x in args[:3])))
    try:
        q = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(2, 6, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ref = torch.full((2, 4, 1, 2), .5, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = module(q, ref, v, torch.tensor([[2, 3]], device="cuda"), torch.tensor([0], device="cuda"))
        output.square().mean().backward()
    finally:
        handle.remove()
    assert observed == [(torch.float32, torch.float32, torch.float32)]
    assert output.dtype == torch.float32 and v.grad is not None and bool(torch.isfinite(v.grad).all())


@pytest.mark.parametrize("layout", [("M", "C"), ("C", "M")])
def test_frame_chunk_matches_full_outputs_all_gradients_and_artifact_restore(layout) -> None:
    torch.manual_seed(1859)
    config = multiscale_config(spatial_layers=layout, msda_chunk_size=6)
    full = DeepMDDMultiScaleQueryDetector(config).cuda()
    chunked = DeepMDDMultiScaleQueryDetector(replace(config, msda_chunk_size=1)).cuda()
    chunked.load_state_dict(full.state_dict(), strict=True)
    rgb = torch.randint(256, (2, 3, 3, 257, 385), dtype=torch.uint8, device="cuda")
    times = torch.tensor([[0., .03, .07], [1., 1.04, 1.09]], device="cuda")
    a, b = full(rgb, times), chunked(rgb, times)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    probe = torch.tensor([[[.2, .9], [1.3, -.7], [.3, .6]]], device="cuda")
    (a*probe).sum().backward()
    (b*probe).sum().backward()
    for (name, p), (other, q) in zip(full.named_parameters(), chunked.named_parameters(), strict=True):
        assert name == other and p.grad is not None and q.grad is not None, name
        assert bool(torch.isfinite(p.grad).all()), name
        torch.testing.assert_close(p.grad, q.grad, rtol=5e-3, atol=2e-5, msg=name)
    gradients = dict(full.named_parameters())
    prefixes = ("encoder.", "level_projections.8.", "level_projections.16.", "level_projections.32.",
                "blocks.0.cross.op.sampling_offsets." if layout[0] == "M" else "blocks.1.cross.op.sampling_offsets.",
                "blocks.0.reference_points." if layout[0] == "M" else "blocks.1.reference_points.")
    for prefix in prefixes:
        assert sum(float(p.grad.abs().sum()) for name, p in gradients.items() if name.startswith(prefix) and p.grad is not None) > 0, prefix
    restored = restore_temporal_model(temporal_model_payload(full)).cuda().eval()
    full.eval()
    with torch.no_grad():
        torch.testing.assert_close(full(rgb, times), restored(rgb, times), rtol=0, atol=0)


def test_inductor_graph_break_mode_keeps_forward_and_backward_correct() -> None:
    from src.tasks.ball_detection.training.runtime.compilation import (
        compile_coordinate_model,
        coordinate_compilation_report,
        coordinate_compile_scope,
    )

    torch.manual_seed(1859)
    config = multiscale_config()
    model = DeepMDDMultiScaleQueryDetector(config).cuda()
    eager = DeepMDDMultiScaleQueryDetector(config).cuda()
    eager.load_state_dict(model.state_dict(), strict=True)
    rgb = torch.randint(256, (1, 3, 3, 65, 97), dtype=torch.uint8, device="cuda")
    times = torch.tensor([[0., .03, .07]], device="cuda")
    compile_coordinate_model(model, mode="default", fullgraph=False)
    with coordinate_compile_scope(model):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = model(rgb, times)
        actual.square().mean().backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = eager(rgb, times)
    expected.square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    report = coordinate_compilation_report(model)
    assert report["fullgraph"] is False and report["graph_breaks"]
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
