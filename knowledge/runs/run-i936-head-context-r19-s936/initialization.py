"""CPU evidence for the sole head-input factor, on one real training window."""
from __future__ import annotations

import json
import resource
import time
from dataclasses import replace
from pathlib import Path

import torch

from src.tasks.ball_refiner.refiner_3d.diffusion.data import rally_window
from src.tasks.ball_refiner.refiner_3d.diffusion.dev_config import load_config
from src.tasks.ball_refiner.refiner_3d.diffusion.flow import training_objective
from src.tasks.ball_refiner.refiner_3d.diffusion.model import TrajectoryDenoiser
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(1)
    bundle = Path(__file__).resolve().parent
    project = bundle.parents[2]
    base = Path('/home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/h-dev/r16-combined512-physics10-s936-t128-20k')
    config_path = project / 'src/tasks/ball_refiner/refiner_3d/training_pilot512_physics10_head_context_t128.yaml'
    config = load_config(config_path)
    control_manifest = json.loads((base / 'manifest.json').read_text())
    checkpoint = base / 'flow/initial-state.pt'
    assert sha256(checkpoint) == control_manifest['arms']['flow']['initial_state_sha256']
    historical = torch.load(checkpoint, map_location='cpu', weights_only=True)
    torch.manual_seed(config.seed)
    control = TrajectoryDenoiser(replace(config.model, position_head_input='temporal')).eval()
    control_rng = torch.get_rng_state().clone()
    torch.manual_seed(config.seed)
    candidate = TrajectoryDenoiser(config.model).eval()
    assert torch.equal(torch.get_rng_state(), control_rng)
    for name, value in historical.items():
        torch.testing.assert_close(control.state_dict()[name], value, rtol=0, atol=0)
        other = candidate.state_dict()[name]
        if name == 'position_head.weight':
            assert torch.count_nonzero(other[:, config.model.width:]) == 0
            other = other[:, :config.model.width]
        torch.testing.assert_close(other, value, rtol=0, atol=0)
    source = SyntheticDataset(Path(control_manifest['dataset']))
    record = next(r for r in source.records if r['rally_id'] == 'train-00000')
    arrays = source.load(record)
    batch = rally_window(arrays, record, start=0, frames=128, allow_nonconverged=True)
    assert batch.condition.weights.shape == (1, 128, 125)
    rng = torch.Generator().manual_seed(config.seed + 1)
    state = torch.randn((1, 128, 3), generator=rng)
    time_tensor = torch.tensor([.375])
    with torch.no_grad():
        a, b = control(state, time_tensor, batch.condition), candidate(state, time_tensor, batch.condition)
    delta = float((a.positions_norm - b.positions_norm).abs().max())
    torch.testing.assert_close(a.positions_norm, b.positions_norm, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(a.event_logits, b.event_logits, atol=0, rtol=0)
    loss, terms = training_objective(candidate, batch, config.loss, rng, objective='flow')
    loss.backward()
    gradient = candidate.position_head.weight.grad
    assert gradient is not None and bool(torch.isfinite(gradient).all())
    added_gradient = gradient[:, config.model.width:]
    assert torch.count_nonzero(added_gradient) == 384
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.step()
    assert torch.count_nonzero(candidate.position_head.weight[:, config.model.width:]) == 384
    report = dict(
        status='complete', historical_initial_state_sha256=sha256(checkpoint), config_sha256=sha256(config_path),
        model_source_sha256=sha256(project / 'src/tasks/ball_refiner/refiner_3d/diffusion/model.py'),
        historical_parameter_tensors_bitwise_equal=len(historical), rng_state_bitwise_equal=True,
        control_parameters=sum(p.numel() for p in control.parameters()),
        candidate_parameters=sum(p.numel() for p in candidate.parameters()), added_parameters=384,
        new_columns_initially_zero=True, maximum_initial_position_abs_norm_difference=delta,
        event_logits_bitwise_equal=True, real_training_rally=record['rally_id'], rally_sha256=record['npz_sha256'],
        batch_shape=list(batch.condition.weights.shape), all_four_loss_terms={k: float(v.detach()) for k, v in terms.items()},
        nonzero_added_gradient_elements=int(torch.count_nonzero(added_gradient)),
        gradient_norm=float(torch.linalg.vector_norm(added_gradient)), new_columns_updated=True,
        cpu_optimizer_steps=1, trained_model_discarded=True, no_gpu_or_val_or_test=True,
        elapsed_seconds=time.perf_counter()-started,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    )
    (bundle / 'initialization.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
