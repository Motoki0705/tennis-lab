# PLCS synthetic-dataset generation

This directory is the isolated, reproducible evaluation boundary for candidate
SMPL-driven Gaussian-avatar control methods. It does not vendor upstream code
or reinterpret upstream spherical-harmonic features as NHT features.

Run from the worktree with the repository environment:

```bash
PYTHONPATH=$PWD /home/kamimura/projects/tennis-lab/.venv/bin/python \
  -m src.synthetic_data_generation.dataset.plcs.validation.avatar_control_comparison \
  --model /home/kamimura/projects/tennis-lab/third_party/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz \
  --output .codex-loop/3dgs-synthetic-data/artifacts/cycle-10/plcs-control-probe-v2
```

The probe compares fixed per-Gaussian SMPL-X weights modelled after
GaussianAvatar with HUGS-style top-k vertex transform blending. Mesh
attachments on the same posed SMPL-X sequence are the geometric reference.
This is an early screening result, not an appearance or release acceptance claim.

The selected GaussianAvatar-style fixture is built in the main tennis-lab
environment because it reads the licensed GVHMR SMPL-X model. It emits only
derived Gaussian tensors and control arrays; it does not copy the model:

```bash
PYTHONPATH=$PWD /home/kamimura/projects/tennis-lab/.venv/bin/python \
  -m src.synthetic_data_generation.dataset.plcs.components.avatar_asset_builder \
  --model /home/kamimura/projects/tennis-lab/third_party/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz \
  --target-appearance .codex-loop/3dgs-synthetic-data/artifacts/cycle-02/nht-composition-smoke-v3/appearance.pt \
  --output .codex-loop/3dgs-synthetic-data/artifacts/cycle-11/plcs-avatar-geometry-v1
```

The hash-verifying diagnostic renderer produces a sparse 3D overlay for manual
inspection:

```bash
PYTHONPATH=$PWD /home/kamimura/projects/tennis-lab/.venv/bin/python \
  -m src.synthetic_data_generation.dataset.plcs.reporting.avatar_control_comparison \
  --artifact .codex-loop/3dgs-synthetic-data/artifacts/cycle-10/plcs-control-probe-v2 \
  --output .codex-loop/3dgs-synthetic-data/artifacts/cycle-10/plcs-control-probe-v2-preview.png
```

Avatar construction is accepted only by `validation/avatar_asset.py`, which verifies the paper/code pins,
byte-repeated SMPL-X asset construction, two independent NHT fits, native
pose-render labels, and a measured CUDA repeat tolerance. It does not turn
GPU nondeterminism into a byte-identity claim.

## Single/multi-person scene plans

`components/scene_plan_builder.py` binds the selected avatar to the verified export,
accepted court transform, and NHT background. It emits a deterministic
12-frame single- or multi-person plan with stable identity, court placement,
yaw, pose, Sim(3), and selected-camera projection labels. It refuses a changed
export, avatar acceptance report, pose tensor, appearance space, or background provider.

The plan is rendered by `src.synthetic_data_generation.dataset.plcs.rendering.nht`; no RGB overlay is
used. `validation/dataset.py` then checks byte-identical plan/render repeats, exact
AOV masks and bboxes, all label fields, stable identities, velocity
reconstruction, nontrivial path/pose control, and projection consistency.
The dataset thresholds are at least 50 exact visible pixels per
person/frame, at most 3 px between projected root and mask centroid, NHT/AOV
alpha drift at most 0.005, and contribution/alpha drift at most `1e-5`.

The generated diagnostic is an RGB panel with exact instance masks overlaid:

```bash
/home/kamimura/projects/tennis-lab/.venv/bin/python \
  -m src.synthetic_data_generation.dataset.plcs.reporting.dataset_preview \
  --render-root /absolute/path/to/plcs-render \
  --output /absolute/new/path/to/plcs-diagnostic.png
```
