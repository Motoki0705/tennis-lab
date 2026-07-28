# 3DGS-native synthetic-data generation

This package builds training datasets by composing movable 3D Gaussian assets
with one reconstructed background before native NHT rasterization. It does not
overlay independently rendered RGB images.

## Ownership

```text
synthetic_data_generation/
├── alignment/          # SfM export and accepted court alignment
├── composition/        # Renderer-independent Gaussian contracts and transforms
├── dataset/            # All current and future dataset generators
│   ├── registry.py     # The single built-in dataset registry
│   ├── pipeline.py     # Immutable plan and stage contracts
│   ├── blcs/
│   ├── plcs/
│   └── court/
├── rendering/nht/      # Pinned subprocess boundary for the NHT environment
├── reporting/          # Read-only previews and release reports
├── validation/         # Cross-dataset release validation
├── scripts/
│   ├── alignment/
│   └── dataset/
└── configs/
    ├── alignment/
    └── dataset/
```

Each directory below `dataset/` owns its `artifacts`, `components`, `pipeline`,
`validation`, `rendering`, and `reporting` concerns. A new dataset is added as
`dataset/<name>/` and registered once in `dataset/registry.py`.

## Pipeline entry point

Build a reviewable plan without executing it:

```bash
python -m src.synthetic_data_generation.scripts.dataset.run_pipeline \
  domain=blcs \
  plan_path=/absolute/new/path/blcs-plan.json
```

Select another dataset with `domain=plcs` or `domain=court`. Set
`execute=true` only after enabling and configuring the desired stages. The
runner refuses to overwrite a plan or execution output.

All entry-point configuration lives in
`src/synthetic_data_generation/configs/dataset`. Stage modules and runtime
ownership are fixed by code; Hydra controls algorithms, inputs, outputs, and
whether a stage is enabled.

## Configurable algorithms

Unknown names fail with the complete list of valid choices. There is no
fallback.

| Dataset | Config key | Algorithms |
|---|---|---|
| BLCS | `algorithms.ball_asset` | `procedural_fibonacci`, `registered_gaussian_asset` |
| BLCS | `algorithms.trajectory` | `rally_physics` |
| PLCS | `algorithms.avatar_control` | `gaussianavatar_query_lbs`, `hugs_topk_lbs` |
| PLCS | `algorithms.motion` | `seeded_court_motion` |
| court | `algorithms.camera_sampling` | `sfm_neighborhood`, `inward_orbit` |
| court | `algorithms.labels` | `symmetric_seven_channel` |

Candidate algorithms remain named implementations selected by configuration.
Development phase numbers and `prototype_*` file names are not part of the
production API.

## NHT boundary

`third_party/nht` is a Git submodule and owns only its independent Python/CUDA
environment, training, checkpoint loading, deferred shader, and rasterizer.
It contains no BLCS, PLCS, court, acceptance, or reporting code.

`rendering/nht/process.py` verifies the exact submodule commit and clean tracked
state, then executes a project-owned worker with:

```text
tennis-lab pipeline
  -> shell-free subprocess
    -> third_party/nht/.venv/bin/python
      -> src.synthetic_data_generation.<project-owned worker>
```

The main environment does not import `gsplat`. NHT does not import tennis-lab
domain code from its own repository. `PYTHONPATH` is supplied only to the
isolated child process.

## Export-first verification

The alignment export is the immutable source of cameras, images, point cloud,
and source hashes. Dataset plans reference the accepted scene contract and are
rendered only after their input identities are verified. Unit tests mirror the
package structure; Hydra composition belongs to integration tests; pipeline
entry points belong to end-to-end tests.
