# Canonical scene dataset pipeline

This package owns the tennis-lab side of the video-to-dataset workflow. One
video and one `scene_id` resolve to one mutable workspace. NHT remains an
independent command that owns reconstruction and rendering; tennis-lab consumes
only its public standard scene export and render files.

## Run

The sole production entrypoint is:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run_scene_pipeline
```

Hydra composition starts at `configs/run_scene_pipeline.yaml`. The typed path
roots, requested dataset targets, camera profile, NHT commands, alignment gates,
and domain policies are all explicit config authority. To rerun a valid
downstream suffix, set `request.from_stage`, for example:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run_scene_pipeline \
  request.from_stage=alignment
```

## Workspace and stages

`SceneWorkspace` resolves the fixed directory
`data/synthetic_data_generation/scenes/<scene_id>/`. Its single `run.json`
records `pending`, `running`, `completed`, `failed`, `invalidated`, and `skipped`
state for this typed DAG:

```text
ingest → reconstruction → alignment
                              ├─ court_dataset ─┐
                              ├─ blcs_dataset  ─┼─ report
                              └─ plcs_dataset  ─┘
```

Each stage has one handler and one owner directory. A rerun validates the
request, retained upstream output, and handler preflight before invalidating the
selected stage and graph-derived descendants. Stage-local output is published
to fixed paths only after semantic validation; a failed attempt removes partial
output and cannot remain `completed`.

## Public reconstruction boundary

`reconstruction/` is the NHT command workspace. tennis-lab invokes
`nht-reconstruct` and `nht-render` as shell-free subprocess argv, then validates
the public schema, files, camera IDs, arrays, coordinate conventions, proper
rotations, intrinsics, shape, dtype, and finite values. It does not import NHT
Python internals or read COLMAP/checkpoint internals.

The configured `nht-reconstruct` and `nht-render` entrypoints are installed
public commands. Their package environment owns PyTorch, gsplat, trainer
selection, and provider defaults; tennis-lab neither locates nor configures
those internals and fails closed when either public command is unavailable.

Alignment uses measured court-line evidence with disjoint fit and holdout
partitions. Only accepted results publish a `MultiCourtLayout` containing every
accepted court, reciprocal metric transforms, complex bounds, and fit/holdout
metrics.

## Dataset domains

The canonical dataset package owns config-driven camera rigs, balanced target-court
assignment, and exact cross-chunk timeline continuity. Task packages provide only
their public domain source contracts.

- Court Detection builds typed 3-D orbit trajectories, deterministic
  coverage selection, uniform arc-length samples, group-disjoint splits,
  attempt-local shards, final semantic labels, and quantitative diagnostics.
- BLCS preserves every source physics frame across multi-object planning,
  config-owned cameras, balanced court assignment, contiguous chunks, labels,
  final assembly, and diagnostics.
- PLCS loads complete ACCAD motion clips, applies SMPL-H and per-frame Gaussian
  LBS, rejects rigid-only motion, composes the full multi-object global timeline,
  and publishes motion/camera/court diagnostics.

The canonical package contains no alternate generic path pipeline, legacy
artifact reader/writer, compatibility conversion, dual-write, fixed-pose
production path, selected captured-camera path, or identity/hash gate.
