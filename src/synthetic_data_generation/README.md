# Canonical scene dataset pipeline

This package owns the tennis-lab side of the video-to-dataset workflow. One
video and one `scene_id` resolve to one mutable workspace. NHT remains an
independent command that owns reconstruction and rendering; tennis-lab consumes
only its public standard scene export and render files.

## Set up NHT with spin

Before running the pipeline for the first time, or after updating the NHT
submodule reference, install NHT's public commands with the project development
CLI:

```bash
.venv/bin/python -m spin setup-nht
```

This checks out the pinned `third_party/nht` submodule commit and installs NHT
and its public rendering runtime as an editable, isolated `uv tool`. It also
rebuilds `third_party/nht/.trainer-venv` as the dedicated gsplat example-trainer
runtime and verifies that the trainer can be imported there. The two NHT
environments are separate because the public SfM pipeline requires current
`pycolmap`, while the trainer's dataset loader requires the pinned legacy
`SceneManager` API. Neither dependency set is added to the tennis-lab `.venv`,
and an existing unmanaged `third_party/nht/.venv` is not used.

To install the optional learned SfM retry backend, run
`.venv/bin/python -m spin setup-nht --with-sfm-learned` instead. The command
validates that `nht-reconstruct` and `nht-render` resolve from the installed
tool's bin directory and reports the required `PATH` update if they do not.

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
public commands. Their package environment owns public-command dependencies and
rendering. The machine-local trainer Python and trainer entrypoint are explicit
typed configuration resolved from the pinned NHT submodule, then bound to a
temporary copy of the public NHT pipeline config for each reconstruction.
tennis-lab still imports no NHT Python internals and fails closed when a public
command or the dedicated trainer runtime is unavailable.

Alignment uses measured court-line evidence with disjoint fit and holdout
partitions. Only accepted results publish a `MultiCourtLayout` containing every
accepted court, reciprocal metric transforms, complex bounds, and fit/holdout
metrics.

## Dataset domains

The canonical dataset package owns config-driven camera rigs, balanced target-court
assignment, and exact cross-chunk timeline continuity. Task packages provide only
their public domain source contracts.

- [Court Detection dataset v1/v2/v3 contract](dataset/court/README.md).
- BLCS preserves every source physics frame across multi-object planning,
  config-owned cameras, balanced court assignment, contiguous chunks, labels,
  final assembly, and diagnostics.
- PLCS loads complete ACCAD motion clips, applies SMPL-H and per-frame Gaussian
  LBS, rejects rigid-only motion, composes the full multi-object global timeline,
  and publishes motion/camera/court diagnostics.

The canonical package contains no alternate generic path pipeline, legacy
artifact reader/writer, compatibility conversion, dual-write, fixed-pose
production path, selected captured-camera path, or identity/hash gate.

## Visualize a generated dataset

The production visualizer reads only the published current-schema owner under
`<scene-root>/datasets/{court,blcs,plcs}`. It validates the current schema and
the selected view's exact canonical inventory, reconstructs compact BLCS/PLCS
RGB views from their NHT background and foreground-delta stores, streams frames
into H.264, and writes a deterministic JSON sidecar beside the MP4. Existing
output files are never overwritten.

Canonical frames with odd dimensions are preserved and padded by one black pixel
on the right and/or bottom for H.264 `yuv420p`; the source dimensions and exact
padding are recorded in the sidecar.

Court visualization follows the versioned Court dataset contract linked above:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.visualize_dataset \
  visualization.domain=court \
  visualization.dataset_root=scenes/<scene_id>/datasets/court \
  visualization.trajectory_id=<trajectory_id> \
  visualization.output_video=previews/court-orbit.mp4
```

BLCS and PLCS selection is one explicit logical scene and generated camera.
For BLCS the logical scene ID is the canonical trajectory ID. BLCS overlays
stable ball identity, presence, renderer observation, and a short trajectory;
PLCS overlays CourtKP20 context plus projected COCO17 skeletons, stable person
identity, physical presence, and renderer-visible pixel state.

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.visualize_dataset \
  visualization.domain=blcs \
  visualization.dataset_root=scenes/<scene_id>/datasets/blcs \
  visualization.logical_scene_id=<trajectory_id> \
  visualization.camera_id=<camera_id> \
  visualization.output_video=previews/blcs-view.mp4

.venv/bin/python -m src.synthetic_data_generation.scripts.visualize_dataset \
  visualization.domain=plcs \
  visualization.dataset_root=synthetic_data_generation/scenes/B00/datasets/plcs \
  visualization.logical_scene_id=B00 \
  visualization.camera_id=court-001-corner-near-left \
  visualization.output_video=previews/plcs-view.mp4
```

`visualization.fps`, `crf`, and BLCS `history_frames` are explicit Hydra
settings. Dataset and video paths are resolved strictly beneath `roots.data_root`
and `roots.output_root`, respectively. IDs are never guessed, camera views are
never substituted, frame gaps/reordering fail closed, and no compatibility-schema
fallback exists.
