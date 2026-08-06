# Scene-based synthetic-data generation

The production entry point owns one mutable workspace per `scene_id` and runs
the following typed DAG:

```text
ingest → reconstruction (NHT subprocess) → alignment
                                           ├─ court_dataset
                                           ├─ blcs_dataset
                                           └─ plcs_dataset
                                                    ↓
                                                  report
```

SfM and NHT training are owned by `neural-harmonic-textures`. This package does
not import its Python modules or inspect COLMAP/checkpoint internals. It invokes
the configured `nht-reconstruct` command, validates the standard scene export,
and invokes `nht-render` for observed-camera RGB, alpha, and depth.

## Canonical workspace

The configured data root and `scene_id=B00` resolve to exactly:

```text
data/synthetic_data_generation/scenes/B00/
├── run.json
├── resolved-config.yaml
├── source/
│   ├── video.mp4
│   └── metadata.json
├── reconstruction/                 # NHT-owned workspace
│   ├── run.json
│   └── export/
│       ├── scene.json
│       ├── cameras.json
│       ├── points_scene.npy
│       ├── images/
│       └── model/
├── alignment/
│   ├── ground-line-map.npz
│   ├── ground-line-preview.png
│   ├── court-geometry.json
│   ├── alignment.json
│   └── diagnostics/fit-holdout.json
├── datasets/
│   ├── court/dataset.json
│   ├── blcs/dataset.json
│   └── plcs/dataset.json
├── report/index.html
└── logs/<stage>/attempt-<n>.log
```

Paths never contain timestamps, hashes, fingerprints, or Git commits. The
single top-level `run.json` records the current stage statuses, attempts,
fixed outputs, summaries, errors, seed, request, and the nested NHT run path.

## Run from a video

Configure roots and the two independent commands in
`configs/pipeline/scene.yaml`, then run:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run \
  scene_id=B00 \
  input_video=relative/to/external_asset_root/tennis.mp4 \
  pipeline_config=src/synthetic_data_generation/configs/pipeline/scene.yaml \
  from_stage=ingest \
  'targets=[court,blcs,plcs]'
```

The input video is copied into the scene workspace. Later stages never depend
on its original external path. Hydra owns invocation composition; the strict
runtime adapter resolves `pipeline_config` against the project root and
`input_video` against the selected pipeline config's `external_asset_root`.

## Rerun and invalidation

Rerun alignment and all requested descendants while preserving source and NHT:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run \
  scene_id=B00 \
  from_stage=alignment \
  'targets=[court,blcs,plcs]'
```

Rerun NHT from SfM and replace every downstream result:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run \
  scene_id=B00 \
  from_stage=reconstruction \
  nht_from_stage=sfm \
  'targets=[court,blcs,plcs]'
```

Before a stage starts, its canonical output and all DAG descendants are
unpublished. Stage work is published from `.staging` only after semantic
validation. A failure remains `failed` in `run.json`; stale outputs are not
restored or reported as completed. A live per-scene lock rejects concurrent
writes, while a dead lock and a `running` record are recovered on the next run.

Configuration changes also choose the earliest affected stage automatically:
NHT settings invalidate reconstruction; alignment or seed settings invalidate
alignment; dataset settings invalidate the common alignment descendant path.

## Contracts

- NHT input boundary: workspace `source/video.mp4`, `scene_id`, NHT config, and
  the fixed `reconstruction/` workspace.
- NHT output boundary: `nht_standard_scene_v1`,
  `nht_standard_cameras_v1`, finite `points_scene.npy`, images, and model.
- Coordinate boundary: NHT canonical scene space to court metres through
  `alignment/scene_from_court` and its inverse.
- Alignment boundary: fit views produce a ground-projected achromatic court-line
  raster and one metric ITF template; disjoint holdout views evaluate the fixed
  transform. A holdout view enters the accepted-view denominator only when the
  aligned target-court ROI contains enough projected evidence; at least one
  independently held-out view must be evaluable. Ground support, camera-side,
  template-score, held-out view, and held-out line-inlier gates must all pass
  before datasets can start. The
  `sparse_control` evidence mode is explicit and reserved for CPU orchestration
  tests; the production config uses `image_achromatic`.
- Renderer boundary: independent `nht-render`, returning
  `nht_render_result_v1` RGB/alpha/depth records. Every requested camera ID,
  relative path, resolution, float32 shape, finite value and numeric range is
  checked before a dataset is published.
- Domain boundary: fixed `dataset.json` plus sample renders and labels beneath
  each of `datasets/court`, `datasets/blcs`, and `datasets/plcs`. BLCS and PLCS
  add deterministic procedural ball/player layers to the standard NHT
  background and publish uint8 instance masks; labels and rendered arrays use
  the same court-to-scene transform and camera projection. Camera sampling
  requires the domain target points to be in-frame, and production refuses to
  publish an empty instance mask.

These are semantic contracts. SHA-256 identity, content fingerprints, Git
revision checks, clean-worktree checks, immutable publication, and overwrite
refusal are not part of this production pipeline.

## Production evidence

The checked-in [B00 production evidence](evidence/2026-08-06-b00-production.json)
records the real `data/tennis_court.mp4` run, accepted SfM and alignment metrics,
all three generated domain datasets, and the observed invalidation behavior for
alignment, SfM, and NHT-training reruns. The evidence contains repository-
relative paths and replayable command arguments; machine-local model and tool
paths remain in the generated workspace manifests only.
