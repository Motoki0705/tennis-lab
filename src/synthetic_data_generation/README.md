# Canonical scene dataset pipeline

This package owns the tennis-lab video-to-Court-Detection-dataset workflow. One
video and one `scene_id` resolve to one mutable workspace. NHT remains an
independent command that owns reconstruction and rendering; tennis-lab consumes
only its public standard scene export and render files.

For reference-guided image edits and retraining with existing SfM, see
[Appearance variants](appearance/README.md).

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

The scene-pipeline production entrypoint is:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run_scene_pipeline
```

Hydra composition starts at `configs/run_scene_pipeline.yaml`. The typed path
roots, requested dataset targets, start/terminal stages, NHT
commands, alignment gates, and domain policies are all explicit config authority.
To rerun a valid downstream suffix, set `request.from_stage`, for example:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run_scene_pipeline \
  request.from_stage=alignment
```

An alignment rerun may change `alignment`, `dataset`, and `request.targets`,
because alignment and all dataset/report descendants are invalidated. Retained
upstream authority (including roots, profile, pipeline, and existing NHT
values) must still match. Legacy NHT configuration may add only the nonempty
`training_python_path` and `trainer_path` runtime paths; replacing existing values
or adding other NHT options is rejected before any publication is invalidated.

To stop after court alignment without constructing or running dataset/report
handlers, set the explicit terminal stage:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run_scene_pipeline \
  profile=b01 request.from_stage=ingest request.through_stage=alignment
```

The selected plan is the requested terminal stage and its dependency closure.
For `through_stage=alignment`, it is exactly `ingest → reconstruction →
alignment`; any stale dataset/report descendants are unpublished and no new
`datasets/` or `report/` owner is created.

## Workspace and stages

`SceneWorkspace` resolves the fixed directory
`data/synthetic_data_generation/scenes/<scene_id>/`. Its single `run.json`
records `pending`, `running`, `completed`, `failed`, `invalidated`, and `skipped`
state for this typed DAG:

```text
ingest → reconstruction → alignment → court_dataset → report
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

Each `nht-render` invocation loads the checkpoint and shader once and processes
its camera request in bounded GPU batches (default four cameras). Resolution
changes split a batch without changing request order. Court generation already
submits a camera shard per invocation, so every shard reuses one loaded scene.
The batch CLI and shader constraints belong to [NHT's README](../../third_party/nht/README.md).

Alignment uses measured court-line evidence with disjoint fit and holdout
partitions. Before detection, the complete fixed camera prefix is rendered from
the learned 3DGS by the public `NHTRenderClient` boundary using the observed
`StandardSceneExport` camera IDs. The detector and ground-plane projection are
therefore bound to the same pose, intrinsics, resolution, pixel convention, and
pre-alignment NHT scene coordinates. Rendering failure aborts alignment;
captured camera images are never substituted. The durable inference cache
separates captured and NHT-rendered sources and fingerprints the
scene/checkpoint, render policy, complete camera geometry, exact detector-input
RGB, and detector model.

The fixed camera prefix and its immutable partition units are chosen without
assuming a court count. Fit views alone form a common weighted ground grid;
bounded residual search adds regulation-court candidates only while each one
explains the configured minimum fraction of weighted evidence. Every
proposal and common-scale refinement uses the same probability-and-proximity
weighted coverage-floor objective, which cannot trade away whole-template
coverage for a small high-confidence fragment. Bounded beam frontiers are
retained by candidate count. After search termination they are refined from the
smallest count upward, and
the first fully valid frontier whose final weighted residual satisfies the gate
is selected. A terminal frontier may retain residual clutter only when no
reliable additional proposal exists. The search fails closed for zero courts or
an exhausted configured maximum without a valid refined frontier. Holdout views
are evaluated once after the positive court count is frozen and never drive
count reselection. Only accepted results publish a `MultiCourtLayout`
containing every accepted court,
reciprocal metric transforms, complex bounds, and fit/holdout metrics. The
alignment owner also publishes `line-heatmaps/`: the exact RGB passed to the
detector, its raw detector heatmap, and its proximity-weighted ground-plane
heatmap for every selected view, plus their weighted aggregate on one common
ground grid. The numeric archive binds every input by a pixel-value SHA-256 and
is the validation authority for the PNG diagnostics.

## Correct court alignment manually

[Court Alignment Studio](alignment/manual/README.md) edits court position, yaw,
shared scale and court count on the projected heatmap, with human confirmation
as the final authority for downstream datasets.

## Court detection dataset

This system generates only Court Detection datasets. The versioned contract,
labels and storage layout are documented in the
[Court Detection dataset v1/v2/v3 contract](dataset/court/README.md).
BLCS/PLCS RGB generation, dynamic foreground composition and their configuration
selectors have been removed. `request.targets=[court]` is the only dataset target.

## Visualization and publication

`src.synthetic_data_generation.scripts.visualize_dataset` renders the selected
Court trajectory. `generate_publication_visualizations` publishes one validated
Court bundle: `dataset-court.gif`, `alignment-progression.gif`,
`alignment-heatmap-court.png`, `captured-camera-trajectory.png`,
`publication-overview.png`, and `manifest.json`.
The publication request, manifest and bundle schemas are version 2. The request
specifies Court trajectory/frame indices, captured camera IDs and drawing settings.
The overview contains Court, alignment and captured-camera panels.

## Court LINE推論の共有

新しいalignment実行は、保存モデル構成を読む共通Court predictorと、その[配布checkpoint](../tasks/court_detection/README.md#共通推論と幾何補正)を使います。LINE確率のnative gridを既存の地面投影・集約へ渡すため、複数コートの観測を保持します。Hへの置換やKPの向き推定はこの経路では行いません。旧LINE専用の手書きモデル再構築・architecture設定は廃止しました。

raw LINE cacheの配列形式は維持し、loaded checkpoint・backbone・保存head仕様・short-side・device・seedを新しいdetector identityへ含めます。checkpointを切り替えても旧cache・保存alignment・manual ownerを削除しません。alignmentを保持してdataset以降だけ再実行する場合は、旧line-modelの設定を保存済み`resolved-config.yaml`から引き継ぎます。検出器の既定変更を許容しても、projection/fit設定やLINE抽出条件の変更を暗黙に適用することはありません。

既存B00/B01/B03の`alignment_line_heatmaps_v2`を現行v3専用loaderで読めない制約は、このモデル移行以前からあります。今回その内容をv3へ書き換えたり、不足している入力画像の出典を補ったりしません。
