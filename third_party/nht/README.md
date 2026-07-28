# Isolated Neural Harmonic Textures runtime

This tracked directory owns the reproducible NHT checkout, CUDA environment, smoke
gate, and training launcher used by the 3DGS-native synthetic-data pipeline.
It does not reuse tennis-lab's `.venv` and does not copy the already-built
environment from `/home/kamimura/projects/gaussian-splating`.

The integration is pinned to:

- Neural Harmonic Textures `7de4cc07ba7f81ce90f7bd90f76ff0260c00c3d0`
- NHT's gsplat branch `20bc323d613258e5d169fdbc962c9ef27d55ca69`
- GLM `e7970a8b26732f1b0df9690f7180546f8c30e48e`
- Python 3.11, Torch 2.9.1 + CUDA 13.0
- Blackwell `sm_120+PTX`

All additional Git dependencies are content-pinned in `requirements.in`.
`pins.env` is the single declaration of checkout and runtime versions.

## Setup

The first build may take more than one scheduled research cycle because gsplat
and tiny-cuda-nn compile CUDA extensions.

```bash
third_party/nht/setup.sh
```

To seed Git objects from the completed reference checkout without sharing its
working tree or environment:

```bash
NHT_SEED_REPOSITORY=/home/kamimura/projects/gaussian-splating/third_party/neural-harmonic-textures \
  third_party/nht/setup.sh
```

The clone uses `--dissociate`, so the resulting checkout is independent. A
successful setup publishes the finite forward/backward report at
`third_party/nht/artifacts/smoke.json` and a resolved package inventory beside
it. These local build artifacts are intentionally ignored by Git.

## Train

The input is a COLMAP dataset containing `sparse/0` and a native
`images_<factor>` directory. The output directory must be absent or empty.

```bash
third_party/nht/train.sh \
  --data-dir /absolute/path/to/colmap-scene \
  --result-dir /absolute/path/to/results \
  --data-factor 2 \
  --max-steps 30000 \
  --cap-max 1000000
```

The launcher refuses a modified or unpinned checkout, records the exact command
and environment in `nht-run.json`, and never overwrites a non-empty run.
Additional upstream flags must be passed explicitly as repeatable
`--trainer-arg=<value>` arguments.

This runtime is a third-party boundary. Tennis-lab modules must communicate
with it through versioned files or subprocess requests and must not import its
packages in the main `.venv`.

## Native composition smoke

After export, verify that background and movable Gaussian tensors are composed
before one NHT rasterization call:

```bash
third_party/nht/.venv/bin/python third_party/nht/composition_smoke.py \
  --checkpoint /absolute/path/to/ckpt.pt \
  --provider-bundle /absolute/path/to/provider-bundle \
  --output-dir /absolute/new/path/to/composition-smoke
```

The smoke refuses replacement, requires an identical deferred-appearance
identity for every asset, and publishes RGB, alpha, depth, instance visibility,
the versioned composition manifest, and exact tensor/provenance artifacts.

## BLCS sequence rendering

`blcs_render.py` is the isolated consumer of
`tennis_blcs_gaussian_scene_plan_v2`. It verifies the plan, embedded asset
registry, every local Gaussian artifact, background composition, shared
appearance identity, and pinned clean gsplat checkout before CUDA work.

```bash
third_party/nht/.venv/bin/python third_party/nht/blcs_render.py \
  --plan-dir /absolute/path/to/blcs-plan \
  --background-composition /absolute/path/to/composition.json \
  --output-dir /absolute/new/path/to/render-output \
  --camera-id frame_000080 \
  --frame-indices 0,15,30,44,45,59 \
  --width 480
```

Each selected frame composes the background with only the active persistent
instances and performs two public `rasterization(...)` calls over the same
Gaussian scene:

1. NHT `RGB+ED` features are passed through the deferred shader for RGB and
   expected depth. This call internally uses NHT's documented eval3d auxiliary
   depth pass.
2. A regular eval3d call renders a one-hot channel for background and every
   persistent instance without changing the deferred appearance features.

The second call publishes exact alpha-composited contribution AOVs,
thresholded per-instance masks, and exclusive instance segmentation as
`instance_contribution.npy`, `instance_mask.npy`, and
`instance_segmentation.npy`. The worker rejects the frame if AOV contributions
do not sum to AOV alpha or if AOV alpha drifts from the NHT pass beyond the
recorded float32 tolerance. The previous projected-centre depth-consistency
method remains in labels only as a comparison proxy. Publication is atomic and
refuses an existing output directory.

## Generated prototype ball

`prototype_ball_fixture.py` constructs a centred 6.7 cm Gaussian sphere without
a user asset. The antipodally symmetric Fibonacci shell uses isotropic
Gaussians whose maximum three-sigma envelope is the declared ball radius. It
renders eight view-distinct calibration targets through the frozen background
shader and publishes an independent-NHT source plus an asset-preparation spec.
The manifest explicitly declares `asset_origin=codex-generated-prototype` and
`source_is_user_asset=false`.

```bash
third_party/nht/.venv/bin/python third_party/nht/prototype_ball_fixture.py \
  --target-appearance /absolute/path/to/background-appearance.pt \
  --output-dir /absolute/new/path/to/prototype-ball
```

The resulting `capture/capture.json`, `prototype-source.pt`, and
`asset-spec.json` are consumed by `ball_calibration_import.py` and
`prepare_ball_assets.py`. This path is intended for prototype acceptance; it
does not claim a photogrammetric user-ball capture.

`prototype_blcs_plan.py` then verifies the export, user-approved alignment and
all decision evidence before running the repository's `BallPhysics` /
`RallySimulator`. It publishes the complete single- or multi-ball trajectory
plan plus simulator configuration and event provenance:

```bash
/absolute/path/to/tennis-lab/.venv/bin/python \
  third_party/nht/prototype_blcs_plan.py \
  --registry /absolute/path/to/registry.json \
  --provider-bundle /absolute/path/to/provider-export \
  --scene-contract /absolute/path/to/approved-scene-contract.json \
  --scene-contract-root /absolute/path/to/contract-repository-root \
  --mode single \
  --output-dir /absolute/new/path/to/prototype-plan
```

Plan generation deliberately uses tennis-lab's main environment because the
physical generators depend on the task stack. CUDA rendering continues to use
only `third_party/nht/.venv`; the two environments exchange the strict plan
files and never import each other's packages.

After producing single/multi plans and renders,
`prototype_p3_acceptance.py` reloads the export and accepted alignment, verifies
the generated-prototype provenance and metric envelope, checks a same-seed
byte-identical plan repeat, rehashes every render artifact, and compares exact
instance masks, positions, and projections with the plan:

```bash
/absolute/path/to/tennis-lab/.venv/bin/python \
  third_party/nht/prototype_p3_acceptance.py \
  --provider-bundle /absolute/path/to/provider-export \
  --scene-contract /absolute/path/to/approved-scene-contract.json \
  --scene-contract-root /absolute/path/to/contract-repository-root \
  --prototype-dir /absolute/path/to/prototype-ball \
  --registry /absolute/path/to/registry.json \
  --single-plan /absolute/path/to/single-plan-run \
  --single-plan-repeat /absolute/path/to/repeated-single-plan-run \
  --multi-plan /absolute/path/to/multi-plan-run \
  --single-render /absolute/path/to/single-render \
  --multi-render /absolute/path/to/multi-render \
  --output /absolute/new/path/to/p3-acceptance.json
```

## Frozen-target ball feature fitting

`ball_feature_fit.py` is the explicit conversion boundary for a standard INRIA
3DGS PLY or an independently trained NHT tensor pack. It copies means, log
scales, and opacity logits, and deterministically normalizes standard PLY
quaternions once before freezing geometry. It initializes a new feature tensor,
freezes the target deferred shader, and optimizes only those new per-Gaussian
features. Independent source features and vanilla SH coefficients are never
silently treated as target-NHT features.

Calibration is a versioned immutable directory containing `manifest.json` and
`calibration.npz`. The latter has exact `camera_to_asset [V,4,4]`,
`intrinsics [V,3,3]`, `rgb [V,H,W,3]`, `mask [V,H,W]`, and `split [V]`
arrays. At least one foreground-containing train view and validation view are
required. Camera matrices use OpenCV camera-to-asset axes.

```bash
third_party/nht/.venv/bin/python third_party/nht/ball_feature_fit.py \
  --source /absolute/path/to/ball.ply \
  --source-format vanilla_3dgs_ply_v1 \
  --calibration-bundle /absolute/path/to/calibration \
  --target-appearance /absolute/path/to/appearance.pt \
  --target-appearance-space-sha256 <64-lowercase-hex> \
  --output-dir /absolute/new/path/to/prepared-ball
```

The output contains the exact six-key NHT tensor pack, a strict
`tennis_ball_asset_conversion_report_v1` accepted by the BLCS ingestion
boundary, optimization history, renderer commits, and validation diagnostics.
The default gate is masked validation PSNR >= 20 dB and cannot be lowered.
Geometry, opacity, and target shader are reported as frozen. The worker verifies
the pinned clean NHT/gsplat checkouts before CUDA work and atomically refuses
replacement.

`ball_feature_fit_fixture.py` exists only to generate a non-user integration
fixture with teacher RGB/masks, an independent-NHT source, and a degree-zero
standard PLY. Passing this fixture proves mechanics and report compatibility;
it does not satisfy the production user-ball asset gate.

### Importing real calibration captures

Real calibration data enters through `ball_calibration_import.py`, not by
assembling NPZ arrays manually. Its `tennis_ball_calibration_capture_v1` JSON
contains at least two train views and one validation view. Every view declares
a unique ID, split, common dimensions, exact RGB/mask file hash and size,
OpenCV `camera_to_asset` matrix, and intrinsics. RGB files must have Pillow mode
`RGB`; masks must have mode `L` and contain only 0/255 with a non-empty,
non-full foreground. Improper rotations, unsafe relative paths, duplicate RGB
content (including split leakage), altered bytes, and missing views fail before
publication.

```bash
third_party/nht/.venv/bin/python third_party/nht/ball_calibration_import.py \
  --capture-manifest /absolute/path/to/capture/capture.json \
  --bundle-id user-ball-calibration-v1 \
  --output-dir /absolute/new/path/to/calibration-import
```

The atomic output contains `capture-import.json` and
`bundle/{manifest.json,calibration.npz}`. Loading the import re-verifies the
source capture manifest, all source images/masks, import fingerprint, and
bundle bytes.

### Source-to-registry preparation

`prepare_ball_assets.py` accepts one or more `--asset-spec` files and publishes
one multi-variant BLCS registry. Each
`tennis_ball_asset_preparation_entry_v1` spec declares the variant/asset IDs,
5--9 cm nominal diameter, source format, exact source `ArtifactRef`, explicit
`asset_from_prepared` Sim(3), and a boolean `source_is_user_asset`; that status
is never inferred from a path or name.

```bash
third_party/nht/.venv/bin/python third_party/nht/prepare_ball_assets.py \
  --asset-spec /absolute/path/to/yellow-ball.json \
  --asset-spec /absolute/path/to/green-ball.json \
  --calibration-import /absolute/path/to/calibration-import \
  --background-composition /absolute/path/to/composition.json \
  --registry-id user-ball-assets-v1 \
  --output-dir /absolute/new/path/to/preparation-run
```

All specs, calibration source bytes, background/appearance bytes, and options
are verified before the output root or CUDA work is created. Each feature-fit
child and the registry publish atomically. A successful run records requests,
worker identities, logs, conversion reports, and the final registry
fingerprint. If a later stage fails, the immutable run root deliberately keeps
`failure.json`, completed fit children, and logs; it never claims or publishes
a registry as successful. Reusing any output root is rejected.

## PLCS avatar NHT fitting

`plcs_avatar_fit.py` consumes the hash-verified SMPL-X Gaussian fixture built by
`third_party/plcs_avatar/build_asset_fixture.py`. It creates teacher captures
with explicit target-NHT features, reinitializes the asset features to zero,
optimizes only those features with the frozen NHT shader, and validates held-out
views. It then applies the identical learned feature tensor to every emitted
SMPL-X pose and renders each pose natively. Standard 3DGS features and RGB
overlays are never used.

```bash
third_party/nht/.venv/bin/python third_party/nht/plcs_avatar_fit.py \
  --fixture /absolute/path/to/plcs-avatar-geometry \
  --target-appearance /absolute/path/to/appearance.pt \
  --appearance-space-sha256 <64-lowercase-hex> \
  --output /absolute/new/path/to/plcs-avatar-nht
```

The atomic output includes canonical and per-pose NHT tensor packs, RGB/alpha
renders, held-out target/prediction diagnostics, pose visibility and change
metrics, exact upstream revisions, and explicit empty dropped-joint/frame
inventories.

## PLCS native sequence rendering

`plcs_render.py` consumes only the strict PLCS plan and the versioned background
composition across the main/NHT environment boundary. For each selected frame
it transforms the requested pose tensor for every persistent identity,
concatenates all persons with the background, and performs one NHT RGB+ED pass
plus one eval3d instance-contribution pass. Output publication is atomic and
contains RGB, alpha, expected depth, exact masks, segmentation, complete
identity/pose/placement/projection labels, renderer commit, and hashes.

```bash
third_party/nht/.venv/bin/python third_party/nht/plcs_render.py \
  --plan-dir /absolute/path/to/plcs-plan \
  --background-composition /absolute/path/to/composition.json \
  --output-dir /absolute/new/path/to/plcs-render \
  --frame-indices 0,4,6,8,11 \
  --width 480
```
