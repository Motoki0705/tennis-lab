# Reference-camera clip reconstruction

`reconstruct_reference_clip` connects DINO + BoT-SORT → ViTPose → PLCS and
outsourced ball observations → BLCS. It uses the axial-reference checkpoints,
3–4 camera inputs and camera_view_v2 CourtKP14 semantics. GVHMR/SMPL is optional;
this route saves PLCS canonical joints and court-space joints in `SceneResult`.

## Reproduce the Meiji verification

Run from a checkout containing the axial-reference model implementation, with
`.venv`, `data`, `ckpt` and the initialized DINO external asset available. The
example paths in `../configs/reference_clip.yaml` identify this workstation's
shared assets explicitly; override them when running elsewhere. A compatible
`MultiScaleDeformableAttention` extension must be on the Python import path.
GPU stages must use the repository's shared training queue.

```bash
.venv/bin/python -m src.tennis_scene.scripts.reconstruct_reference_clip stage=observe
.venv/bin/python -m src.tennis_scene.scripts.reconstruct_reference_clip stage=infer
.venv/bin/python -m src.tennis_scene.scripts.reconstruct_reference_clip stage=render
```

`observe` validates `clip.json` and imports `outsource/<camera>_annotations.json`
(`video_ball_annotation.v2`) into `ball_detection_result.json`. Source video
hashes, dimensions, complete frame indices, status values and pixel coordinates
are checked. `observed`, `interpolated` and `occlusion_estimated` are included;
`unresolved` has zero UV/score and false visibility. Original statuses and source
hashes are retained in `ball_import.metadata.json`. The required `score` field
means binary coordinate availability, **not a detector probability**.

The same stage runs the official DINO detector with BoT-SORT, using a ground-plane
playing-area polygon to exclude neighboring courts. ViTPose-H runs on the two
selected completed tracks. Camera-local tracks and poses are cached separately.
Track boxes use the existing interpolation/smoothing behavior; the visual overlay
is needed to audit tracking mistakes and pose confidence. No YOLO detector runs.

`infer` requires all 14 manual court points to be visible and reads the static `annotations/manual_court_kp_result.json`. All cameras
share the clip timeline. Camera-local near/far semantics are declared explicitly
with `view_half_turns`; the Meiji example is `[false, false, true]` and reference
`cam0`. CourtKP14 slots are permuted into the shared reference order. Camera-side
metadata uses approximate pinhole fits to manual court points; the fit error and
estimated intrinsics/extrinsics are recorded, not represented as measured
calibration. These estimates are not model inputs.

This initial association policy targets two-player singles rallies with no end
change: a fixed per-camera ordering by median ankle ground position assigns P1
near cam0 and P2 far cam0. It rejects cases where the selected tracks do not cover
both ends. It does not perform a new per-frame near/far identity sort. Inspect
`player_association_result.json` and the evaluation movie for identity continuity.

Inference samples every second source frame (59.94 → 29.97 fps), uses 128-frame
windows with 64-frame overlap, and blends overlapping predictions. It restores
the original timeline by linear interpolation of positions and canonical joints,
and circular heading interpolation; the final unobserved half-step is held.
Neither out-of-court positions nor predicted ball heights are clamped.

## Artifacts

- `scene.npz` and mandatory `scene.metadata.json`: original-timeline court-space
  positions, yaw, canonical joints, world joints, 2D observations, checkpoint
  hashes and reference provenance. Unavailable SMPL fields are absent rather
  than fabricated zero-valued parameters.
- `reference_context.json`: camera ordering, near/far permutation, approximate
  camera fits, and model contract.
- `evaluation.mp4`: 1920×1080 H.264, original playback speed. Left: predicted 3D
  skeletons/ball/court plus front/side pose details at a fixed metric scale.
  Pose details retain absolute ankle heights above the ground line. Right: the three source cameras with 2D pose/ball inputs
  and the original ball observation status. This is qualitative validation;
  the clip does not supply 3D ground truth.
- `evaluation_preview.jpg`, `video_receipt.json`, per-stage configuration copies.

The three-column source-video layout currently requires exactly three cameras;
the model input contract supports three or four.
