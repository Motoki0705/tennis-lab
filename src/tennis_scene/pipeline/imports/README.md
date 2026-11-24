# Saved scene inputs

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`BallAnnotationImporter` reads saved task v3 annotations. `FixedGeometryImporter`
registers saved court detection, calibration, side and camera alignment as
external fixed inputs. Both publish new immutable scene artifacts through
`bind_import` / `publish_import`. Neither runs GPT, models, calibration, nor
training-data publication. Workflow/CLI composition belongs to the caller.

## Contracts

| Input | Scene artifact schema | Interpretation |
|---|---|---|
| Saved v3 per camera | `ball_annotations` v1 | Task-owned reader and usage masks |
| Annotation point pass-through | `ball_annotation_points` v1 | Unchanged coordinates; separate usable/observed/estimated masks |
| Annotation triangulation | `ball_annotation_trajectory` v1 | New 3D, contributing views, missing reasons and annotation provenance |
| Pinned fixed bundle | Existing geometry output schemas | External inputs; original dependencies retained as provenance |

The task [v3 contract, reader and usage API](../../../tasks/ball_detection/annotation/video/README.md)
remain authoritative; scene does not define another state table. Pass
`VideoAnnotationInput` values pinned by video registration, independently of the
annotation documents. The importer matches source byte identities, camera, image size,
FPS, complete clip frame axis and exact source PTS. Rows must cover
`[0, source.num_frames)`; the original video may be longer. Unprocessed rows are
rejected; completed `unknown` rows are accepted as missing.
Video bytes and decoded PTS are verified during registration; import does not decode video.

`BallAnnotationOutput` retains the task document, its original path and digest,
source PTS, native pixels, task point-kind strings (empty without a position),
and separate `position_usable`, `observed`, `estimated` masks. Confidence is
binary position acceptance, never a model probability. Supports, occlusion,
continuity breaks, notes and active-ball target semantics remain in the task
document. Stored derived fields are validated against the task API.

## Geometry and point policy

Observed and estimated positions both receive weight **1**. Annotation points
pass through unchanged. Triangulation uses all usable views with DLT and pixel
least-squares refinement. It applies **no smoothing, interpolation-duration
limit, reprojection-based view selection, speed/height bound, or minimum
trajectory-length gate**. The model/v2 path retains its existing observed-only
LOESS, consensus and reconstruction policy and schemas.

Fewer than two usable views produce a zero-filled missing point with
`missing_reasons="insufficient_views"` and rejection code 1. A failed geometric
solve (including degenerate rays or nonpositive depth) produces
`invalid_multiview_geometry` and the common solver-failure code 7. Missing
points contribute no views. `position_usable` records available inputs, including
single-view frames; `used_views` / `trajectory.inliers` record contributions to
valid 3D. All valid points use every available view. No input point is silently
removed. Reprojection errors are diagnostic only. The 3D artifact retains each
view's masks, point kinds, original annotation document/reference, and policy.

## Fixed bundle and usage

Use a pinned `tennis_lab_ball3d_fixed_inputs.v1` bundle with `bundle.json` and
`scene.json`; the format originates in the [evaluation bundle reader](../../evaluation/ball3d/bundle.py).
Supply the expected manifest digest explicitly. File/array checksums, source
identity, clip, camera order, calibration view indices, court frames, reference
camera and side/alignment consistency are checked before publication. Source
paths may relocate across machines; bytes, camera identity and timeline cannot.
The manifest's clip ID names the requested clip; the original scene's clip ID
and paths remain historical metadata.

`bind_nodes` requires load-only geometry nodes and makes them external leaves.
It does not attach old side/alignment artifacts to current ball points. New
descriptors retain the bundle reference, original artifact reference, original
identity/provenance/dependencies, and historical code provenance. The source
bundle is read-only and the destination store must be separate. No old ball
points or old 3D outputs are imported. Existing destination ball artifacts are
recomputed under the new annotation schemas.

The default `court_detection="stored_artifact"` requires original court arrays.
The supplied comparison bundle omits these arrays while retaining their
descriptors and the exact static court broadcast in calibration. Explicit
`court_detection="verified_calibration_frame0"` recovers frame 0 only when each
reconstructed NPY's shape, dtype and bytes match the original array checksum.
A mismatch fails; there is no automatic fallback or court inference.

```python
from src.tennis_scene.pipeline.imports import BallAnnotationImporter, FixedGeometryImporter
from src.tennis_scene.pipeline.runner import ComponentRunner

# source/store/nodes come from the caller's declared clip recipe.
# Geometry and ball_detection nodes must already be declared load-only.
fixed = FixedGeometryImporter(bundle_json, pinned_bundle_digest,
                             court_detection="verified_calibration_frame0")
balls = BallAnnotationImporter(annotation_paths, registered_inputs)
nodes = fixed.bind_nodes(nodes)
nodes = balls.bind_nodes(nodes)
fixed.publish(nodes, store, source)
balls.publish(nodes, store, source)
runner = ComponentRunner(nodes, store)
runner.run(targets=["ball_triangulation"])
result = runner.output("ball_triangulation")
```

Binding also pins expected import identities. Loading an older adopted artifact
after changing the requested bundle or annotation revision fails; import the
requested revision first. Final scene assembly accepts the shared ball-output
base contract; the binder declares its annotation trajectory port.

## Legacy v2 import

`ball_annotations.py::import_ball_annotations` remains the temporary v2 import
used by [component qualification](../../../../tests/benchmarks/component_pipeline.py).
It reads `<directory>/<camera>_annotations.json`, preserves numeric point kinds,
and only accepts observed coordinates as measurements. It does not acquire v3
usage semantics or convert old annotations. Imports record `origin=import`;
their declared upstream artifact bindings remain mandatory.

## Validation

```bash
.venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/pipeline/test_imports.py \
  tests/unit/tennis_scene/pipeline/test_video_ball_imports.py \
  tests/unit/tennis_scene/pipeline/components/test_ball_points.py \
  tests/integration/tennis_scene/test_ball_annotation_imports.py
```

Small asymmetric synthetic cameras and independently projected known 3D points
cover saved v3 reads, estimated contributions, missingness, persistence, fixed
geometry without execution, changed inputs/arrays and equal-weight noisy-view
fitting. They do not establish real-video annotation accuracy or acceptance.

## External clip ball positions

`src.tennis_scene.scripts.import_external_ball publish` adds ball 3D to a
structured player-only dataset. It binds each `<clip>_ball_positions.json` to
`clip.json`, matching camera names, dimensions, FPS and the complete zero-based
frame axis. The JSON provides frame ordinals and pixels. The saved dataset and
shared court pin the media SHA-256. All four point statuses are accepted inside
the image. The import records normalized `visible_in_frame` as a source
attribute and uses pixel bounds for the visibility mask.

The update reads the existing `scene_assembly`, calibration and camera
alignment as fixed artifacts by reference and checksum. It checks the ordered
cameras, half-turns and camera parameters against the pinned ball-based video
court side before reusing the camera alignment. A clip with a recorded failure
upstream of camera alignment uses its existing calibration with the pinned
half-turns. The scene records the historical failure reason, an empty player
axis with validity masks, and that player validation was skipped for those
cameras. The import runs on CPU and leaves the standard runner's cache
identity checks in effect.

The ball fit uses the annotation geometry policy above. In the exported
archive, `metadata.external_ball.status_codes` and `status_labels` describe the
input positions in camera/frame order. `ball_vis` identifies those accepted
in frame, and `ball_rejection_code` distinguishes insufficient views (1) from
a failed geometric solve (7). The immutable `external_ball_positions` component
also records original pixels and the normalized source visibility.

```bash
.venv/bin/python -m src.tennis_scene.scripts.import_external_ball publish \
  --dataset "$DATASET" --video video_000 --positions "$POSITIONS" \
  --side-store "$SIDE_STORE" --side-digest "$SIDE_DIGEST" \
  --court-store "$COURT_STORE" --drift-store "$DRIFT_STORE" \
  --report "$REPORT" --clip clip_000 --clip clip_020
```

Omit the repeated `--clip` flags to process the entire pinned video. Omit
`--drift-store` if the side decision pins no drift artifact. A rejected clip
keeps its previous pointers, and the report records the error and exits nonzero.

The command stages and verifies the new component and export before switching
`scene.json` and `annotation.json` by rename. A receipt at
`updates/external_ball/<scene_assembly ID>.json` holds both versions of the
pointers, the previous artifact/export references, and the pinned inputs.
Repeat the same command after an interruption to finish the pointer switch.

```bash
.venv/bin/python -m src.tennis_scene.scripts.import_external_ball restore \
  --dataset "$DATASET" --video video_000 --clip clip_000 --clip clip_020
```

Restore writes the exact previous pointer bytes and removes the new marker
where the clip had no previous marker. Both generations of immutable files
remain available. Omit `--clip` for the whole video. Restore checks that each
pointer still belongs to the recorded update. [Executed Meiji update and reproduction command](../../../../knowledge/runs/run-meiji-external-ball-20261124/README.md)
