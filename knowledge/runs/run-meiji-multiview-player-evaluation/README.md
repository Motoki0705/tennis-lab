# Meiji 4-camera player evaluation

This run builds `meiji_4cam/multiview-player-evaluation-v1` for the six clips agreed in [#2492](https://github.com/Motoki0705/tennis-lab/issues/2492). It uses the adopted scene exports and component arrays in `meiji_4cam/multiview-v1`.

## Reproduction

From the repository root, provide the pinned input benchmark, the installed GVHMR auxiliary assets and a new destination on the same data volume. For an official run, use a clean checkout of the generator revision recorded in `manifest.json`.

```bash
benchmark=data/tennis_multiview/processed/meiji_4cam/benchmarks
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -B knowledge/runs/run-meiji-multiview-player-evaluation/build.py \
  --source "$benchmark/multiview-v1" --assets ckpt/body_models/gvhmr \
  --output "$benchmark/multiview-player-evaluation-v1"
(cd "$benchmark/multiview-player-evaluation-v1" && sha256sum -c SHA256SUMS)
```

## Reference policy fixed before scoring

- Use the union of the saved GVHMR segment frame indices for the image cohort. An individual player needs a saved GVHMR prediction on that frame and valid placed SMPL. Keep each player's missing frames explicit. The original `clip_028` has 1,281/1,288 frames with four valid placed bodies in the source archive.
- Use the scene's Y-up root-relative mesh, with the saved placement scale already applied. Reuse the renderer's transform and the pinned joint-0 regressor. Regress COCO17 on root-relative geometry and add court translation once. Use the saved pinhole calibration at native pixel resolution. Diagnostic residuals never replace a joint or enter the quality thresholds.
- Require at least 12 in-image, positive-depth COCO17 joints to construct a box. Expand the joint envelope by 10% of its width and height on each respective side and clip to `[0,width] × [0,height]`. The preflight sampled placed meshes in all six clips. Per-clip median IoU with the projected mesh box was roughly 0.87–0.89 at 10%, 0.67–0.68 at 0%, and 0.80–0.83 at 15%. The full audit recomputes this comparison, including sensitivity at the margins listed in the output manifest.
- For detection, require observations for the person in at least two cameras and at least 12 in-image joints after excluding likely inter-player occlusions. The latter flag requires a missing same-view joint observation and projection inside the triangle of a nearer player's torso, with a depth gap of at least 0.25 m. A missing observation without that evidence leaves visibility undetermined. For pose, require the eligible box, same-view observation support for the joint and joint support from at least two views. A camera without same-view observations can still contribute detection references. The masks use evidence of an observation, independent of its pixel error.
- ViTPose and triangulated joints are diagnostic and visibility evidence. These statistics are coupled to the models used to create the underlying scene and cannot establish independent accuracy, resolve self-occlusion, or measure every non-player in the image. Precision and AP are unavailable. Overlap for clip selection is the fraction of original frames with IoU greater than 0.1 between two tight observed keypoint boxes in any camera.

## Test design

Use small CPU unit fixtures. An asymmetric body with a nonzero translation and a deliberately non-unit regressor row checks the metric-space transform and catches a second translation, inverse rotation or use of triangulation as the reference. A pinhole fixture and overlapping players check behind-camera, out-of-image, likely occluded and undetermined states. Hand-computed two-person matches and a missing-prediction frame check matching cardinality, partial labels, threshold-specific recall, OKS, PCK and joint coverage. A minimal bundle fixture checks the manifest checksum and frame axes.

Reuse `test_hip_anchor_is_not_assumed_to_be_smpl_root`, `test_root_centering_axis_swap_yaw_and_translation`, the scene v2 archive round-trip, and the existing partial-label assignment tests. New tests exercise the evaluation API and use no external assets. Measure execution with `pytest --durations=0 --durations-min=0`. Real-data generation reports the independent OpenCV projection check, left/right swap check and smoke scoring with the stored ViTPose observations.
