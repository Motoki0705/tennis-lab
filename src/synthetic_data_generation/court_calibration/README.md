# Synthetic court-line calibration database (CPU baseline)

Independent research baseline inspired by Chen & Little,
[Sports Camera Calibration via Synthetic Data](https://arxiv.org/html/1810.10658)
(CVPR Workshops 2019; [author implementation](https://github.com/lood339/SCCvSD)).
The paper uses a camera prior, learned Siamese edge descriptors, nearest-neighbor
retrieval and truncated-distance pose refinement. This implementation deliberately
substitutes a NumPy HOG descriptor, exact L2 top-k retrieval and OpenCV homography
ECC. It is not a reproduction of the learned method or its reported accuracy.

Camera center, target and horizontal FOV are uniformly sampled with a local
NumPy seed, then passed to the existing `make_look_at_camera`. Geometry and line
connectivity come only from `src.utils.schema.court` (ground KP0–13; no net).
No NHT renderer, GPU, network download, RGB detector or trained model is needed.
The provided camera envelope is an illustrative near-baseline prior, not a
measured Meiji camera distribution. Invalid samples abort generation; there is
no hidden resampling or fallback camera. Centered principal point, square pixels,
zero roll and no lens distortion are assumptions of this baseline.

## Reproduce

From the repository root, using the existing project environment:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.court_line_database
.venv/bin/python -m src.synthetic_data_generation.scripts.court_line_database \
  mode=query output_dir='${tennis_output:court_detection,analyze,court_line_database,${tennis_run_id:}}'
```

The query command requires a binary uint8 PNG at the configured `query_path`:
0 background, 255 observed lines. Threshold detector probabilities explicitly
upstream; RGB images, soft masks, blank and overly dense masks are rejected.
Use `--config /absolute/config.yaml` or strict `key=value` overrides, for example
`database.count=100 database.camera.height_m='[3,8]'`. Camera-prior changes require
a new `database_path` and identical settings at query time. Existing archives
are never overwritten. Config/source/OpenCV mismatches fail before retrieval.

Paths use existing `RuntimePathRoots`/`PathResolver`; output names use the
registered OmegaConf `tennis_output` resolver. This minimal argparse entrypoint
loads OmegaConf directly (no Hydra working-directory changes). Generated DBs
are DATA-root-relative and versioned; resolved config and result JSON belong
under OUTPUT `court_detection/{generate,analyze}/<experiment>/<run-id>/`, following
[output conventions](../../tasks/OUTPUTS.md). The YAML's default output is for
generation; the query command above changes its purpose to analyze.

## API and artifact contract

- `DatabaseConfig.from_mapping`, `generate`, `LineDatabase.save/load` in
  `database.py` create and reopen a DB. Loading requires `expected_config`.
- The NPZ contains K, R, t, H, binary line masks, HOG descriptors and metadata:
  schema, complete config/seed, axes, geometry/source implementation SHA-256,
  descriptor identity and OpenCV version. Shape/dtype/finite checks, proper
  rotations, K/R/t/H consistency and regenerated masks/descriptors are verified.
  The baseline bounds generation to 20 million pixels and archive loading to
  1 GB uncompressed; it loads the complete DB in memory.
- `query_database(db, binary_mask, top_k=...)` returns HOG-ranked `Match` objects.
  Each contains descriptor distance, initial world-to-original-query H, explicit
  success/reason, ECC score and refined H (or `None` on failure). Candidates are
  not silently reranked or substituted; the caller owns selection and evidence
  gating. Retained K/R/t describe the *retrieved initial* camera only.
- H maps canonical court XY metres to image pixel coordinates, with Z=0 and
  world Z up. R is world-to-camera (right/down/forward), t=-R*C. No semantic
  half-turn or near/far keypoint relabeling is performed.
- `findTransformECC(templateImage=query, inputImage=synthetic)` returns W mapping
  query pixels to template pixels. Refined H is `inverse(W) @ H_template`.
  Nonmatching image sizes are explicitly resized anisotropically using
  nearest-exact interpolation; S includes OpenCV's half-pixel center offset.
  Returned original-image H is `inverse(S) @ inverse(W) @ H_template`, and the
  reported query-to-template map is `W @ S`. There is no guessed crop/padding.
- HOG uses unsigned 9-bin orientation interpolation, 8-pixel cells, 2x2-cell
  overlapping blocks and L2-Hys normalization; it omits spatial interpolation.
  This fixed NumPy implementation also works with the repository's OpenCV
  build, which has no `HOGDescriptor` module.

## Limits and validation

ECC success means the optimizer returned a finite orientation-preserving map,
not that camera geometry is correct. A large mismatch can converge to a local
optimum, and the iteration cap is not a proof of convergence. Low viewpoints,
partial/occluded courts, ambiguous parallel lines and the court's bilateral and
half-turn symmetries remain unresolved. An arbitrary refined planar H does not
uniquely determine K/R/t; no refined physical camera is fabricated.

The existing RGB court checkpoint remains an independent source of evidence.
Real-scene assessment must use independently observed RGB lines and held-out
line evidence. Connecting predicted keypoints is not independent evidence of
improvement. This module has no production-pipeline integration and no Meiji
accuracy claim. Siamese training, RGB extraction, physically constrained
refinement, prior coverage studies and real-data comparisons remain future work.

```bash
.venv/bin/python -m pytest tests/unit/synthetic_data_generation/court_calibration -n 0
```

CPU tests cover deterministic generation, archive roundtrip/config rejection,
K/R/t/H and canonical axes, retrieval, known perspective recovery (<1 pixel at
all 14 ground points), nonmatching-aspect resize direction, malformed/blank masks,
explicit ECC failure and full CLI generation/query with root-escape rejection.
