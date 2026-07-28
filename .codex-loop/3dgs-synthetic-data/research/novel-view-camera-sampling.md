# P6 novel-view camera sampling decision

Decision date: 2026-07-28 JST. Scope: generate court-detection cameras near the
accepted B00 SfM trajectory without treating an unconstrained renderer path as
observed scene support.

## Primary methods and official implementations

### Mip-NeRF 360

- Paper: **Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields**
  ([paper](https://arxiv.org/abs/2111.12077)).
- Official code:
  [google-research/multinerf](https://github.com/google-research/multinerf),
  pinned at `5b4d4f64608ec8077222c52fdf814d40acc10bc1`.
- Relevant mechanism: the official `internal/camera_utils.py` first aligns and
  scales captured poses with PCA, estimates a focus point, and can generate an
  elliptical render path.
- Applicability: captured-pose normalization and focus statistics are useful
  ways to describe the support of an SfM trajectory.
- Limitation and decision: a global ellipse is a presentation trajectory, not
  a proof of radiance-field support. B00 is a forward-facing walk around a
  court, and only 42/491 captured poses fully contain the fourteen line
  keypoints. An unclipped ellipse can leave both the observed camera
  neighborhood and the court-facing orientations. We therefore use the
  captured poses as local support centres and do **not** import the ellipse
  generator.

### NeRF Director

- Paper: **NeRF Director: Revisiting View Selection in Neural Volume
  Rendering** ([paper](https://arxiv.org/abs/2406.08839),
  [CVPR version](https://openaccess.thecvf.com/content/CVPR2024/html/Wei_NeRF_Director_Revisiting_View_Selection_in_Neural_Volume_Rendering_CVPR_2024_paper.html)).
- Project:
  [wenwhx.github.io/nerfdirector](https://wenwhx.github.io/nerfdirector/).
- Official code:
  [wenwhx/nerfdirector](https://github.com/wenwhx/nerfdirector), pinned at
  `9471c8698077f0edac9e749208db9ef987cb5ca8`.
- Relevant result: view diversity is a first-order factor, and farthest-view
  sampling (FVS) produces a more uniform selected camera set than unstructured
  random selection.
- Applicability: after hard safety rejection, FVS over a normalized SE(3)
  distance is a good deterministic selector for distributing accepted novel
  views over all safe anchors.
- Limitation and decision: the repository's licence is non-commercial and its
  implementation is coupled to its own COLMAP/photogrammetric distance setup.
  No source is copied. We independently implement the algorithmic selection
  idea over our explicit metric translation and geodesic-rotation score.

### FisherRF

- Paper: **FisherRF: Active View Selection and Uncertainty Quantification for
  Radiance Fields using Fisher Information**
  ([paper](https://arxiv.org/abs/2311.17874)).
- Project: [jiangwenpl.github.io/FisherRF](https://jiangwenpl.github.io/FisherRF/).
- Official code:
  [JiangWenPL/FisherRF](https://github.com/JiangWenPL/FisherRF), pinned at
  `b74732812b295189f230a192418375f56cec3bd6`.
- Relevant mechanism: Fisher information estimates expected information gain
  for active acquisition and uncertainty.
- Limitation and decision: the official implementation depends on its own 3DGS
  and differentiable-Hessian boundary. The frozen NHT renderer used here does
  not expose an equivalent validated Fisher objective, and active camera
  acquisition is not the same objective as a safe synthetic court trajectory.
  It is rejected as the P6 proposal generator. It may be evaluated later only
  as an uncertainty audit after an explicit NHT adaptation.

### NeRF++

- Paper: **NeRF++: Analyzing and Improving Neural Radiance Fields**
  ([paper](https://arxiv.org/abs/2010.07492)).
- Official code:
  [Kai-46/nerfplusplus](https://github.com/Kai-46/nerfplusplus), pinned at
  `ebf2f3e75fd6c5dfc8c9d0b533800daaf17bd95f`.
- Applicability: the paper explicitly separates interpolation from
  extrapolation and shows that novel views outside the training-camera support
  are substantially less reliable.
- Limitation and decision: it does not provide a court-aware sampling
  algorithm. It supplies the reason for a strict nearest-captured-pose
  extrapolation bound rather than a generator.

### Nerfstudio

- Paper: **Nerfstudio: A Modular Framework for Neural Radiance Field
  Development** ([paper](https://arxiv.org/abs/2302.04264)).
- Official code:
  [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio),
  pinned at `50e0e3c70c775e89333256213363badbf074f29d`.
- Applicability: its camera-path UI and keyframe interpolation demonstrate
  useful trajectory authoring and rotation interpolation.
- Limitation and decision: interpolation alone has no scene-support,
  collision, near-plane, or court-framing guarantee. It is a supporting
  implementation reference, not the selected sampler.

## Selected hybrid and pre-registered thresholds

The implementation is independent and has two strictly ordered stages:

1. Generate a six-dimensional uniform-ball perturbation around captured poses
   that already contain all 14 line keypoints.
2. Reject unsafe candidates, then apply NeRF Director-style farthest-view
   selection. No gate is relaxed when the requested output count is unavailable.

All quantities are evaluated in the accepted metric court frame:

| Gate | Threshold | Pre-registration basis |
|---|---:|---|
| translation from nearest captured pose | coupled limit 0.25 m | 25.3% of the measured 0.9895 m median adjacent SfM step |
| geodesic rotation | coupled limit 1.5 degrees | below the measured 3.0889 degree median adjacent step |
| extrapolation score | `sqrt((d/0.25)^2 + (theta/1.5)^2) <= 1` | explicit local SE(3) support ball |
| camera height | at least 1.20 m | accepted captured range is 1.4339--2.8224 m; a full downward perturbation remains bounded |
| sparse-scene collision | 8th-nearest point at least 0.25 m away | minimum over 491 captured cameras is 0.2626 m |
| near plane | all CourtKP20 depths greater than 0.10 m | explicit positive-depth/near-plane guard |
| framing | all first 14 CourtKP20 line points visible, margin at least 0 px | 42 captured poses meet this stronger full-court gate |

The six-dimensional support ball couples translation and rotation; separately
maximal translation and rotation cannot combine into an out-of-bound corner.
FVS initializes each candidate's distance with its nearest captured-pose
distance and repeatedly maximizes the minimum distance to captured and already
selected poses.

## Failed or rejected sampling hypotheses

- **Unclipped global MultiNeRF ellipse:** rejected because it is not local
  support and can face away from the court.
- **FisherRF expected-information proposal generation:** rejected because the
  validated NHT Fisher/Hessian boundary does not exist and the objective differs.
- **Independent random jitter as the final dataset:** rejected because it can
  cluster around a few anchors; randomness is only a proposal mechanism before
  FVS.
- **Keyframe interpolation without court projection gates:** rejected because
  valid camera interpolation can still crop or place line keypoints behind the
  camera.
- **Camera-position convex hull alone:** rejected because a position inside the
  hull can have an unsupported orientation. The selected score couples position
  and rotation and is followed by exact projection gates.
