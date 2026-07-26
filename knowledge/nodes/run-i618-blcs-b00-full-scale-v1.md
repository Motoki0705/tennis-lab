---
id: run-i618-blcs-b00-full-scale-v1
type: run
title: B00 3DGS × BLCS full-scale TrackNet学習データ
issue: 618
provider: codex
date: '2026-07-26'
status: done
config:
  model: deterministic-cpu-sphere-reference-v1-over-four-b00-rgb-depth-views
  loss: none-dataset-generation-and-label-contract
  data: 16-blcs-trajectories-x-4-camera-groups-x-64-contiguous-frames
metrics:
  frame_count: 4096
  clip_count: 64
  positive_frames: 2913
  negative_frames: 1183
  fully_visible_frames: 2830
  partially_occluded_frames: 89
  fully_occluded_frames: 623
  out_of_frame_frames: 554
  median_ball_diameter_px: 1.8381074380904354
  q90_ball_diameter_px: 3.806149277612179
  median_visible_displacement_px: 7.987878581954192
  q90_visible_displacement_px: 19.905615169169295
  dataloader_windows: 64
  dataloader_heatmap_max: 0.9911755919456482
  focused_tests_passed: 28
  tennis_scene_tests_passed: 208
artifacts:
  report: .codex-loop/C10_B00_FULL_SCALE_SYNTHETIC_DATASET.md
  output_dir: data/tennis/3dgs_synthetic/b00-default-v1/full-scale-v1/5d108d8c9cc2e27224a4d294095b01c6f1de702e634367eafe0aef2771412c9f
  manifest: data/tennis/3dgs_synthetic/b00-default-v1/full-scale-v1/5d108d8c9cc2e27224a4d294095b01c6f1de702e634367eafe0aef2771412c9f/manifest.json
  scene_contract: data/tennis/3dgs_scenes/b00-default-v1/scene-contract-ground-line-user-override-v2.json
parents:
- run-i618-blcs-b00-single-frame-pilot-v1
relations: []
tags:
- 3dgs
- blcs
- rendering
- tracknet
- synthetic-data
- full-scale
---

## 考察 / Findings

### 要約

16独立BLCS trajectory、4つのB00 captured camera group、各64連続frameから、
4,096 frame / 64 clipのtraining-only TrackNet datasetをatomicに公開した。
scene-depth occlusion、negative保持、全payload hash、既存DataLoader gateがpassした。

### アーキテクチャ詳細

synthetic pipelineは`BLCSSceneData`、accepted `SceneContract`、`RendererPort`だけを受け、
gsplat application module、ball-detection、numeric `BLCSDatasetWriter`をimportしない。
court-metre trajectory全体をaccepted Sim(3)で一度だけsceneへ変換し、半径0.0335 mの
green sphereを4x supersamplingする。公開後だけ既存`TrackNetDataModule`が
`Label.csv`とtraining splitを読む。

### メトリクスの解釈

2,913 positiveに対して、fully-occluded 623とout-of-frame 554を含む1,183 negativeを
連続clip内に保持した。見かけ径median 1.838 px / q90 3.806 pxでsmall-object領域が中心、
可視時変位median 7.988 px / q90 19.906 pxでtemporal motionにも幅がある。
64個の8-frame DataLoader window smokeはheatmap max 0.991を得た。

### アーキテクチャ⇄メトリクスの因果考察

仮説として、四方向の固定captured poseとBLCS rallyの連続性により、single-frame pilotでは
無かった背景・速度・遮蔽分布を学習に渡せる。ROI rendererはfull-frame referenceと
pixel-exactであり、小球でもRGBとlabel evidenceの幾何を分離して調整していない。
ただしflat green sphereのdomain gapがreal F1へ正に効くかはまだ未検証である。

### 既存実験との比較

親runは1 positive + 1 negativeだけで、trajectory continuityやcamera diversityを測れなかった。
本runは16 trajectory × 4 cameraへ拡張し、court occupancy、diameter、speed、visible
displacement、visibility、background contrast、clip durationをmanifestへ固定した。
初回full publicationの画面外変位集計と背景混入を観測後、一つの集計修正だけを適用した。
さらにprovenance監査で無関係なworktree差分をcode identityから除外し、関連source hashと
base revisionだけで安定するcanonical fingerprintへ再公開した。

### 次に有効な実験

固定real evaluation manifestと同一initialization/optimizer/budget/augmentation/seedを使い、
controlはreal trainingのみ、treatmentは同じreal trainingに本datasetを宣言済み比率で混ぜる。
mixing選択はreal validationだけで行い、paired seedの最終test差とbootstrap CIを報告する。
