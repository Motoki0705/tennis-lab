---
id: run-i618-b00-ground-line-court-fit-v1
type: run
title: B00 ground-line aggregation と metric multi-court fit
issue: 618
provider: codex
date: '2026-07-25'
status: done
config:
  model: ckpt/court_detection/line/court-detection-epoch19.ckpt
  backbone: third_party/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
  data: B00 provider fit groups 0,1,3,4,5,7,8,9,11,12,13,15
  ground_plane: camera-up constrained deterministic RANSAC
  projection: K^-1 ray-plane + camera-range proximity weighting
  geometry: two-instance ITF metric template fit
metrics:
  fit_views: 363
  accepted_projected_views: 361
  projected_line_pixels: 719526
  ground_support_points: 65558
  ground_residual_rms_scene: 0.003102641529236107
  selected_template_score: 1.0759825706481934
  selected_scale_scene_per_m: 0.07002697494876446
  adjacent_court_orientation_difference_deg: 0.2650263723899497
  adjacent_court_relative_scale_difference: 0.005647665758975289
artifacts:
  ground_line_map: data/tennis/3dgs_alignment/b00-default-v1/ground_line_maps/b00-ground-line-map-v1-43a11bfbcddf6a2c
  court_geometry: data/tennis/3dgs_alignment/b00-default-v1/court_geometry/b00-ground-court-geometry-v1-dabea42f4e223951.json
  report: .codex-loop/C04_GROUND_LINE_ALIGNMENT.md
parents:
- run-i618-b00-fit-court-detections-v1
relations: []
tags:
- 3dgs
- blcs
- alignment
- court-line
- fit-only
---

## 考察 / Findings

### 要約

C03 の global keypoint argmax 失敗に対し、fit camera だけから地面を推定し、line
segmentation を ray-plane 投影して近距離 camera を優先すると、隣接する 2 面を明瞭に
分離できた。metric template fit は 2 面を平均せず、最高 score の `court-0` を
physical-court candidate として選択した。holdout は未推論なので alignment acceptance
はまだ宣言しない。

### アーキテクチャ詳細

line checkpoint と同名・固定 SHA-256 の local DINOv3 backbone を明示的に結合した。
provider point cloud と fit-camera up/height から ground plane を deterministic RANSAC
で求め、original 959x539 pixel centre を共有 K で back-project した。各 raster cell
では view 内 max を取ってから、`1/(1+(range/0.35)^2)` を掛けて view 間加算した。
最後に ITF doubles/singles/service/center-service line template を Sim(2) で探索し、
ground basis と normal を用いて proper-handed Sim(3) へ持ち上げた。

### メトリクスの解釈

363 fit views のうち 361 views、719,526 line pixels が ground map に寄与した。ground
support は 65,558 points、plane RMS は 0.003103 scene units、全 fit camera height は
正だった。2 candidate の orientation 差 0.265°、scale 差 0.565% は、同一 scene 内の
隣接標準 court として整合する。選択 court の scale は 0.070027 scene units/m。
訓練 run ではないため convergence curve は存在しない。

### アーキテクチャ⇄メトリクスの因果考察

観測として、集約 map には 2 面の baseline、sideline、service line が別々に現れた。
近距離 weight と per-view max が、遠方 court の誤検出や同一 view 内の重複 pixel の
影響を抑えたと考えられる。これは因果仮説であり、最終的な正しさは未使用 holdout の
reprojection と independent point/depth support で検証する必要がある。

### 既存実験との比較

親 `run-i618-b00-fit-court-detections-v1` は 363/363 views が 12-inlier gate を満たさず、
full-court keypoint homography を作れなかった。本 run は partial line evidence を
ground coordinate で統合し、361/363 views を利用して 2 physical instances を分離した。
ただし、metric template score は C03 の reprojection acceptance metric の代替ではない。

### 次に有効な実験

fit で固定した `court-0` transform を一切再最適化せず、隔離済み groups
`{2,6,10,14}` の line/keypoint evidence へ投影して held-out reprojection/line distance、
camera height、court-plane point/depth support、fit-group subset stability を評価する。
