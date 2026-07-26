---
id: run-i618-blcs-b00-single-frame-pilot-v1
type: run
title: BLCS metric ball × B00 single-frame supervised pilot
issue: 618
provider: codex
date: '2026-07-26'
status: done
config:
  model: deterministic-cpu-sphere-reference-v1-over-b00-rgb-depth
  loss: none-single-frame-geometry-label-gate
  data: blcs-seed-20260726-frame-502-camera-frame-000242
metrics:
  focused_tests_passed: 23
  tennis_scene_tests_passed: 203
  apparent_ball_diameter_px: 2.899375186065505
  visible_pixel_fraction: 1.0
  dataloader_windows: 2
  positive_heatmap_max: 0.9809619188308716
artifacts:
  report: .codex-loop/C09_BLCS_B00_SINGLE_FRAME_PILOT.md
  output_dir: data/tennis/3dgs_synthetic/b00-default-v1/single-frame-pilot-v1/c7019079c3228dffd50aeef5996fa0f69a69c0c47a71206f8342efd3266c7735
  scene_contract: data/tennis/3dgs_scenes/b00-default-v1/scene-contract-ground-line-user-override-v2.json
parents:
- run-i618-b00-gsplat-subprocess-smoke-v1
relations: []
tags:
- 3dgs
- blcs
- rendering
- tracknet
- synthetic-data
---

## 考察 / Findings

### 要約

既存BLCS physics trajectoryのmetric pointをaccepted Sim(3)で一度だけB00 sceneへ変換し、
0.0335 m green sphereのpositiveと同一背景empty negativeをTrackNet互換pilotとして公開した。
geometry/label/decode/DataLoader gateがすべてpassした。

### アーキテクチャ詳細

新しいsynthetic dataset pipelineは`BLCSSceneData`、`SceneContract`、`RendererPort`だけを受け、
numeric `BLCSDatasetWriter`、gsplat、ball-detectionをimportしない。positive/negative RGBと
coverage/alpha/depth/evidenceをatomic fingerprinted directoryへ保存する。ball detector側は
公開済み`Label.csv`とsplitだけを既存`TrackNetDataModule`で読む。

### メトリクスの解釈

BLCS frame 502のcourt位置は`(1.009,3.178,3.107) m`で、B00 camera上の見かけ径は2.899 px。
coverage 7.3125 pixel-equivalentが100% visibleで、JPEG後も65 pixelsにpositive/negative差が
残った。DataLoaderは2 windowsを復号し、positive heatmap max 0.981、negative 0.0を得た。

### アーキテクチャ⇄メトリクスの因果考察

scene depthとsphere rayを同じOpenCV camera-Z単位で比較したため、visibilityとRGBを別々に
調整せずfully-visible判定を得られた。仮説として、約3 pxの球でも4x supersamplingと高品質
JPEGなら学習targetを保持できるが、temporal contextでの挙動はまだ未検証である。

### 既存実験との比較

親runはstatic B00 RGB/depthまででballを含まなかった。本runはreal BLCS physics sample、
metric radius、TrackNet label、negative、既存DataLoaderまで接続した。一方、2 framesのみで
trajectory continuity、diversity、real-data accuracy改善は評価できない。

### 次に有効な実験

同じfrozen BLCS trajectoryとfixed cameraから短い連続clipを作り、全frameのvisibility state、
pixel displacement、diameter、背景contrastを集計する。8-frame TrackNet windowのdecode/
DataLoader gateをpassしてからtrajectory/camera数を増やす。
