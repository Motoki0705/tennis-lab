---
id: run-i618-renderer-port-cpu-reference-v1
type: run
title: 3DGS非依存renderer portとCPU reference
issue: 618
provider: codex
date: '2026-07-25'
status: done
config:
  model: deterministic-cpu-sphere-reference-v1
  loss: OpenCV camera-z ray/sphere depth and shared supersample evidence
  data: SceneCamera plus scene-space sphere primitives
metrics:
  focused_tests_passed: 10
  tennis_scene_tests_passed: 195
  b00_court_center_visible_cameras: 323
  b00_tennis_ball_diameter_median_px: 2.3745279221318105
artifacts:
  report: .codex-loop/C07_RENDERER_PORT.md
  scene_contract: data/tennis/3dgs_scenes/b00-default-v1/scene-contract-ground-line-user-override-v2.json
parents:
- run-i618-b00-alignment-user-override-v1
relations: []
tags:
- 3dgs
- rendering
- contract
- synthetic-data
---

## 考察 / Findings

### 要約

gsplat/BLCS非依存のtyped renderer portとdeterministic CPU referenceを実装した。
RGB、scene/sphere depth、alpha/coverage、投影中心・径、visible fraction、4状態visibilityを
同じsupersample ray集合から一貫して生成できる。

### アーキテクチャ詳細

入力はaccepted `SceneCamera`、scene fingerprint、frame index、scene-space sphereである。
portはcourt metresを知らず、trajectory変換を下流で重複させない。CPU adapterはOpenCV
pixel-centre rayとexact ray/sphere intersectionを使い、static scene camera-Z depthおよび
複数sphere間の最前面を比較する。空sphere requestもnegative frameとして保持する。

### メトリクスの解釈

focused 10 testsとtennis-scene全195 testsがpassした。B00 contractでcourt centreの高さ
1 mに半径0.0335 mのballを置くと323/491 camerasでcentreがin-frameとなり、見かけ径は
min/median/max 1.365/2.375/11.745 pxだった。小球には4x supersamplingが必要な範囲である。

### アーキテクチャ⇄メトリクスの因果考察

coverageをocclusion前、alphaをscene/sphere occlusion後のsample fractionとして分けたため、
fully occludedでも幾何学的ball footprintを失わずlabel stateを決められる。仮説として、
median約2.4 pxの対象ではbackend側のantialiasing一致がsynthetic transferに強く影響する。

### 既存実験との比較

親runはscene/camera/Sim(3)境界までを固定した。本runはそのcontract型だけへ依存し、
BLCS numeric writer、ball detector、外部gsplat moduleをimportせずrender/label境界を追加した。

### 次に有効な実験

B00 providerを読むsubprocess/file adapterを実装し、同一cameraでscene RGBとcamera-Z depthを
返すsingle-frame smokeを行う。外部repository moduleはtennis-lab processへimportせず、
request/response schema・backend commit・checkpoint hashを固定する。
