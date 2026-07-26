---
id: run-i618-b00-gsplat-subprocess-smoke-v1
type: run
title: B00 gsplat file/subprocess captured-camera smoke
issue: 618
provider: codex
date: '2026-07-25'
status: done
config:
  model: b00-gsplat-file-subprocess-v1
  loss: none-static-rgb-ed-render
  data: accepted-scene-contract-frame-000242
metrics:
  focused_tests_passed: 20
  gaussian_count: 1286654
  render_elapsed_seconds: 1.609865801059641
  finite_depth_pixels: 516901
  scene_alpha_mean: 0.997829258441925
artifacts:
  report: .codex-loop/C08_B00_GSPLAT_SUBPROCESS_SMOKE.md
  output_dir: data/tennis/3dgs_renders/b00-default-v1/static-smoke-v1/frame_000242-41383e0dcae24605bd00aa9cbc00b130358b68405aeaf0b536dbf071bce5fae3
  scene_contract: data/tennis/3dgs_scenes/b00-default-v1/scene-contract-ground-line-user-override-v2.json
parents:
- run-i618-renderer-port-cpu-reference-v1
relations: []
tags:
- 3dgs
- gsplat
- subprocess
- rendering
- synthetic-data
---

## 考察 / Findings

### 要約

B00 final checkpointを外部application importなしのfile/subprocess境界で実renderできた。
accepted captured cameraの959×539 RGB、OpenCV camera-Z depth、scene alphaを厳格検証し、
content-addressed artifactとして原子的に公開した。

### アーキテクチャ詳細

tennis-lab側adapterはSceneCameraとplain JSONだけをworkerへ渡し、wrapper/CUDA module/
worker/checkpointのSHAとgsplat commitを事前検証する。standalone workerはprovider application
moduleをimportせず、pinned trainerと同じparameter activationおよびSH degree 3で`RGB+ED`
をrenderする。応答NPZはsize/SHA/shape/dtype/depth conventionをmain processで再検証する。

### メトリクスの解釈

1,286,654 Gaussianを1.61秒でrenderし、全516,901画素で有限positive depthを得た。
scene alpha mean 0.99783で、depth中央値は1.376 scene unitsだった。focused rendering testsは
20件passし、captured pose previewも対象テニスコートsceneを再構成した。

### アーキテクチャ⇄メトリクスの因果考察

prebuilt extensionをSHA検証後にalias loadすることで、直接gsplat import時に観測した不要な
JIT buildを回避できた。仮説として、この同一camera/depth bufferをC07 sphere supersampling
へ供給すれば、ball RGBとocclusion labelを同じscene geometryに結びつけられる。

### 既存実験との比較

親runはrenderer-independent sphere/visibility semanticsをCPU referenceで固定した。本runは
その前段に実B00 static scene RGB/depth portを加えたが、BLCS numeric writerやball detectorへ
gsplat依存を漏らしていない。まだtrajectoryまたはball合成の精度結果ではない。

### 次に有効な実験

accepted Sim(3)でmetric BLCS ball sampleを一度だけsceneへ変換し、本runのRGB/depthとC07
rendererを合成する。0.0335 m green sphereのRGB、coverage、visibilityと空negative frameを
single-frame pilotで検証してからclip生成へ進む。
