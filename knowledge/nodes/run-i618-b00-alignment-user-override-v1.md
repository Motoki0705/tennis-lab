---
id: run-i618-b00-alignment-user-override-v1
type: run
title: B00 alignment user override SceneContract公開
issue: 618
provider: codex
date: '2026-07-25'
status: done
config:
  model: b00-ground-line-alignment-user-override-v2
  loss: explicit user acceptance over immutable rejected holdout
  data: B00 provider 491 cameras / court-0 Sim(3)
metrics:
  accepted_by_user_override: 1
  machine_validation_accepted: 0
  failed_gate_count: 2
  scene_camera_count: 491
artifacts:
  decision: data/tennis/3dgs_alignment/b00-default-v1/acceptance_decisions/b00-ground-line-user-override-v2-8a15434eabea83b6.json
  scene_contract: data/tennis/3dgs_scenes/b00-default-v1/scene-contract-ground-line-user-override-v2.json
  report: .codex-loop/C06_SCENE_CONTRACT_OVERRIDE.md
parents:
- run-i618-b00-alignment-holdout-v1
relations: []
tags:
- 3dgs
- alignment
- scene-contract
- user-override
---

## 考察 / Findings

### 要約

ユーザーの定性的確認による明示的overrideを、machine holdoutの棄却を改変せず
`accepted_by_user_override` decisionとして固定した。491 camerasと`court-0` Sim(3)を持つ
SceneContractを公開し、renderer port実装を開始できる状態にした。
初版のmutable STATE参照は検出後にimmutable source record参照のv2で置き換え、v1 artifactも
negative provenanceとして保持した。

### アーキテクチャ詳細

decision artifactはuser authority/source hash、fit calibration、rejected holdout、実際に
falseだった2 gate、provider fingerprint、court cluster/symmetryをstrictに照合する。
SceneContractのalignment manifestはholdoutを直接pass扱いせず、このdecision artifactを
参照する。外部3DGS application moduleへのimportやpath injectionはない。

### メトリクスの解釈

これは学習runではない。`accepted_by_user_override=1`と同時に
`machine_validation_accepted=0`を保持することが主要結果である。自動判定のq95
1.23986 mとgroup coverage 0.22348の失敗は親runから変えていない。

### アーキテクチャ⇄メトリクスの因果考察

overrideを別artifactにしたため、下流は採用済みcontractを狭いinterfaceで利用できる一方、
自動gate passと誤認できない。仮説として、対象courtを向くviewが少ないholdout groupの
coverage不足はpilot renderingの実用幾何を必ずしも否定しないが、これはmachine acceptance
の改善を意味しない。

### 既存実験との比較

親 `run-i618-b00-alignment-holdout-v1` のstatusとartifactはrejectedのままである。本runは
その結果をsupersedeせず、ユーザー権限による別decision layerを追加した。

### 次に有効な実験

SceneContractだけに依存するrenderer portを実装し、cameraとscene-space sphereからRGB、
scene/sphere depth、alpha/coverage、occlusion evidenceを返すCPU fake adapterのcontract
testsを先に通す。その後3DGS subprocess adapterへ接続する。
