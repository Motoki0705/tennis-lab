---
id: proposal-plcs-body-anchored-avatar-layers
type: proposal
title: PLCS avatarをbody-anchored semantic layerで制御する
curator: chatgpt-schedule
date: 2026-08-07
status: candidate
task: synthetic_data_generation
repo_paths:
  - src/synthetic_data_generation/composition/contracts.py
  - src/synthetic_data_generation/dataset/plcs/components/avatar_control.py
  - src/synthetic_data_generation/dataset/plcs/rendering/nht.py
hypothesis:
  statement: face-bound barycentric coordinate、正のnormal offset、semantic layer orderでGaussian geometryを更新すると、現行monolithic controlよりbody penetrationとlayer-order violationが減る
  expected_effect: penetration rate 2%以下、depth 1 mm以下、layer-order violation 1%以下を達成し、mask IoUとLPIPSの悪化を小さく保つ
  failure_condition: NHT appearance contract不一致、mask IoU 0.05以上低下、offsetの20%以上が上限へ張り付く、または16 GB GPUでOOM
evaluation:
  metrics: [penetration_rate, penetration_depth_mm, layer_order_violation_rate, temporal_mask_iou, lpips, fps, peak_vram_gb]
  baseline_nodes: []
  seeds: 3
  acceptance: 3 assets平均でpenetration rate 2%以下、depth 1 mm以下、layer-order violation 1%以下、mask IoU低下0.02以内、LPIPS悪化0.01以内、frame time 1.2倍以内にする
evidence_runs: []
parents: []
relations:
  - to: paper-arxiv-2605-21001
    rel: derived-from
tags: [literature, plcs, synthetic-data, avatar, garments]
---

## 背景

現行avatar controlはsurface embeddingを持つが、garment layerの正の厚みとlayer orderを明示しない。正式なrenderer baseline runを特定できないため`candidate`とする。

## 現行実装との差分

skin、upper、lowerの固定semantic splitを与え、Gaussian位置だけをface index、barycentric coordinate、正値normal offset、layer orderで更新する。appearance、camera、motion、Gaussian countは固定する。

## 最小検証

3 player assetsと100-frameのserve、forehand、lateral motionを固定し、現行monolithic controlとbody-anchored treatmentをheld-out 2 camerasで比較する。

## 比較対象

baselineは現行`avatar_control.py`とNHT rendererである。正式baseline runを登録後、`baseline_nodes`と`parents`を設定する。

## 合格条件と停止条件

frontmatterのacceptanceを満たした場合だけmulti-view reconstruction adapterへ進む。appearance contract不一致、mask IoU大幅低下、offset飽和、OOMで停止する。

## リスク

正のoffsetはpenetrationを減らす一方、外観を歪め得る。segmentationやSMPL fitの誤差がlayer assignmentへ伝播する。NHT appearance backendは初期実験で変更しない。
