---
id: group-i1899-controlled-query-ablation-s42
type: group
task: ball_detection
sequence: 94
recorded_at: '2026-11-13'
title: 旧queryを基準にしたquery構成・CNN・前半系列長の4条件比較
issue: 1899
members:
- run-ball-sequential-length-20261025
- run-i1899-convnext-4query-mhc-t64-s42-12k
- run-i1899-x3d-1query-t64-s42-12k
- run-i1899-convnext-1query-t64-t64-s42-12k
parents: []
papers: []
tags:
- ablation
- validation
- single-seed
status: done
summary_impact: updated
provider: codex
date: '2026-11-13'
config:
  seed: 42
  query_updates_per_condition: 12000
  final_T: 64
  observed_validation_frames: 85473
metrics:
  baseline_mean_error_px: 28.200801773291
  query_change_mean_error_px: 36.42065879366264
  cnn_change_mean_error_px: 33.980226994374796
  length_change_mean_error_px: 28.526567810816783
---

## 4条件の結論

[全4条件の実測値と基準との差](../../runs/run-i1899-convnext-1query-t64-t64-s42-12k/comparison.md)を1表にまとめた。全条件query12k更新、同じ固定validationの位置教師85,473 frames、最終T64。scored/target/teacher identityと支持数は全4条件で一致している。

- ①は旧CNNのまま4query＋mHCへ変更した条件。mean36.42 pxは基準28.20 pxより8.22 px悪く、P95も82.40 px悪い。medianは0.10 px小さいが、20px以内率は0.99ポイント低い。典型誤差が近いまま裾が悪化しており、query数とmHCは分離できない。
- ②は旧1queryのままCNNをX3D/CSNへ変更した条件。mean33.98 pxで基準より5.78 px悪く、他の3指標も悪い向き。CNNの事前学習も同時に置き換わるため、構造単独の効果とはしない。
- ③は旧ConvNeXt＋1queryの前半をT64にした条件。mean28.53 px（+0.33）、P95120.72 px（+3.83）、20px以内76.17%（−0.12ポイント）。medianは−0.01 pxで、他の差も①②より小さい。前半T64の優位は今回の観測では確認できなかった。

したがって、指定の3変更はいずれも旧基準の平均誤差を改善しなかった。これは単一seed・validationの観察であり、優劣の一般化や統計的な同等性の証明ではない。特に③は途中でL4からRTX5060Tiへ移管し、学習state・データ・recipeを保存していてもGPU間の数値差が交絡する。変更同士の交互作用も調べていない。

## 完了した範囲

①②の既存終端に、③の前半・後半6k checkpoint、固定validation、Drive保存と独立readbackを追加した。source・初期checkpoint・LR/freeze・全12kのpack順とteacher・評価identityを確認し、指定3条件の比較を完了した。各runのcheckpointと予測は元のDrive出力に保持している。指標と差分の丸め前の値は[comparison.json](../../runs/run-i1899-convnext-1query-t64-t64-s42-12k/comparison.json)。

summaryの未完了項目をこの比較へ更新するためsummary_impact=updatedとする。sceneの採用モデルは変更せず、追加の学習や独立testも実施していない。mHC単独の効果、seed分散、独立testでの再現性は未確定のまま残る。
