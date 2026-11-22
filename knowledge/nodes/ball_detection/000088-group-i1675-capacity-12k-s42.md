---
id: group-i1675-capacity-12k-s42
type: group
task: ball_detection
sequence: 88
recorded_at: '2026-11-12'
title: mHC query容量7条件をquery累積12kまで継続した固定validation比較
issue:
- 1675
- 1913
provider: claude
date: '2026-11-11'
status: done
summary_impact: updated
members:
- run-i1675-capacity-d256-l4-t64-s42-12k
- run-i1675-capacity-d384-l4-t64-s42-12k
- run-i1675-capacity-d512-l4-t64-s42-12k
- run-i1675-capacity-d768-l4-t64-s42-12k
- run-i1675-capacity-d1024-l4-t64-s42-12k
- run-i1675-capacity-d512-l6-t64-s42-12k
- run-i1675-capacity-d512-l8-t64-s42-12k
parents:
- group-i1675-capacity-scaling-s42-52k
relations:
- to: run-ball-sequential-length-20261025
  rel: compares
  note: 旧#1514のquery T64（query累積12k、mean 28.20px）と同じvalidationで比較。CNN・query数・前半Tが異なる
config:
  seed: 42
  query_updates: 12000
  conditions: 7
  held_conditions: [d512-l12, d512-l24]
metrics:
  best_mean_error_px: 32.482105524036534
  baseline_1514_query_t64_mean_error_px: 28.200801773291
papers: []
tags:
- capacity
- mhc
- validation
- single-seed
- no-deployment-change
---

## 何を確かめたか

[6kの容量比較](000078-group-i1675-capacity-scaling-s42-52k.md)では、7条件の最良（d512・6層）でもmean 39.47pxで、旧#1514のmHCなしquery T64（28.20px）に届かなかった。旧モデルはquery系統の累積が12k（T32 6k→T64 6k）で、LRも最終1e-6まで減衰していた。[2026-11-10の合意](https://github.com/Motoki0705/tennis-lab/issues/1675#issuecomment-6758820955)は、学習量とLRの終点の差が原因かを見るため、7条件を各12kまで継続した。同じ合意の12層・24層は[#1899の方針変更](https://github.com/Motoki0705/tennis-lab/issues/1899#issuecomment-6787127916)で保留となり、学習していない。

## 結果（固定validation、seed42、observed 85,473 frames、source px）

| 条件 | 総parameter | 6k mean | 12k mean | 12k median | 12k P95 | 12k 20px以内 |
|---|---:|---:|---:|---:|---:|---:|
| d256・4層 | 8,025,638 | 48.32 | 35.38 | 9.54 | 169.69 | 73.62% |
| d384・4層 | 13,311,014 | 45.88 | 35.11 | 9.56 | 167.50 | 74.51% |
| d512・4層 | 21,021,222 | 48.73 | 33.58 | 8.44 | 164.14 | 75.79% |
| d768・4層 | 41,946,662 | 43.45 | 33.88 | 9.56 | 160.81 | 75.87% |
| d1024・4層 | 71,784,998 | 53.96 | 33.38 | 8.64 | 161.51 | 77.00% |
| d512・6層 | 29,673,008 | 39.47 | 34.26 | 9.23 | 162.06 | 75.24% |
| d512・8層 | 38,324,794 | 42.20 | 32.48 | 9.01 | 149.47 | 75.59% |

- 7条件すべてで、6kから12kへmeanが下がった。12kの最良はd512・8層の32.48px。12kの7条件のmeanは32.48〜35.38pxの約2.9pxの幅に収まり、6kでの約14.5pxの幅（39.47〜53.96px）より大きく縮んだ。
- 6kで最も悪かったd1024・4層（53.96px）は、12kでは33.38pxで2番目に良い。6kでの容量の順位は、学習途中の差を大きく含んでいたと考えられる。容量（総parameter 8.0M〜71.8M）とmeanの間に単調な関係は見えない。
- 旧#1514のquery T64（mean 28.20、median 7.66、P95 116.89、20px以内76.29%）と比べると、12kの最良でもmeanとP95は悪い。medianも7条件すべてが旧モデルより大きい（8.44〜9.56px）。20px以内率はd1024・4層（77.00%）だけが旧モデルを上回り、他の6条件は下回る。

## 解釈と限界

- 学習量とLRをそろえると差は縮んだが、mHC query 7条件はmeanで旧#1514のquery T64に届かなかった。ただしCNN（X3D/CSN対ConvNeXt）、query数（4対1）、前半の系列長（T64対T32）が同時に異なるので、mHCの有無の効果とはしない。この切り分けは#1899で行っている（[ConvNeXt＋4query/mHC](000089-run-i1899-convnext-4query-mhc-t64-s42-12k.md)、[X3D/CSN＋1query](000090-run-i1899-x3d-1query-t64-s42-12k.md)）。
- 単一seed・validationだけの比較で、seed間のばらつき・独立test・統計的な差は評価していない。12kでの約2.9pxの幅は、seed差で入れ替わりうる大きさとして扱う。
- d512・8層は12kの途中（5,000更新）で一度止まり、full stateから再開した。他の6条件は再開なし。
- 評価JSON 7件のSHA-256はDriveの`_SNAPSHOT.json`の記載と一致した（各runの`knowledge/runs/<run ID>/`に保存）。checkpoint本体の再load・学習ログの再集計はしていない。
- sceneの採用モデル・checkpointは変更しない。summaryのBall Detection節に12kの結果と限界を反映した。
