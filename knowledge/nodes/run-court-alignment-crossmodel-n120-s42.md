---
id: run-court-alignment-crossmodel-n120-s42
type: run
title: コートアライメント交差モデルCPU比較（各120件）
issue: 882
provider: codex
date: '2026-09-16'
status: done
config:
  model: court_detection_epoch17_vs_yastrebksv_tcd
  loss: inference_only
  data: real_validation_n120_and_synthetic_test_n120_seed42
  device: cpu
  pck_diagonal_fractions: [0.005, 0.01, 0.02, 0.05]
  ransac_diagonal_fraction: 0.01
  line_samples_per_segment: 21
metrics:
  real_ours_completeness: 1.0
  real_ours_pck_001: 1.0
  real_ours_pair_median_px: 2.915476
  real_ours_h_success: 1.0
  real_tcd_completeness: 0.995218
  real_tcd_pck_001: 0.977884
  real_tcd_pair_median_px: 3.0
  real_tcd_h_success: 1.0
  synthetic_ours_completeness: 1.0
  synthetic_ours_pck_001: 0.931268
  synthetic_ours_pair_median_px: 3.650254
  synthetic_ours_h_success: 1.0
  synthetic_tcd_completeness: 0.028074
  synthetic_tcd_pck_001: 0.001936
  synthetic_tcd_pair_median_px: 66.82623
  synthetic_tcd_h_success: 0.016667
repro:
  commit: b780aeb71db6bacacaf2c1fa2a74ac2087a33efa
  branch: codex/court-alignment-paper
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: bash knowledge/runs/run-court-alignment-crossmodel-n120-s42/repro.sh
artifacts:
  run_dir: knowledge/runs/run-court-alignment-crossmodel-n120-s42
  predictions: knowledge/runs/run-court-alignment-crossmodel-n120-s42/predictions
  manifest: knowledge/runs/run-court-alignment-crossmodel-n120-s42/manifest.json
  metrics: knowledge/runs/run-court-alignment-crossmodel-n120-s42/metrics.json
  provenance: knowledge/runs/run-court-alignment-crossmodel-n120-s42/provenance.json
  summary_figure: knowledge/runs/run-court-alignment-crossmodel-n120-s42/figures/summary_bars.png
  real_montage: knowledge/runs/run-court-alignment-crossmodel-n120-s42/figures/montage_real_validation.png
  synthetic_montage: knowledge/runs/run-court-alignment-crossmodel-n120-s42/figures/montage_synthetic_test.png
parents: []
relations: []
tags:
- court-detection
- court-alignment
- 3dgs
- synthetic-data
- benchmark
- cpu
---

## 考察 / Findings

### 要約

同一の決定論的 manifest（seed 42、実写 validation 120 枚、合成 test 120 枚）で、提案 checkpoint と公開 TennisCourtDetector を CPU 比較した。実写では両モデルとも高精度だったが、合成では提案モデルの PCK@0.01 が `0.931268`、共通ホモグラフィ成功率が `1.0` だったのに対し、公開モデルはそれぞれ `0.001936`、`0.016667` だった。

### アーキテクチャ詳細

提案側は epoch 17 の DINOv3 ViT-B/16 ベース checkpoint で、実写と 3DGS 由来合成データを混合学習している。公開側は yastrebksv/TennisCourtDetector の固定 commit `e5cd4f1ce26b15361700d3d89e068cbf0e82749e` と公式重みを用い、640×360 BGR 入力、14 点と補助中心点の TrackNet 系ヒートマップを上流既定の Hough 後処理で復号した。検出器固有のホモグラフィ実装は使わず、両者の14点を同一の規制コートテンプレートと RANSAC 登録器へ入力した。

実写側は公開データの held-out validation であり、公式 test とは呼ばない。合成側は B00--B03 の trajectory-group-disjoint test で、同じ会場群の未学習軌道であって未知会場ではない。実写の可視性は画像内、合成の可視性はカメラ前方・画像内・レンダラ支持の積なので、completeness の絶対値をドメイン間では比較しない。

### メトリクスの解釈

PCK の分母は GT-visible 点で、検出不能点も不正解に含めた。実写では提案側／公開側の completeness が `1.000000 / 0.995218`、PCK@0.01 が `1.000000 / 0.977884`、対応のある点だけの誤差中央値が `2.915 / 3.000 px`、共通ホモグラフィ成功率はいずれも `120/120` だった。公開側の画像内線再投影誤差は中央値 `2.104 px` に対し平均 `24.804 px` で、可視 GT 10 点のうち 5 点しか検出しない1標本の約 `2691 px` が平均を押し上げた。

合成では提案側が全 `1,033` 可視点を報告し、PCK@0.01 `0.931268`、点誤差中央値 `3.650 px`、ホモグラフィ成功 `120/120` を得た。公開側は completeness `0.028074`、PCK@0.01 `0.001936`、有効な29対応だけの中央値 `66.826 px`、成功 `2/120` で、117標本は有効点不足、1標本は RANSAC 失敗だった。B01--B03 では有効検出が0だった。

提案側でも合成の画像内線誤差は中央値 `7.426 px` に対して q90 `199.272 px` と裾が重い。画像内へ clip した doubles IoU は50/120標本でのみ定義でき、中央値 `0.949045` である。残る70標本を除外した平均だけで全体性能を表さず、画像内線被覆率 `0.712668` と併記する必要がある。

### アーキテクチャ⇄メトリクスの因果考察

公開モデルが実写で高精度なのに合成でほぼ検出不能になったことは、学習分布に再構成由来の斜視・部分コート・複数コートが無いというドメインシフト仮説と整合する。提案側が合成 test で可視点を落とさなかったことは、3DGS 由来教師を含む混合学習が同一会場群の新規軌道に対応した可能性を示す。ただしアーキテクチャ、入力処理、学習データが同時に異なるため、3DGS 拡張の因果効果とは断定しない。

提案側の部分コートで線外挿誤差が増えるのは、少数の可視点から画面外の規制テンプレート全体を外挿するホモグラフィの条件が悪化するため、という仮説である。実際、coverage=`full / near_full / partial` の画像内線誤差中央値は `4.244 / 6.332 / 11.377 px` と増加した。ただし q90 の大きさには点配置や RANSAC 内点構成も影響するため、可視範囲だけの効果とは断定しない。

### 既存実験との比較

この checkpoint に対応する学習 run は知識グラフへ未登録のため、`parents` は空とした。既存の mixed-source court detection 群と同じく実写・合成を混合するが、本 run は学習指標ではなく、外部公開モデルと同一標本・同一登録器で比較した初の評価である。checkpoint 選択時の synthetic validation（KP 平均 `1.6223 px`）とは標本、可視性、集約方法が異なるので直接比較しない。

### 次に有効な実験

同一アーキテクチャで (1) 実写のみ、(2) 同枚数の狭視点合成、(3) 広視点合成を複数 seed 学習し、既知会場の未学習軌道と未知会場を分けて評価する。さらに部分コートでは、全テンプレート外挿とは別に GT 投影が画像内にある線だけの robust 分位点を主指標とし、可視点数・点配置・coverage mode ごとの失敗率を報告する。
