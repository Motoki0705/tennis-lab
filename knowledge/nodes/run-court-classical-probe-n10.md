---
id: run-court-classical-probe-n10
type: run
title: 古典コートモデルCPU補助プローブ（実写・B00各10件）
issue: 882
provider: codex
date: '2026-09-16'
status: done
config:
  model: gchlebus_tennis_court_detection_762d077
  loss: inference_only
  data: deterministic_real_n10_and_b00_synthetic_test_n10
  device: cpu_docker_opencv_4_5_4
  thresholds: upstream_defaults
metrics:
  real_process_success_rate: 1.0
  real_finite_16_point_rate: 1.0
  real_all_points_inside_rate: 0.9
  synthetic_process_success_rate: 0.6
  synthetic_process_failure_rate: 0.4
  synthetic_opencv_crash_rate: 0.3
  synthetic_exit_3_rate: 0.1
  b00_test_two_court_fraction: 1.0
repro:
  commit: 762d077541a77abf4923f5f8f689a1410927d35e
  branch: upstream-fixed-commit-with-opencv4-patch
  remote: https://github.com/gchlebus/tennis-court-detection.git
  command: bash knowledge/runs/run-court-classical-probe-n10/repro.sh
artifacts:
  run_dir: knowledge/runs/run-court-classical-probe-n10
  report: knowledge/runs/run-court-classical-probe-n10/REPORT.md
  results: knowledge/runs/run-court-classical-probe-n10/sweep_results.json
  patch: knowledge/runs/run-court-classical-probe-n10/tcd_opencv4.patch
  real_overlay: knowledge/runs/run-court-classical-probe-n10/figures/real_overlay.png
  synthetic_wrong_fit: knowledge/runs/run-court-classical-probe-n10/figures/synthetic_wrong_fit.png
  synthetic_degenerate_fit: knowledge/runs/run-court-classical-probe-n10/figures/synthetic_degenerate_fit.png
parents: []
relations:
- to: run-court-alignment-crossmodel-n120-s42
  rel: compares
tags:
- court-detection
- court-alignment
- classical-baseline
- cpu
- robustness
---

## 考察 / Findings

### 要約

Farin 系の古典的な規制コートモデル当てはめ実装を、既定閾値のまま実写10枚と B00 合成test 10枚へ CPU 適用した。実写は10/10で16有限点を出力した一方、合成は6/10だけが出力まで到達し、残る4/10は1件の通常エラーと3件の OpenCV 例外によるクラッシュだった。出力成功は幾何的成功を意味せず、合成の成功例にも誤fitと縮退fitがあった。

### アーキテクチャ詳細

`gchlebus/tennis-court-detection` commit `762d077541a77abf4923f5f8f689a1410927d35e`（BSD-3-Clause）は、色・線候補から単一の規制コートテンプレートを画像平面へ当てはめる古典手法である。Ubuntu 22.04、CPU版 OpenCV 4.5.4、CUDAデバイスなしの Docker image で実行した。互換patchは Conan依存と旧OpenCV定数を置換し、検証用overlay出力を追加するだけで、検出閾値は変更していない。入力は同一PNGを3フレームのlossless FFV1へ変換し、実装が読む中央フレームとの画素一致を確認した。

### メトリクスの解釈

この補助runは主比較と標本が異なり、古典出力は16点で共通14点の意味付き出力でもないため、PCK表へ混ぜない。実写では10/10がexit 0かつ16有限点だったが、1件は16点中8点しか画像内になく、成功率だけでは外挿破綻を捉えられない。B00合成ではexit 0が6/10、line candidate不足のexit 3が1/10、OpenCV `perspectiveTransform` 例外による異常終了が3/10だった。既定のモデルfitを得られないまま空の変換行列を出力へ渡す経路がクラッシュ原因である。

### アーキテクチャ⇄メトリクスの因果考察

B00 test 238枚はすべて二面のコートを含むのに対し、古典実装は単一コートを前提にする。このscope不一致と、放送映像以外の斜視・部分コートが、線候補選択とテンプレートfitを不安定にしたという仮説が観測と整合する。ただし本runはB00だけの10標本であり、「合成画像全般」や「3DGS画像全般」に対する因果結論ではない。

### 既存実験との比較

主比較 `run-court-alignment-crossmodel-n120-s42` は意味付き14点を同一manifest・同一RANSACへ渡す定量評価である。本runはそれと異なり、古典実装を改造せずにプロセス成功、有限性、画像内点、overlayを調べる robustness probe である。主比較の公開学習モデルも合成でホモグラフィ成功2/120だったが、失敗機構は異なるため同じ列の数値として扱わない。

### 次に有効な実験

単一コートだけを含む合成testを別に固定し、古典16点から規制テンプレートの共通14点へ事前定義した写像を適用して、PCK・共通ホモグラフィ・画像内線再投影を測る。その後に二面コート条件を加えれば、外観・視点のdomain shiftと単一／複数コートscopeの影響を分離できる。
