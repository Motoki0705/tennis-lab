---
id: run-i535-asym-wide-rerun
type: run
title: i535_asym_wide_rerun
issue: 535
provider: claude
session: 8722d9dc-5894-4536-8e54-d03e3e34949a
date: '2026-06-21'
status: done
config:
  model: multiview_axial_split_wide
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 0.894287
  position_error_std_m: 0.512892
  position_error_median_m: 0.808025
  angular_error_deg: 60.562759
  angular_error_std_deg: 52.051292
  angular_error_median_deg: 40.625477
  x_error_m: 0.441681
  y_error_m: 0.676823
  z_error_m: 0.05689
  position_accuracy: 0.256967
  angle_accuracy: 0.192595
  position_accuracy_0.5m: 0.256967
  position_accuracy_1m: 0.646212
  position_accuracy_2m: 0.9633
  angle_accuracy_10deg: 0.127633
  angle_accuracy_15deg: 0.192595
  angle_accuracy_30deg: 0.387282
repro:
  commit: 6d24b4dccad36a73bd56a526602757e16bac0275
  branch: exp/i525-asym
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split_wide loss=canonical_rot
    data=multiview_sequence data.batch_size=2 training.trainer.max_epochs=200 run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i535-asym-wide-rerun
  predictions: knowledge/runs/run-i535-asym-wide-rerun/pred_test.npz
  log: .training_queue/logs/1781994475592903754_1061691_i535_asym_wide_rerun.log
parents:
- run-i518-exp10
- run-i525-asym
relations:
- to: run-i540-asym-wide
  rel: supersedes
- to: run-i535-asym-deep16-rerun
  rel: compares
- to: run-i518-exp10
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- asymmetric
- width
- capacity-frontier
- valid-rerun
- under-converged
---

## 考察 / Findings

正しい worktree(`exp/i525-asym`, commit `6d24b4d`)で再実行した、**有効**な非対称幅+深さ計測。hidden 768 / heads 12、rot 10 層 / pose 6 層 ＝ **228.7M params**(起動ログで確認)。no-op だった旧 `run-i540-asym-wide` の対称 768(172M)とは**別物**で、本ノードがそれを supersede する。

- **test: 回転 60.56°(median 40.63°)/ 位置 0.894m** ＝ ほぼ学習破綻(回転はランダム水準)。
- **未収束(最適化失敗)**: early stopping で **ep92** 終了、**train ang すら 40.10°(train loss 0.61)止まり**。228.7M を batch=2(VRAM 16GB 制約)で回したため、過小バッチで勾配が不安定 + 過大容量 × 固定小データ で最適化に失敗した。
- **解釈の注意**: これは「容量(幅)が無効」の証拠**ではない**。データ枯渇 + 小バッチによる最適化失敗であり、@Motoki0705 の「データ不足・未収束で精度が出ない」という指摘と整合する。固定小データでの公平な幅 vs 深さ判定にはなり得ない。
- **当初結論の撤回**: 旧 `run-i540-asym-wide`(12.27°)は no-op で対称 768(172M)を学習しただけ。公平な容量判定は **#539**(chunked + 勾配累積で effective batch を 8 に戻す)で行う。
