---
id: run-i539-wide-chunked
type: run
title: i539_wide_chunked
issue: 539
provider: claude
session: 8722d9dc-5894-4536-8e54-d03e3e34949a
date: '2026-06-21'
status: done
config:
  model: multiview_axial_split_wide
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.205996
  position_error_std_m: 0.196673
  position_error_median_m: 0.139112
  angular_error_deg: 10.330529
  angular_error_std_deg: 11.399992
  angular_error_median_deg: 7.61869
  x_error_m: 0.080536
  y_error_m: 0.167964
  z_error_m: 0.039252
  position_accuracy: 0.921283
  angle_accuracy: 0.786782
  position_accuracy_0.5m: 0.921283
  position_accuracy_1m: 0.992914
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.634695
  angle_accuracy_15deg: 0.786782
  angle_accuracy_30deg: 0.957678
repro:
  commit: d407e54cdb903d7082aa4011b2a6f8cb0426c7cc
  branch: exp/i525-asym
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split_wide data.batch_size=2
    training.trainer.accumulate_grad_batches=4 data=chunked_multiview_sequence_bs8
    data.seq_len_range=[64,256] loss=canonical_rot training.trainer.max_epochs=200
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i539-wide-chunked
  predictions: knowledge/runs/run-i539-wide-chunked/pred_test.npz
  log: .training_queue/logs/1782037138499188850_185302_i539_wide_chunked.log
parents:
- run-i535-asym-wide-rerun
- run-i518-exp10
relations:
- to: run-i535-asym-wide-rerun
  rel: supersedes
- to: run-i539-deep16-chunked
  rel: compares
- to: run-i539-ex10-chunked
  rel: compares
- to: run-i518-exp10
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- asymmetric
- width
- chunked
- data-rich
- capacity-frontier
- best
---

## 考察 / Findings

**本群最良。** wide(hidden 768, rot10/pose6, 228.7M)を **chunked**(data-rich)+ **勾配累積で effective batch=8**(bs2×accum4)で学習。seq[64,256], test は固定 scene_dir。

- **test: 回転 10.33°(median 7.62)/ 位置 0.206m(median 0.139)**。**位置は全ラン中最良**(固定 EX10 0.238m・固定 deep16 0.252m を上回る)、回転も固定 EX10(9.98°)と同水準。
- **full 200ep 完走**(early-stop なし)、train ang 5.54°(本群最良)・val 10.60°。大容量が data-rich の新規データを取りきり、最後まで右肩下がりで収束。
- **【@Motoki0705 の仮説を支持】固定データの大崩壊(60.56°)はデータ枯渇＋小バッチの最適化失敗だった。** chunked(データ拡充)＋勾配累積(effective batch 回復)で交絡を外すと、wide は最良に転じた。固定データの結論([[run-i535-asym-wide-rerun]])を supersede する。
- **data-rich では容量が効く / 幅 ≫ 深さ**: 同一 chunked 条件で wide(228.7M, 10.33°/0.206m)≫ ex10(78.1M, 15.84°/0.542m)≫ deep16(142.3M, 19.11°/0.632m)。「データをスケールすれば wide > deep16」は成立。
- 留意: ex10/deep16 は ep86–95 で early-stop しており、本比較は wide に有利な方向の交絡を含む(small モデルの早期停止)。とはいえ wide が full 学習で位置最良に到達した事実は頑健。Phase2(early-stop 緩和・容量/データ量スケーリング曲線)で確定させる。
