---
id: run-i539-ex10-chunked
type: run
title: i539_ex10_chunked
issue: 539
provider: claude
session: 8722d9dc-5894-4536-8e54-d03e3e34949a
date: '2026-06-21'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.542376
  position_error_std_m: 0.319346
  position_error_median_m: 0.487207
  angular_error_deg: 15.839146
  angular_error_std_deg: 14.926085
  angular_error_median_deg: 11.135011
  x_error_m: 0.268624
  y_error_m: 0.409325
  z_error_m: 0.053363
  position_accuracy: 0.509147
  angle_accuracy: 0.624715
  position_accuracy_0.5m: 0.509147
  position_accuracy_1m: 0.903832
  position_accuracy_2m: 0.997239
  angle_accuracy_10deg: 0.458958
  angle_accuracy_15deg: 0.624715
  angle_accuracy_30deg: 0.852813
repro:
  commit: d407e54cdb903d7082aa4011b2a6f8cb0426c7cc
  branch: exp/i525-asym
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split data.batch_size=8
    training.trainer.accumulate_grad_batches=1 data=chunked_multiview_sequence_bs8
    data.seq_len_range=[64,256] loss=canonical_rot training.trainer.max_epochs=200
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i539-ex10-chunked
  predictions: knowledge/runs/run-i539-ex10-chunked/pred_test.npz
  log: .training_queue/logs/1782037138466268778_185272_i539_ex10_chunked.log
parents:
- run-i518-exp10
relations:
- to: run-i518-exp10
  rel: compares
- to: run-i539-wide-chunked
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- chunked
- data-rich
- capacity-frontier
---

## 考察 / Findings

EX10(対称 6 層/512, 78.1M)を **chunked backend**(生成データを連続供給)で再学習した #539 基準。effective batch=8(bs8×accum1), `seq_len_range=[64,256]`(固定側と一致), test/val は固定 `scene_dir` から(chunked と固定で同一 test → 直接比較可)。

- **test: 回転 15.84°(median 11.14)/ 位置 0.542m**。**固定データ EX10(9.98°/0.238m)より悪化**。
- **ep95 で early-stop**(200 未満)。train ang 13.50° ≒ val 15.60° で**過学習ではなく未収束**。固定小データ(≈800窓を 200ep 反復＝記憶可能)と違い、chunked は毎チャンク新規 1000 scene を供給するため記憶が効かず、78M ではこの高分散ストリームを 95ep で取りきれず val/pos が plateau して early-stop。
- 同一 chunked 条件での容量比較では **wide(228.7M, 10.33°/0.206m)に大きく劣る**。data-rich 体制では小容量が不利＝[[run-i539-wide-chunked]] 参照。
- 注意: early-stop×chunk ローテーション(10ep ごとに分布シフト)の相互作用で early-stop が早まった可能性があり、本値はやや悲観的。patience 拡大での再確認は #539 Phase2。
