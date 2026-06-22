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
  curves: knowledge/runs/run-i539-ex10-chunked/curves.png
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

### 要約
EX10（対称6層/512, 78.1M）を chunked（data-rich）で再学習した #539 基準。回転 15.84°/位置 0.542m で**固定データ EX10（9.98°/0.238m）より悪化**、ep95 early-stop。

### アーキテクチャ詳細
`multiview_axial_split`（EX10）+ `canonical_rot` + `data=chunked_multiview_sequence_bs8`。effective batch=8（bs8×accum1）、`data.seq_len_range=[64,256]`（固定側と一致）。**train のみ** chunked 生成（毎チャンク 1000 scene を回す）、val/test は固定 `scene_dir` から取るため固定ランと同一 test で直接比較可。`exp/i525-asym` worktree（commit `6d24b4d`）。

### メトリクスの解釈
test 回転 `15.84°`（median `11.14°`）/ 位置 `0.542m`。curves.png: **ep95 で early-stop**、train ang `13.50°` ≒ val `15.60°`（過学習ではなく未収束）。注目すべきは、監視指標 `val/pos_error_m` が step ~2000 で頭打ちした一方 **`val/ang_error_deg` はまだ低下中**だった点 ＝ 位置の plateau により回転改善の途中で停止している。

### アーキテクチャ⇄メトリクスの因果考察
固定小データは ≈800 窓を 200ep 反復＝記憶可能だが、chunked は新規 scene を供給し続け記憶が効かない高分散ストリーム。78M ではこれを 95ep で取りきれず未収束のまま early-stop したと解釈（仮説）。early-stop（monitor=`val/pos`, patience 10）× chunk ローテーション（10ep ごと分布シフト）の相互作用で停止が早まった可能性が高い。

### 既存実験との比較
固定データ [[run-i518-exp10]]（9.98°/0.238m）より悪化（`compares`）＝ 小容量は data-rich でかえって不利。同一 chunked 条件で [[run-i539-wide-chunked]]（228.7M, 10.33°/0.206m）に明確に劣る。

### 次に有効な実験
early-stopping を緩和/無効化（または monitor を `val/ang` に）して full 学習で取り直し、early-stop 交絡を排した chunked EX10 基準を確定する（#539 Phase2）。
- 同一 chunked 条件での容量比較では **wide(228.7M, 10.33°/0.206m)に大きく劣る**。data-rich 体制では小容量が不利＝[[run-i539-wide-chunked]] 参照。
- 注意: early-stop×chunk ローテーション(10ep ごとに分布シフト)の相互作用で early-stop が早まった可能性があり、本値はやや悲観的。patience 拡大での再確認は #539 Phase2。
