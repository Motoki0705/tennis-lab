---
id: run-i545-s2-h8
type: run
title: i545_s2_h8
issue: 545
provider: claude
session: 8722d9dc-5894-4536-8e54-d03e3e34949a
date: '2026-06-23'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.354113
  position_error_std_m: 0.274708
  position_error_median_m: 0.291546
  angular_error_deg: 8.353188
  angular_error_std_deg: 7.550225
  angular_error_median_deg: 6.206733
  x_error_m: 0.133641
  y_error_m: 0.294171
  z_error_m: 0.045838
  position_accuracy: 0.786599
  angle_accuracy: 0.859486
  position_accuracy_0.5m: 0.786599
  position_accuracy_1m: 0.987887
  position_accuracy_2m: 0.994047
  angle_accuracy_10deg: 0.709974
  angle_accuracy_15deg: 0.859486
  angle_accuracy_30deg: 0.97432
repro:
  commit: 674818c567169bd3bee4bab17dd417a7308fdcc6
  branch: feat/training-queue-auto-prune-ckpt
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=8
    model.num_task_layers=2 data=chunked_multiview_sequence_bs8 data.batch_size=8
    data.seq_len_range=[64,256] training.trainer.accumulate_grad_batches=1 loss=canonical_rot
    training.trainer.max_epochs=200 training.early_stopping.enabled=false run.gpus=1
    run.resume=/home/kamimura/projects/wt/i545-prune/outputs/plcs/plcs_multiview_axial_split/logs/version_4/checkpoints/last.ckpt
artifacts:
  run_dir: knowledge/runs/run-i545-s2-h8
  predictions: knowledge/runs/run-i545-s2-h8/pred_test.npz
  log: .training_queue/logs/1782099273773937194_481797_i545_s2_h8.log
  curves: knowledge/runs/run-i545-s2-h8/curves.png
parents:
- run-i518-exp10
relations:
- to: run-i545-s4-h4
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- chunked
- data-rich
- trunk-allocation
- param-matched
---

## 考察 / Findings

### 要約
共有寄り（S=2/H=8, 77.845M）。回転 8.35°/位置 0.354m。回転はスイープ中位で安定、位置も中位。共有を厚くしても回転は崩れないことを示す。

### アーキテクチャ詳細
`multiview_axial_split`（`num_layers=8`=共有 H=8、`num_task_layers=2`=分岐 S=2）。H+2S=12、77.845M（EX10 と delta 0.000M）。共有が分岐の 4 倍。スイープ共通条件（chunked, eff batch=8, seq[64,256], canonical_rot, early-stop OFF, 200ep 完遂）。

### メトリクスの解釈
test 回転 mean 8.35°/median 6.21°、位置 mean 0.354m/median 0.292m。位置精度 `@1m=0.988`（スイープ最高）だが `@0.5m=0.787` は中位。角度精度 `@15°=0.859`。curves 正常収束。

### アーキテクチャ⇄メトリクスの因果考察
共有を厚く（H=8）しても回転は崩れない（8.35°）＝低レベル共有でも回転 readout は薄い分岐 S=2 で足りる（仮説）。位置の細かさ（@0.5m）は分岐容量低下でやや低下するが、粗い精度（@1m）はむしろ最高で、共有特徴が大域的な位置推定に効いている可能性。

### 既存実験との比較
バランス点 [[run-i545-s4-h4]] に僅差で劣る（回転 8.35 vs 8.23、位置 0.354 vs 0.337）。さらに共有を厚くした [[run-i545-s1-h10]] より位置で勝る（0.354<0.385）＝共有過多の手前の安定域。

### 次に有効な実験
H をさらに増やした極限（S=0 純共有 = base model）端点を取得し、カーブの共有側端を確定する。
