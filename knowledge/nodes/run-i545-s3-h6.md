---
id: run-i545-s3-h6
type: run
title: i545_s3_h6
issue: 545
provider: claude
session: 8722d9dc-5894-4536-8e54-d03e3e34949a
date: '2026-06-22'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.370544
  position_error_std_m: 0.271984
  position_error_median_m: 0.33261
  angular_error_deg: 8.404209
  angular_error_std_deg: 7.648553
  angular_error_median_deg: 6.277582
  x_error_m: 0.142673
  y_error_m: 0.307894
  z_error_m: 0.047821
  position_accuracy: 0.777196
  angle_accuracy: 0.848366
  position_accuracy_0.5m: 0.777196
  position_accuracy_1m: 0.96638
  position_accuracy_2m: 0.996257
  angle_accuracy_10deg: 0.686814
  angle_accuracy_15deg: 0.848366
  angle_accuracy_30deg: 0.977732
repro:
  commit: 4396ee5bcad62007b6b6ee154cd7e28d995af41b
  branch: feat/training-queue-auto-prune-ckpt
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=6
    model.num_task_layers=3 data=chunked_multiview_sequence_bs8 data.batch_size=8
    data.seq_len_range=[64,256] training.trainer.accumulate_grad_batches=1 loss=canonical_rot
    training.trainer.max_epochs=200 training.early_stopping.enabled=false run.gpus=1
    run.resume=/home/kamimura/projects/wt/i545-prune/outputs/plcs/plcs_multiview_axial_split/logs/version_2/checkpoints/last.ckpt
artifacts:
  run_dir: knowledge/runs/run-i545-s3-h6
  predictions: knowledge/runs/run-i545-s3-h6/pred_test.npz
  log: .training_queue/logs/1782099273755305294_481782_i545_s3_h6.log
  curves: knowledge/runs/run-i545-s3-h6/curves.png
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
スイープ中点寄り（S=3/H=6, 77.845M）。回転 8.40°/位置 0.371m。回転は良好だが位置はスイープ下位＝共有を厚くすると位置がやや痩せる兆候。

### アーキテクチャ詳細
`multiview_axial_split`（`num_layers=6`=共有 H=6、`num_task_layers=3`=分岐 S=3）。H+2S=12、77.845M（EX10 と delta 0.000M）。共有が分岐の倍。スイープ共通条件（chunked, eff batch=8, seq[64,256], canonical_rot, early-stop OFF, 200ep 完遂）。

### メトリクスの解釈
test 回転 mean 8.40°/median 6.28°、位置 mean 0.371m/median 0.333m。位置精度 `@0.5m=0.777` はスイープ下位、角度精度 `@15°=0.848`。curves 正常収束。

### アーキテクチャ⇄メトリクスの因果考察
共有 H を厚く（6）・分岐 S を薄く（3）した構成。回転は維持される一方、位置 readout の分岐容量低下で位置精度が落ちた（仮説）＝位置は分岐容量に、回転は配分にそれほど敏感でない、という非対称の一例。

### 既存実験との比較
バランス点 [[run-i545-s4-h4]]（0.337m）より位置で劣る（0.371）。回転は同程度（8.40 vs 8.23）。親 [[run-i518-exp10]] および早期終了 chunked EX10 との関係はスイープ共通（後者を大きく上回る）。

### 次に有効な実験
位置律速の確認として、S 固定で hidden_dim を上げ位置が改善するか（容量 vs 配分の分離）を検証。
