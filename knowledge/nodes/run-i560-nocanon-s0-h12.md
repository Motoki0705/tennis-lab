---
id: run-i560-nocanon-s0-h12
type: run
title: i545_nocanon_s0_h12
issue: 560
provider: codex
session: 019ef7e7-03a8-7e53-9c92-21cdcb326a79
date: '2026-06-26'
status: done
config:
  model: multiview_axial_base
  loss: no_canonical
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.282941
  position_error_std_m: 0.374798
  position_error_median_m: 0.194387
  angular_error_deg: 50.344444
  angular_error_std_deg: 56.385094
  angular_error_median_deg: 23.359777
  x_error_m: 0.110277
  y_error_m: 0.237108
  z_error_m: 0.034487
  position_accuracy: 0.911795
  angle_accuracy: 0.377725
  position_accuracy_0.5m: 0.911795
  position_accuracy_1m: 0.966259
  position_accuracy_2m: 0.985906
  angle_accuracy_10deg: 0.269931
  angle_accuracy_15deg: 0.377725
  angle_accuracy_30deg: 0.556883
repro:
  commit: ed6eef1b4fbfa5431d1ce40f3a010e90115b09fb
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_base model.num_layers=12
    model.predict_canonical_pose=false data=chunked_multiview_sequence_bs8 data.batch_size=8
    training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256] loss=no_canonical
    training.trainer.max_epochs=200 training.early_stopping.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i560-nocanon-s0-h12
  predictions: knowledge/runs/run-i560-nocanon-s0-h12/pred_test.npz
  log: .training_queue/logs/1782307133788233852_638972_i545_nocanon_s0_h12.log
  output_dir: outputs/plcs/plcs_multiview_axial/logs/version_33
  curves: knowledge/runs/run-i560-nocanon-s0-h12/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_33
parents:
- run-i545-s0-h12
relations:
- to: run-i545-s0-h12
  rel: compares
- to: run-i560-nocanon-rs-s0-h12
  rel: compares
tags:
- plcs
- no-canonical
- shared-trunk
- chunked
- data-rich
- trunk-allocation
- loss-tuning
- strict-no-canonical
---

## 考察 / Findings

### 要約
S=0/H=12 純共有（base モデル）で strict no-canonical。位置 0.664m→0.283m に回復するが、**閾値 0.27m はわずかに割れず**、回転も 50.34° に崩壊。完全共有では strict 化しても位置が 0.27m 以下に届かない＝共有が極端だと位置にも下限が残る。

### アーキテクチャ詳細
`multiview_axial_base`（分岐 trunk なし、num_layers=12 の単一共有 trunk）。`predict_canonical_pose=false`, `loss=no_canonical`（rotation≈0.02）。#545 の純共有端点 [[run-i545-s0-h12]] と同モデル・同データ。

### メトリクスの解釈
位置 mean 0.283 / median 0.194、位置@0.5m=0.912。回転 mean 50.34 / median 23.36 と崩壊。strict 化の位置改善（0.664→0.283）は大きいが、split 系（0.18m）には届かない。

### アーキテクチャ⇄メトリクスの因果考察
分岐 trunk が無いため、位置の表現も rot/pose readout も全て単一 trunk を共有。rotation 教師を切っても、位置専用容量が確保できず 0.27m 手前で頭打ち（仮説）。分岐 trunk を少しでも持つ split 系（H≤4）が strict で 0.18m に達するのと対照的。

### 既存実験との比較
- baseline [[run-i545-s0-h12]]（10.85°/0.664m）から位置回復・回転崩壊。
- split の strict（[[run-i560-nocanon-s5-h2]] 0.181m, [[run-i560-nocanon-s4-h4]] 0.185m）に位置で劣る＝**分岐 trunk の有無**が strict 下の位置下限を決める。
- Wave B [[run-i560-nocanon-rs-s0-h12]] では位置 0.584m に再悪化。

### 次に有効な実験
- 純共有は位置・回転とも常に劣勢で候補外（#545 の結論を no-canonical でも追認）。最低 1 層の分岐を持たせるべき。
