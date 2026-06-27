---
id: run-i560-nocanon-rs-s0-h12
type: run
title: i545_nocanon_rs_s0_h12
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
  position_error_m: 0.583569
  position_error_std_m: 0.516808
  position_error_median_m: 0.453323
  angular_error_deg: 10.410501
  angular_error_std_deg: 14.779783
  angular_error_median_deg: 6.456656
  x_error_m: 0.216347
  y_error_m: 0.504597
  z_error_m: 0.048715
  position_accuracy: 0.556767
  angle_accuracy: 0.826206
  position_accuracy_0.5m: 0.556767
  position_accuracy_1m: 0.883267
  position_accuracy_2m: 0.988118
  angle_accuracy_10deg: 0.679572
  angle_accuracy_15deg: 0.826206
  angle_accuracy_30deg: 0.947836
repro:
  commit: ed6eef1b4fbfa5431d1ce40f3a010e90115b09fb
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_base model.num_layers=12
    model.predict_canonical_pose=false data=chunked_multiview_sequence_bs8 data.batch_size=8
    training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256] loss=no_canonical
    loss.rotation_weight=0.5 +loss.angle_weight=1.0 training.trainer.max_epochs=200
    training.early_stopping.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i560-nocanon-rs-s0-h12
  predictions: knowledge/runs/run-i560-nocanon-rs-s0-h12/pred_test.npz
  log: .training_queue/logs/1782307133857545536_638926_i545_nocanon_rs_s0_h12.log
  output_dir: outputs/plcs/plcs_multiview_axial/logs/version_34
  curves: knowledge/runs/run-i560-nocanon-rs-s0-h12/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_34
parents:
- run-i560-nocanon-s0-h12
relations:
- to: run-i560-nocanon-s0-h12
  rel: compares
- to: run-i545-s0-h12
  rel: compares
tags:
- plcs
- no-canonical
- shared-trunk
- chunked
- data-rich
- trunk-allocation
- loss-tuning
- rot-strong
---

## 考察 / Findings

### 要約
S=0/H=12 純共有（base）で canonical head 無し ＋ rot-strong。回転 10.41°/位置 0.584m。回転は base としては戻る（strict 50.34°→10.41°）が、位置は strict 0.283m から 0.584m に再悪化し全構成中ほぼ最悪。純共有は loss を変えても位置・回転とも劣勢。

### アーキテクチャ詳細
`multiview_axial_base`（単一共有 trunk、分岐なし）。`predict_canonical_pose=false`, `loss=no_canonical` ＋ `rotation_weight=0.5` `+angle_weight=1.0`。strict 版 [[run-i560-nocanon-s0-h12]] との差は rotation/angle weight のみ。

### メトリクスの解釈
回転 mean 10.41 / median 6.46、角@15=0.826。位置 mean 0.584 / median 0.453、位置@0.5m=0.557（研究内ワースト級）。

### アーキテクチャ⇄メトリクスの因果考察
分岐 trunk が無いため rotation を戻すと単一 trunk が rotation に占有され位置が大きく崩れる（0.283→0.584m）。位置専用容量を確保できない純共有の構造的弱点が、loss 変更では救えないことを示す（仮説）。

### 既存実験との比較
- baseline [[run-i545-s0-h12]]（10.85°/0.664m）と同水準（回転微改善・位置やや改善だが依然劣勢）。
- 同 rot-strong の split（[[run-i560-nocanon-rs-s6-h0]] 8.49°/0.207m, [[run-i560-nocanon-rs-s5-h2]] 7.54°/0.332m）に全面的に劣る＝分岐 trunk の有無が決定的。

### 次に有効な実験
- 純共有は候補外で確定。これ以上の純共有探索は不要。
