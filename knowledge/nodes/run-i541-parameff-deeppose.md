---
id: run-i541-parameff-deeppose
type: run
title: i541_parameff_deeppose
issue: 536
provider: claude
session: d22b7d68-7d91-4a6f-862d-434085e5d2d9
date: '2026-06-20'
status: done
config:
  model: multiview_axial_split_deeppose
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 0.201796
  position_error_std_m: 0.234454
  position_error_median_m: 0.143698
  angular_error_deg: 9.553147
  angular_error_std_deg: 9.066118
  angular_error_median_deg: 7.296664
  x_error_m: 0.082538
  y_error_m: 0.159862
  z_error_m: 0.039267
  position_accuracy: 0.956659
  angle_accuracy: 0.8136
  position_accuracy_0.5m: 0.956659
  position_accuracy_1m: 0.983177
  position_accuracy_2m: 0.995441
  angle_accuracy_10deg: 0.642032
  angle_accuracy_15deg: 0.8136
  angle_accuracy_30deg: 0.96873
repro:
  commit: 6399aa6f6848994957412eebabf4a4330c95cd15
  branch: feat/issue-533-experiment-log-format
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.plcs.scripts.train
    model=multiview_axial_split_deeppose loss=canonical_rot data=multiview_sequence
    training.trainer.max_epochs=200 run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i541-parameff-deeppose
  predictions: knowledge/runs/run-i541-parameff-deeppose/pred_test.npz
  log: .training_queue/logs/1781927076593996965_760047_i541_parameff_deeppose.log
  curves: knowledge/runs/run-i541-parameff-deeppose/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_2
parents:
- run-i525-parameff
- run-i518-exp10
relations:
- to: run-i525-parameff
  rel: contradicts
- to: run-i541-parameff-medcap
  rel: compares
- to: run-i518-exp10
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- parameter-efficiency
- depth
- efficiency-frontier
---

## 考察 / Findings

### 要約
縮小 split の幅は据え置き深さだけ 3→6 層に増やした構成（200ep, 19.5M）。位置 0.202m / 回転 9.55° で効率フロンティア最良、EX10 を約 1/4 の容量で凌駕。

### アーキテクチャ詳細
`multiview_axial_split_deeppose` + `canonical_rot`：`hidden_dim 256` / `num_heads 4`（`eff` と同一の幅）で深さのみ 6 層、約 19.5M params（EX10 の ~25%）。`max_epochs=200`。

### メトリクスの解釈
位置 `0.202m` / 回転 `9.55°`。回転は EX10 (`9.98°`) とほぼ同等、位置は EX10 (`0.238m`) を上回る。これを EX10 の約 1/4（19.5M）で達成。

### アーキテクチャ⇄メトリクスの因果考察
eff の回転ギャップは「幅」ではなく「深さ」だった。ベースライン [[run-i525-parameff]]（256 幅, 3 層, 9.9M, `15.55°/0.569m`）は幅を一切変えず深さを 6 層にしただけで回転 `15.55→9.55°`・位置 `0.569→0.202m` と両方が大幅改善—「回転は容量（幅）依存」の読みを反証。split-trunk の構造的利得（位置）に深さを足すと回転も小容量で EX10 水準に届く。

### 既存実験との比較
親 [[run-i525-parameff]] の結論を反証（`contradicts`）。[[run-i518-exp10]] を効率で凌駕（`compares`）。非対称側の [[run-i540-asym-deep16]] と別容量帯で揃って「深さが回転の主レバー」を示す。#525 の「回転=容量依存」は「回転=深さ依存（幅ではない）」へ精緻化。

### 次に有効な実験
deeppose を新しい効率ベースラインとして、(1) 深さ 6→8 層でさらに回転が伸びるか、(2) 19.5M 未満（幅 192 等）でも深さで EX10 水準を保てるか、効率フロンティアの下限を再探索する。
