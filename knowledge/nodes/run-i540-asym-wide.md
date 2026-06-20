---
id: run-i540-asym-wide
type: run
title: i540_asym_wide
issue: 535
provider: claude
session: d22b7d68-7d91-4a6f-862d-434085e5d2d9
date: '2026-06-20'
status: done
config:
  model: multiview_axial_split_wide
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 0.367522
  position_error_std_m: 0.38396
  position_error_median_m: 0.289272
  angular_error_deg: 12.274314
  angular_error_std_deg: 10.713239
  angular_error_median_deg: 9.571921
  x_error_m: 0.127536
  y_error_m: 0.311158
  z_error_m: 0.04342
  position_accuracy: 0.800577
  angle_accuracy: 0.702749
  position_accuracy_0.5m: 0.800577
  position_accuracy_1m: 0.959503
  position_accuracy_2m: 0.986992
  angle_accuracy_10deg: 0.518218
  angle_accuracy_15deg: 0.702749
  angle_accuracy_30deg: 0.933957
repro:
  commit: 6399aa6f6848994957412eebabf4a4330c95cd15
  branch: feat/issue-533-experiment-log-format
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.plcs.scripts.train
    model=multiview_axial_split_wide loss=canonical_rot data=multiview_sequence training.trainer.max_epochs=200
    run.gpus=1 run.resume=/home/kamimura/projects/tennis-lab/outputs/plcs/plcs_multiview_axial_split/logs/version_3/checkpoints/last.ckpt
artifacts:
  run_dir: knowledge/runs/run-i540-asym-wide
  predictions: knowledge/runs/run-i540-asym-wide/pred_test.npz
  log: .training_queue/logs/1781927908633902411_828149_i540_asym_wide_resume.log
  curves: knowledge/runs/run-i540-asym-wide/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_4
parents:
- run-i525-asym
- run-i518-exp10
relations:
- to: run-i540-asym-deep16
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
---

## 考察 / Findings

### 要約
両 trunk の幅を 768 まで広げた非対称構成（200ep, resume 要）。幅は効くが高くつき、EX10 にも deep16 にも届かない。回転改善は幅 < 深さ。

### アーキテクチャ詳細
`multiview_axial_split_wide` + `canonical_rot`：`hidden_dim 768` / `num_heads 12`、非対称深さ rot=10・pose=6、約 172M params。単発では 200ep に到達できず `version_3/last.ckpt` から resume して収束（`run.resume` 付き; repro.sh 参照）。

### メトリクスの解釈
位置 `0.368m` / 回転 `12.27°`。

### アーキテクチャ⇄メトリクスの因果考察
[[run-i525-asym]]（rot=10, 103M, `19.94°/0.700m`）に対し幅を広げただけで改善—「深化が負なら幅を試す」という #535 示唆を部分支持。だが回転 `12.27°` は EX10 (`9.98°`)・deep16 (`8.40°`) に劣り、位置 `0.368m` も及ばない。172M + resume を要して 78M の deep16 に負ける＝幅は容量 / 学習コストの割にリターンが小さい。

### 既存実験との比較
深さ振りの [[run-i540-asym-deep16]] に明確に劣る（`compares`）。[[run-i518-exp10]] とも比較（`compares`）。

### 次に有効な実験
非対称容量配分は「幅ではなく深さ」に集約する（deep16 路線）。172M 級の幅広モデルはコスト効率が悪く、本タスク規模では非推奨。
