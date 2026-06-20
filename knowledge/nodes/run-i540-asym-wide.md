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

両 trunk の**幅**を 768 まで広げた非対称構成(`multiview_axial_split_wide`: hidden_dim 768 / num_heads 12、非対称深さ rot=10・pose=6、約 172M params)の 200epoch 結果。**位置 0.368m / 回転 12.27°**。172M は単発では 200ep に到達できず、`version_3/last.ckpt` から resume して 200ep を収束させた(`run.resume` 付き; repro.sh 参照)。

- **幅は効くが高くつく**: `run-i525-asym`(rot=10, 103M, 19.94°/0.700m)に対し幅を広げただけで 12.27°/0.368m まで改善。#535 の「深化が負なら幅を試す」という示唆は部分的に支持された。
- **ただし EX10 にも deep16 にも届かない**: 回転 12.27° は EX10(9.98°)・deep16(8.40°)に劣り、位置 0.368m も EX10(0.238m)・deep16(0.207m)に及ばない。172M という最大級の容量を投じ、resume まで要して、78M の deep16 に負けている。
- **結論(幅 vs 深さ)**: 回転改善の効率は**幅 < 深さ**。同 issue の `run-i540-asym-deep16`(深さ振り)が幅振りの本構成を明確に上回る。幅の拡大はパラメータ/学習コスト(resume 必須)の割にリターンが小さい。

次の示唆: 非対称容量配分は「幅ではなく深さ」に集約する(deep16 路線)。172M 級の幅広モデルはコスト効率が悪く、本タスク規模では非推奨。
