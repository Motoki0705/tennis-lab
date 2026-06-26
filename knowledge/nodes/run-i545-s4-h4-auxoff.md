---
id: run-i545-s4-h4-auxoff
type: run
title: i545_s4_h4_auxoff
issue: 545
provider: codex
session: 019ef7e7-03a8-7e53-9c92-21cdcb326a79
date: '2026-06-25'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.303682
  position_error_std_m: 0.230623
  position_error_median_m: 0.245923
  angular_error_deg: 8.15782
  angular_error_std_deg: 7.544045
  angular_error_median_deg: 6.124016
  x_error_m: 0.11752
  y_error_m: 0.249097
  z_error_m: 0.045938
  position_accuracy: 0.84954
  angle_accuracy: 0.850352
  position_accuracy_0.5m: 0.84954
  position_accuracy_1m: 0.976774
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.712419
  angle_accuracy_15deg: 0.850352
  angle_accuracy_30deg: 0.977722
repro:
  commit: 02024548fa0bb35e732b5d1fef92d77281a20b9a
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=4
    model.num_task_layers=4 data=chunked_multiview_sequence_bs8 data.batch_size=8
    training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256] loss=canonical_rot
    loss.canonical_pose_weight=0.0 loss.joint_angle_weight=0.0 loss.torsion_angle_weight=0.0
    loss.torso_twist_weight=0.0 loss.bone_length_weight=0.0 training.trainer.max_epochs=200
    training.early_stopping.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i545-s4-h4-auxoff
  predictions: knowledge/runs/run-i545-s4-h4-auxoff/pred_test.npz
  log: .training_queue/logs/1782305635208315185_634509_i545_s4_h4_auxoff.log
  curves: knowledge/runs/run-i545-s4-h4-auxoff/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_8
parents:
- run-i545-s4-h4
relations:
- to: run-i545-s4-h4
  rel: compares
- to: run-i545-s4-h4-posw4
  rel: compares
- to: run-i545-s4-h4-auxoff-posw8
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- chunked
- data-rich
- trunk-allocation
- loss-tuning
- aux-off
---

## 考察 / Findings

### 要約
S=4/H=4 に pose naturalness 系 aux を一括 OFF（`canonical_pose` / `joint_angle` / `torsion_angle` / `torso_twist` / `bone_length` weight = 0）。回転 8.16°/位置 0.304m で、**回転は baseline [[run-i545-s4-h4]]（8.23°）と同等、位置は改善（0.337→0.304）**。pose 自然さ補助は test の回転・位置精度に寄与しておらず、外しても劣化しない（むしろ位置微改善）。

### アーキテクチャ詳細
[[run-i545-s4-h4]] と同一容量/データ/プロトコル。差は loss の補助 5 項を全て 0 にした点: `canonical_pose_weight=joint_angle_weight=torsion_angle_weight=torso_twist_weight=bone_length_weight=0`。`position_weight` / `rotation_weight` / `angle_weight` は既定のまま。

### メトリクスの解釈
回転 mean 8.16 / median 6.12（baseline 8.23/5.94 と同等）、位置 mean 0.304 / median 0.246（baseline 0.337/0.265 から改善）。角@15=0.850（baseline 0.871 から微減）、位置@0.5m=0.850（baseline 0.828 から改善）。収束は崩壊なし。

### アーキテクチャ⇄メトリクスの因果考察
naturalness aux は pose の解剖学的妥当性を正則化する補助教師で、回転（剛体姿勢）/位置（ルート3D）の test 指標には直接効かない。これらを外すと共有/分岐 trunk の容量が主要タスクへ振り向けられ位置が僅かに改善した（仮説）。回転が変わらないのは canonical_pose を外しても rotation supervision 自体は残るため。

### 既存実験との比較
- baseline [[run-i545-s4-h4]] と回転同等・位置良化。
- 位置改善は posw 増量 [[run-i545-s4-h4-posw4]]（0.258m）の方が大きい＝aux-off と posw4 は独立な位置改善レバーで併用余地あり。
- aux-off に posw8 を足した [[run-i545-s4-h4-auxoff-posw8]] は逆に崩壊（aux 除去では posw8 の害を救えない）。

### 次に有効な実験
- aux-off + posw4 を併用し位置を更に詰められるか。
- pose plausibility 系の別評価で aux 除去の副作用（不自然な pose 増加）を確認し、test 精度との trade を把握。
