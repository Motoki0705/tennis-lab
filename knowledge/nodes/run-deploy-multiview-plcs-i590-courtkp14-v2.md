---
id: run-deploy-multiview-plcs-i590-courtkp14-v2
type: run
title: deploy_multiview_plcs_i590_courtkp14_v2
issue: 590
provider: codex
session: 019f5ddc-9e4a-7d50-85e6-0db5262b88b1
date: '2026-07-14'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.175284
  position_error_std_m: 0.15513
  position_error_median_m: 0.132863
  angular_error_deg: 6.443357
  angular_error_std_deg: 5.960453
  angular_error_median_deg: 4.797728
  x_error_m: 0.071406
  y_error_m: 0.137416
  z_error_m: 0.039195
  position_accuracy: 0.962584
  angle_accuracy: 0.913884
  position_accuracy_0.5m: 0.962584
  position_accuracy_1m: 0.995843
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.799327
  angle_accuracy_15deg: 0.913884
  angle_accuracy_30deg: 0.994127
repro:
  commit: cd4e72c79536be1f9fdced12d2070c7dd7e77b23
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=0
    model.num_task_layers=6 data=chunked_multiview_sequence_bs8 data.num_court_kp=14
    data.batch_size=8 training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256]
    loss=canonical_rot loss.position_weight=8.0 loss.canonical_pose_weight=0.0 loss.joint_angle_weight=0.0
    loss.torsion_angle_weight=0.0 loss.torso_twist_weight=0.0 loss.bone_length_weight=0.0
    training.trainer.max_epochs=200 training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-deploy-multiview-plcs-i590-courtkp14-v2
  predictions: knowledge/runs/run-deploy-multiview-plcs-i590-courtkp14-v2/pred_test.npz
  log: .training_queue/logs/1783986831894575841_1649543_deploy_multiview_plcs_i590_courtkp14_v2.log
  output_dir: outputs/plcs/plcs_multiview_axial_split/logs/version_26
  checkpoint: ckpt/plcs/run-multiview-plcs-i590-courtkp14-epoch197.ckpt
  curves: knowledge/runs/run-deploy-multiview-plcs-i590-courtkp14-v2/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_26
parents:
- run-i590-courtkp14
relations:
- to: run-i590-courtkp14
  rel: confirms
tags:
- plcs
- multiview
- deploy
- court-kp-14
- split-trunk
- chunked
- sim-to-real
---

## 考察 / Findings

### 要約
prune 済みだった [[run-i590-courtkp14]] の deploy checkpoint を現行 main で再学習した run。test は位置 **0.175m**、yaw **6.44°** で、元 run の 0.189m / 6.28°をほぼ再現した。val position 最良の epoch 197 を `ckpt/` に配備する。

### アーキテクチャ詳細
`multiview_axial_split` H=0/S=6、3--6 camera、court keypoint 14点。`canonical_rot` の position weight は 8.0、canonical pose / joint angle / torsion / torso twist / bone length の補助損失は 0。200 epochs、early stopping off は親 run と同じで、推論に影響しない qualitative logging のみ無効化した。

### メトリクスの解釈
test position mean/median は 0.175/0.133m、yaw mean/median は 6.44/4.80°。位置 0.5m 以内 96.3%、yaw 15°以内 91.4%で、3カメラ実入力へ配備できる水準を再確認した。checkpoint callback の `val/pos_error_m` 最良は epoch 197 の 0.2072m。

### アーキテクチャ⇄メトリクスの因果考察
fully separate trunk により position weight 8.0 が rotation branch を壊さず、14点 court token が実 detector 契約との mismatch を避けたことが、位置とyawの両立に寄与したと考える（仮説）。親 run と独立再学習で近い値を得たため、結果は単発のseed偶然ではない。

### 既存実験との比較
[[run-i590-courtkp14]] 比で位置は 0.189→0.175mと改善、yawは6.28→6.44°とほぼ同等。deploy判断を変更する差ではなく、親 run の recipe を確認する再現結果である。

### 次に有効な実験
実3カメラ `meiji_3cam` で player association 後の軌跡整合性を確認し、cameraごとの検出欠損がposition/yawへ与える影響を可視化する。
