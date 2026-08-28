---
id: run-plcs-multiview-axial-all-outputs-beta01-v4-t128
type: run
title: PLCS axial全出力等重みベースライン（V=4, T=128）
provider: codex
session: 01a03ee0-c172-7d22-81d3-c07127757135
date: '2026-08-28'
status: done
config:
  model: multiview_axial_base
  loss: all_outputs_beta01
  data: plcs_canonical_pose_beta
  num_views: 4
  seq_len: 128
  batch_size: 4
  precision: bf16-mixed
  max_epochs: 50
  seed: 42
  position_weight: 1.0
  position_smooth_l1_beta: 0.1
  rotation_weight: 1.0
  angle_weight: 1.0
  canonical_pose_weight: 1.0
  reprojection_weight: 0.0
metrics:
  position_error_m: 1.386235
  position_error_std_m: 0.804937
  position_error_median_m: 1.311628
  angular_error_deg: 66.968224
  angular_error_std_deg: 51.365856
  angular_error_median_deg: 50.81916
  x_error_m: 0.696738
  y_error_m: 1.025382
  z_error_m: 0.096257
  position_accuracy: 0.118984
  angle_accuracy: 0.141563
  position_accuracy_0.5m: 0.118984
  position_accuracy_1m: 0.355625
  position_accuracy_2m: 0.811562
  angle_accuracy_10deg: 0.093594
  angle_accuracy_15deg: 0.141563
  angle_accuracy_30deg: 0.303438
  loss: 1.477733
  loss_canonical_pose: 0.008194
  loss_reprojection: 0.025546
repro:
  commit: 00a2bb2af251a86038600c871b1f30a3c664b9b8
  branch: exp/plcs-reprojection-loss
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    .venv/bin/python -m src.tasks.plcs.scripts.train model=multiview_axial_base loss=all_outputs_beta01
    data.scene_dir=plcs_canonical_pose_beta data.batch_size=4 "data.num_views_range=[4,4]"
    "data.seq_len_range=[128,128]" data.num_workers=2 training.trainer.accumulate_grad_batches=1
    training.trainer.precision=bf16-mixed training.trainer.max_epochs=50 training.compile.enabled=false
    training.qualitative_logging.enabled=false training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false run.output_dir=plcs/plcs_multiview_axial_all_outputs_beta01_v4_t128_bf16_gpu0
    paths.project_root=/home/kamimura/projects/tennis-lab paths.artifact_root=$TENNIS_REPRO_DIR
artifacts:
  run_dir: knowledge/runs/run-plcs-multiview-axial-all-outputs-beta01-v4-t128
  predictions: knowledge/runs/run-plcs-multiview-axial-all-outputs-beta01-v4-t128/pred_test.npz
  log: .training_queue/logs/1787847972019077668_797896_plcs-multiview-axial-all-outputs-beta01-v4-t128-bf16-gpu0.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/plcs_multiview_axial_all_outputs_beta01_v4_t128_bf16_gpu0/logs/version_0
  curves: knowledge/runs/run-plcs-multiview-axial-all-outputs-beta01-v4-t128/curves.png
  visualization: knowledge/runs/run-plcs-multiview-axial-all-outputs-beta01-v4-t128/visualization_scene_000233_position_rotation.mp4
  tb_logdir: outputs/plcs/plcs_multiview_axial_all_outputs_beta01_v4_t128_bf16_gpu0/logs/version_0
parents: []
relations:
- {to: run-plcs-multiview-all-outputs-beta01, rel: compares}
tags:
- plcs
- multiview
- axial
- canonical-pose
- loss-ablation
- smooth-l1-beta01
- v4
- t128
- bf16
---

## 考察 / Findings

### 要約

`PLCSMultiViewAxialModel`でposition・rotation・wrapped angle・canonical poseをすべてweight 1、position Smooth-L1 betaを0.1として学習したV=4・T=128の対照run。GPU #0とBF16を用いて50 epochを完走し、test position errorは1.386235 m、angular errorは66.968224度だった。

### アーキテクチャ詳細

`multiview_axial_base`はhidden 512、8 heads、8 stageで、各stage内にcamera軸attentionとtime軸attentionを1層ずつ持つ。COCO17 poseとcourt keypointsをcamera・frameごとのgroup tokenへ埋め込み、camera/time attentionを交互に適用した後、共有readoutからposition 3次元、rotation `(cos, sin)` 2次元、canonical pose 17×3を推定する。固定`plcs_canonical_pose_beta` split、4 views、128 frames、physical batch 4、seed 42、BF16、50 epochで実行し、reprojectionだけはweighted totalへ加えていない。

### メトリクスの解釈

test position mean/medianは1.386235/1.311628 m、angular mean/medianは66.968224/50.819160度だった。positionは0.5/1/2 m以内が11.90%/35.56%/81.16%、angleは15/30度以内が14.16%/30.34%である。raw reprojectionは0.025546、canonical pose lossは0.008194で、前者はweight 0でも比較用に同じclean 2D targetから計測している。収束曲線ではvalidation positionが3.689 mから1.254 mまで低下し、step 8399で最良1.238 mだった。一方validation angular errorはstep 1799の69.896度が最良で、最終75.509度まで戻っており、後半はposition改善とrotation汎化が一致していない。

### アーキテクチャ⇄メトリクスの因果考察

4 camera・128 frameをcamera/time別のattentionで集約するため、単一frame/viewでは曖昧なコート上の位置を複数視点と長い時間文脈で拘束できた可能性が高い。ただしこれは旧runからarchitecture・V/T・precision・epoch数を同時に変更した観測であり、改善量をaxial attentionだけの効果とは断定できない。後半にvalidation angular errorが悪化したことから、共有readoutに対する等重み4教師だけではpositionとrotationの最適化時点が揃わない、という仮説が残る。

### 既存実験との比較

旧`PLCSMultiViewModel`の[[run-plcs-multiview-all-outputs-beta01]]（V=2、T=16）に対し、position errorは6.605493から1.386235 m、angular errorは91.731071から66.968224度、raw reprojectionは0.105332から0.025546へ改善した。ただし旧runはinterleaved model・FP16・42 epochであり、本runはaxial model・BF16・50 epochかつ入力contextも大きいため、これはprotocol全体の比較であってarchitecture単独のablationではない。同一protocolでreprojection weightだけを変えた直接の対照は[[run-plcs-multiview-axial-all-outputs-beta01-reprojection-w1-v4-t128]]である。

### 次に有効な実験

対になるreprojection runの効果を複数seedで再現確認し、weight 0.1/0.3/1.0を比較する。併せてpositionとrotationで別々のbest epochを確認し、共有readoutの勾配競合なのかcheckpoint選択の問題なのかを切り分ける。
