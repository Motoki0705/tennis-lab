---
id: run-plcs-multiview-all-outputs-beta01
type: run
title: PLCS全出力等重みベースライン（position beta 0.1）
provider: codex
session: 01a03ee0-c172-7d22-81d3-c07127757135
date: '2026-08-27'
status: done
config:
  model: multiview_canonical
  loss: all_outputs_beta01
metrics:
  position_error_m: 6.605493
  position_error_std_m: 2.956382
  position_error_median_m: 6.226156
  angular_error_deg: 91.731071
  angular_error_std_deg: 52.893692
  angular_error_median_deg: 91.482315
  x_error_m: 2.517299
  y_error_m: 5.702384
  z_error_m: 0.093369
  position_accuracy: 0.01
  angle_accuracy: 0.1
  position_accuracy_0.5m: 0.01
  position_accuracy_1m: 0.015625
  position_accuracy_2m: 0.04
  angle_accuracy_10deg: 0.070625
  angle_accuracy_15deg: 0.1
  angle_accuracy_30deg: 0.161875
  loss: 2.383725
  loss_canonical_pose: 0.009055
  loss_reprojection: 0.105332
repro:
  commit: 101120f100474781121ca0e83d65b5da9a2d7a20
  branch: exp/plcs-reprojection-loss
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    TENNIS_RUN_ID=1787796355222865602_direct_plcs-multiview-all-outputs-beta01-gpu1
    TENNIS_REPRO_DIR=/home/kamimura/projects/tennis-lab/.training_queue/repro/1787796355222865602_direct_plcs-multiview-all-outputs-beta01-gpu1
    .venv/bin/python -m src.tasks.plcs.scripts.train model=multiview_canonical loss=all_outputs_beta01
    data.scene_dir=plcs_canonical_pose_beta data.batch_size=4 'data.num_views_range=[2,2]'
    'data.seq_len_range=[16,16]' data.num_workers=2 training.trainer.accumulate_grad_batches=1
    training.trainer.precision=16-mixed training.trainer.max_epochs=50 training.compile.enabled=false
    training.qualitative_logging.enabled=false training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false run.output_dir=plcs/plcs_multiview_all_outputs_beta01_gpu1
    paths.artifact_root=/home/kamimura/projects/tennis-lab/.training_queue/repro/1787796355222865602_direct_plcs-multiview-all-outputs-beta01-gpu1
artifacts:
  run_dir: knowledge/runs/run-plcs-multiview-all-outputs-beta01
  predictions: knowledge/runs/run-plcs-multiview-all-outputs-beta01/pred_test.npz
  output_dir: outputs/plcs/plcs_multiview_all_outputs_beta01_gpu1/logs/version_0
  curves: knowledge/runs/run-plcs-multiview-all-outputs-beta01/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_all_outputs_beta01_gpu1/logs/version_0
parents: []
relations: []
tags:
- plcs
- multiview
- canonical-pose
- loss-ablation
- smooth-l1-beta01
---

## 考察 / Findings

### 要約

position・rotation・wrapped angle・canonical poseをすべてweight 1で学習し、position Smooth-L1 betaを0.1へ固定した対照run。seed 42、2 camera、16 frame、effective batch 4で42 epoch（global step 8400）まで進み、EarlyStoppingで終了した。test position errorは6.605493 m、angular errorは91.731071度だった。

### アーキテクチャ詳細

`PLCSMultiViewModel`（39.4M parameters、hidden 512、12層、8 heads）のcamera-time interleaved M-RoPE経路を用いた。入力はCOCO17 poseとCourtKP20、出力はcourt座標のposition 3次元、rotationの`(cos, sin)` 2次元、canonical pose 17×3。`data/plcs_canonical_pose_beta`の固定split（train/val/test = 800/100/100 scenes）、physical batch 4、FP16、最大50 epochで学習した。GPU #1（GTX 1650）上で完走し、全4教師のweightは1、reprojection weightだけ0である。

### メトリクスの解釈

testのposition mean/medianは6.605493/6.226156 m、angular mean/medianは91.731071/91.482315度で、30度以内の向き正解率は0.161875だった。reprojection項はweighted totalへ加えていないが、同じclean 2D targetに対するraw評価値として0.105332を記録した。canonical poseのraw lossは0.009055である。

### アーキテクチャ⇄メトリクスの因果考察

4教師を等重みにしただけでは、shared interleaved trunk上でposition・rotation・canonical poseの勾配競合を解消できていない。特に約90度のangular errorと低いangle accuracyは、2D整合を直接課さない3D教師だけでは向きとposeの組合せを画像面で拘束し切れていない可能性と整合する。一方、これは単一seedの観測であり、原因の確定ではない。

### 既存実験との比較

本runは[[group-plcs-reprojection-loss-w1]]の対照条件であり、[[run-plcs-multiview-all-outputs-beta01-reprojection-w1]]との差はreprojection weight 1の有無だけである。モデル、seed、split、view/frame数、有効batch、学習step数は一致している。

### 次に有効な実験

対になるreprojection runの結果を踏まえ、weight 0.1/0.3/1/3のsweepを3 seedsで行う。position・angular・canonical pose・raw reprojectionを同時に判定し、2D整合だけを改善して3D depthを悪化させる条件を除外する。
