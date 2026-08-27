---
id: run-plcs-multiview-all-outputs-beta01-reprojection-w1
type: run
title: PLCS全出力等重み＋reprojection weight 1
provider: codex
session: 01a03ee0-c172-7d22-81d3-c07127757135
date: '2026-08-27'
status: done
config:
  model: multiview_canonical
  loss: all_outputs_beta01_reprojection
metrics:
  position_error_m: 6.603486
  position_error_std_m: 2.961226
  position_error_median_m: 6.215611
  angular_error_deg: 88.649841
  angular_error_std_deg: 52.761093
  angular_error_median_deg: 89.419098
  x_error_m: 2.522979
  y_error_m: 5.691253
  z_error_m: 0.275332
  position_accuracy: 0.01
  angle_accuracy: 0.10375
  position_accuracy_0.5m: 0.01
  position_accuracy_1m: 0.015625
  position_accuracy_2m: 0.04
  angle_accuracy_10deg: 0.071875
  angle_accuracy_15deg: 0.10375
  angle_accuracy_30deg: 0.18
  loss: 2.390718
  loss_canonical_pose: 0.00915
  loss_reprojection: 0.087569
repro:
  commit: 101120f100474781121ca0e83d65b5da9a2d7a20
  branch: exp/plcs-reprojection-loss
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    TENNIS_RUN_ID=1787816665730150531_direct_plcs-multiview-all-outputs-beta01-reprojection-w1-gpu1
    TENNIS_REPRO_DIR=/home/kamimura/projects/tennis-lab/.training_queue/repro/1787816665730150531_direct_plcs-multiview-all-outputs-beta01-reprojection-w1-gpu1
    .venv/bin/python -m src.tasks.plcs.scripts.train model=multiview_canonical loss=all_outputs_beta01_reprojection
    data.scene_dir=plcs_canonical_pose_beta data.batch_size=4 'data.num_views_range=[2,2]'
    'data.seq_len_range=[16,16]' data.num_workers=2 training.trainer.accumulate_grad_batches=1
    training.trainer.precision=16-mixed training.trainer.max_epochs=50 training.compile.enabled=false
    training.qualitative_logging.enabled=false training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false run.output_dir=plcs/plcs_multiview_all_outputs_beta01_reprojection_w1_gpu1
    paths.artifact_root=/home/kamimura/projects/tennis-lab/.training_queue/repro/1787816665730150531_direct_plcs-multiview-all-outputs-beta01-reprojection-w1-gpu1
artifacts:
  run_dir: knowledge/runs/run-plcs-multiview-all-outputs-beta01-reprojection-w1
  predictions: knowledge/runs/run-plcs-multiview-all-outputs-beta01-reprojection-w1/pred_test.npz
  output_dir: outputs/plcs/plcs_multiview_all_outputs_beta01_reprojection_w1_gpu1/logs/version_merged_gpu1_to_gpu0_v2
  resume_run: knowledge/runs/run-plcs-multiview-all-outputs-beta01-reprojection-w1/resume_gpu0_run.json
  resume_repro: knowledge/runs/run-plcs-multiview-all-outputs-beta01-reprojection-w1/resume_gpu0_repro.sh
  curves: knowledge/runs/run-plcs-multiview-all-outputs-beta01-reprojection-w1/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_all_outputs_beta01_reprojection_w1_gpu1/logs/version_merged_gpu1_to_gpu0_v2
parents:
- run-plcs-multiview-all-outputs-beta01
relations: []
tags:
- plcs
- multiview
- canonical-pose
- reprojection-loss
- loss-ablation
- smooth-l1-beta01
---

## 考察 / Findings

### 要約

対照runの全4教師weight 1とposition beta 0.1を維持し、予測position・rotation・canonical poseを全cameraへ投影するreprojection lossをweight 1で追加した。42 epoch（global step 8400）でEarlyStopping終了し、test raw reprojectionは0.087569、position errorは6.603486 m、angular errorは88.649841度だった。

### アーキテクチャ詳細

モデル・データ・seed・batch・step budgetは[[run-plcs-multiview-all-outputs-beta01]]と同一。予測canonical poseをrotationでworld向きへ回し、positionを加えてworld poseを構成し、選択された全cameraのintrinsics/extrinsicsでnormalized UVへ投影する。clean 2D poseとのmasked Smooth-L1（beta 0.01）をvisibilityとpaddingでmaskし、weight 1でtotalへ加えた。epoch 0–11はGPU #1、`last.ckpt`（epoch 11/global step 2400）からepoch 12–41をGPU #0へoptimizer・scheduler・EarlyStopping state込みでresumeした。実行segmentは`artifacts.resume_run`/`resume_repro`に保存し、曲線はcheckpoint境界で重複stepを除いた統合eventから生成する。

### メトリクスの解釈

test raw reprojectionは0.105332から0.087569へ0.017764（16.86%）低下した。angular errorは3.081230度（3.36%）改善し、30度以内accuracyは0.161875から0.180000へ1.8125 percentage points上昇した。一方、position errorは0.002007 m（0.03%）の改善に留まり、canonical pose lossは0.009055から0.009150へ1.05%悪化した。軸別ではYが0.011131 m改善した一方、Zは0.093369から0.275332 mへ悪化した。

### アーキテクチャ⇄メトリクスの因果考察

reprojection項が直接最適化する画像面整合とangular errorが同時に改善したため、position・rotation・poseを組み合わせた2D拘束が向き推定へ有効だったという解釈と整合する。ただし2D reprojectionは視線方向depthを一意に拘束しないため、Z悪化とposition全体の横ばいも幾何学的に説明可能である。GPU architectureを途中で切り替えたためbitwise同一trajectoryではなく、単一seedでもあることから、改善の再現性は追加seedで確認が必要である。

### 既存実験との比較

[[run-plcs-multiview-all-outputs-beta01]]に対し、同じ42 epoch/global step 8400でreprojection weightだけを0から1へ変更した比較である。主要差分はraw reprojection -16.86%、angular error -3.36%、position error -0.03%、Z error +0.181963 m。したがってweight 1は2D整合と向きには有望だが、3D position改善策として単独採用する根拠は弱い。

### 次に有効な実験

同じprotocolを3 seedsで再実行し、weight 0.1/0.3/1/3を比較する。採用条件はraw reprojectionとangular errorの改善に加え、position errorおよび特にZ errorをbaselineから悪化させないこととする。必要ならdepth/ground-contact拘束を別項として追加し、reprojectionの視線方向不定性を切り分ける。
