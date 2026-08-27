---
id: run-plcs-multiview-axial-all-outputs-beta01-reprojection-w1-v4-t128
type: run
title: PLCS axial全出力等重み＋reprojection weight 1（V=4, T=128）
provider: codex
session: 01a03ee0-c172-7d22-81d3-c07127757135
date: '2026-08-28'
status: done
config:
  model: multiview_axial_base
  loss: all_outputs_beta01_reprojection
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
  reprojection_weight: 1.0
  reprojection_smooth_l1_beta: 0.01
metrics:
  position_error_m: 1.352761
  position_error_std_m: 0.934536
  position_error_median_m: 1.187532
  angular_error_deg: 63.600704
  angular_error_std_deg: 50.296989
  angular_error_median_deg: 46.408375
  x_error_m: 0.754972
  y_error_m: 0.960795
  z_error_m: 0.093188
  position_accuracy: 0.088828
  angle_accuracy: 0.149609
  position_accuracy_0.5m: 0.088828
  position_accuracy_1m: 0.391094
  position_accuracy_2m: 0.852031
  angle_accuracy_10deg: 0.079219
  angle_accuracy_15deg: 0.149609
  angle_accuracy_30deg: 0.346641
  loss: 1.403884
  loss_canonical_pose: 0.008204
  loss_reprojection: 0.023028
repro:
  commit: 00a2bb2af251a86038600c871b1f30a3c664b9b8
  branch: exp/plcs-reprojection-loss
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    .venv/bin/python -m src.tasks.plcs.scripts.train model=multiview_axial_base loss=all_outputs_beta01_reprojection
    data.scene_dir=plcs_canonical_pose_beta data.batch_size=4 "data.num_views_range=[4,4]"
    "data.seq_len_range=[128,128]" data.num_workers=2 training.trainer.accumulate_grad_batches=1
    training.trainer.precision=bf16-mixed training.trainer.max_epochs=50 training.compile.enabled=false
    training.qualitative_logging.enabled=false training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false run.output_dir=plcs/plcs_multiview_axial_all_outputs_beta01_reprojection_w1_v4_t128_bf16_gpu0
    paths.project_root=/home/kamimura/projects/tennis-lab paths.artifact_root=$TENNIS_REPRO_DIR
artifacts:
  run_dir: knowledge/runs/run-plcs-multiview-axial-all-outputs-beta01-reprojection-w1-v4-t128
  predictions: knowledge/runs/run-plcs-multiview-axial-all-outputs-beta01-reprojection-w1-v4-t128/pred_test.npz
  log: .training_queue/logs/1787847972038871666_797920_plcs-multiview-axial-all-outputs-beta01-reprojection-w1-v4-t128-bf16-gpu0.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/plcs_multiview_axial_all_outputs_beta01_reprojection_w1_v4_t128_bf16_gpu0/logs/version_0
  curves: knowledge/runs/run-plcs-multiview-axial-all-outputs-beta01-reprojection-w1-v4-t128/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_all_outputs_beta01_reprojection_w1_v4_t128_bf16_gpu0/logs/version_0
parents:
- run-plcs-multiview-axial-all-outputs-beta01-v4-t128
relations:
- {to: run-plcs-multiview-all-outputs-beta01-reprojection-w1, rel: compares}
tags:
- plcs
- multiview
- axial
- canonical-pose
- reprojection-loss
- loss-ablation
- smooth-l1-beta01
- v4
- t128
- bf16
---

## 考察 / Findings

### 要約

V=4・T=128のaxial baselineと全条件を揃え、予測position・rotation・canonical poseを組み合わせるreprojection lossだけをweight 1で追加したrun。GPU #0とBF16で50 epochを完走し、baseline比でtest position errorを0.033474 m、angular errorを3.367519度、raw reprojectionを0.002518改善した。

### アーキテクチャ詳細

モデル、固定split、4 views、128 frames、batch 4、seed 42、BF16、50 epoch、および4つの3D教師weight 1は[[run-plcs-multiview-axial-all-outputs-beta01-v4-t128]]と同一である。差分は、予測canonical poseを予測rotationでworld向きへ回して予測positionを加え、選択された全cameraのintrinsics/extrinsicsでnormalized UVへ投影する項だけである。clean 2D poseとのSmooth-L1（beta 0.01）をvisibilityとpaddingでmaskし、weight 1でtotal lossへ加えた。

### メトリクスの解釈

baseline比でposition meanは1.386235から1.352761 m（-2.42%）、medianは1.311628から1.187532 m（-9.46%）、angular meanは66.968224から63.600704度（-5.03%）、medianは50.819160から46.408375度（-8.68%）へ改善した。raw reprojectionは0.025546から0.023028（-9.86%）、30度以内angle accuracyは30.34%から34.66%（+4.32 percentage points）へ改善し、canonical pose lossは0.008194から0.008204（+0.13%）でほぼ横ばいだった。一方、0.5 m以内position accuracyは11.90%から8.88%へ低下し、position標準偏差も0.804937から0.934536 mへ増えた。軸別にはY/Zが0.064587/0.003069 m改善したがXは0.058234 m悪化しており、近距離精度と外れ値を含む分布上のトレードオフがある。収束曲線ではvalidation positionの最良値が1.238から1.143 mへ改善し、validation angular errorも後半まで71.8–72.2度付近を維持した。

### アーキテクチャ⇄メトリクスの因果考察

唯一の実験差分であるmulti-view 2D整合項と、raw reprojection・position中央値・angular errorが同時に改善したため、3 headを画像面で結合する幾何拘束がaxial featureの学習に寄与したと解釈できる。4視点により単眼よりdepth不定性は減るため、Y/Zも悪化しなかった。一方でX誤差、0.5 m以内率、分散の悪化は、normalized UV誤差を一律に最適化するとcamera配置や投影感度の高いsampleが優先され、3Dで非常に近いsampleと一部の外れ値を犠牲にしたという仮説と整合する。単一seedなので、この分布変化の再現性は未確定である。

### 既存実験との比較

直接比較対象の[[run-plcs-multiview-axial-all-outputs-beta01-v4-t128]]との差はreprojection weight 0→1だけである。旧`PLCSMultiViewModel`の[[run-plcs-multiview-all-outputs-beta01-reprojection-w1]]（V=2、T=16）に対しても、position errorは6.603486から1.352761 m、angular errorは88.649841から63.600704度、raw reprojectionは0.087569から0.023028へ改善した。ただし旧runとの間にはarchitecture・V/T・precision・epoch数の差があるため、改善要因は分離できない。

### 次に有効な実験

seedを3つ以上へ増やしてmean/median、0.5/1/2 m accuracy、軸別誤差、上位tailを同時比較する。その上でreprojection weight 0.1/0.3/1.0をsweepし、中央値と角度の改善を保ちつつ0.5 m以内率とX/tailを悪化させない条件を選ぶ。camera/jointごとの寄与を記録する診断も、外れ値の原因切り分けに有効である。
