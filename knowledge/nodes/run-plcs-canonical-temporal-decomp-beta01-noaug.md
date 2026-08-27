---
id: run-plcs-canonical-temporal-decomp-beta01-noaug
type: run
title: PLCS canonical pose temporal decomposition（β=0.1 / augmentationなし）
issue: 530
provider: codex
session: 01a03ee0-c172-7d22-81d3-c07127757135
date: '2026-08-27'
status: done
config:
  model: multiview_axial_base_temporal_pose
  loss: canonical_only
  data: multiview_sequence
  seed: 42
  epochs: 50
  sequence_length: 125
  num_views: 4
  augmentation: false
  canonical_pose_smooth_l1_beta: 0.1
  qualitative_logging: false
metrics:
  position_error_m: 10.860352
  position_error_std_m: 3.448579
  position_error_median_m: 10.601802
  angular_error_deg: 88.823158
  angular_error_std_deg: 51.081795
  angular_error_median_deg: 82.939171
  x_error_m: 5.004099
  y_error_m: 5.787912
  z_error_m: 6.111476
  position_accuracy: 0.0
  angle_accuracy: 0.05624
  position_accuracy_0.5m: 0.0
  position_accuracy_1m: 0.0
  position_accuracy_2m: 0.00112
  angle_accuracy_10deg: 0.03656
  angle_accuracy_15deg: 0.05624
  angle_accuracy_30deg: 0.13664
  canonical_mpjpe_m: 0.091136
  canonical_pred_temporal_deviation_m: 0.096652
  canonical_target_temporal_deviation_m: 0.08226
  canonical_motion_amplitude_ratio: 1.174967
  canonical_scene_motion_ratio_median: 1.118058
  canonical_centered_pose_pearson: 0.795146
  canonical_scene_centered_pose_cosine_median: 0.849395
  canonical_raw_velocity_ratio: 1.905053
  canonical_lowpass_velocity_correlation: 0.667964
  canonical_pred_high_frequency_fraction: 0.391067
  canonical_target_high_frequency_fraction: 0.06893
repro:
  commit: e57e1d9f868e289135dc7881fba5ffc0ee6a0282
  branch: exp/plcs-canonical-pose-beta
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.plcs.scripts.train
    model=multiview_axial_base_temporal_pose data=multiview_sequence loss=canonical_only
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.external_asset_root=/home/kamimura/projects/tennis-lab/data
    data.scene_dir=plcs_canonical_pose_beta data.seq_len_range=[125,125] data.num_views_range=[4,4]
    data.augmentation.enabled=false run.seed=42 run.gpus=1 run.test_after_fit=true
    training.trainer.max_epochs=50 training.trainer.check_val_every_n_epoch=1 training.early_stopping.enabled=false
    training.checkpoint.monitor=val/loss_canonical_pose training.checkpoint.mode=min
    training.checkpoint.save_top_k=1 training.qualitative_logging.enabled=false run.output_dir=plcs/plcs_multiview_axial_canonical_only_temporal_decomp_beta01_noaug
artifacts:
  run_dir: knowledge/runs/run-plcs-canonical-temporal-decomp-beta01-noaug
  predictions: knowledge/runs/run-plcs-canonical-temporal-decomp-beta01-noaug/pred_test.npz
  output_dir: outputs/plcs/plcs_multiview_axial_canonical_only_temporal_decomp_beta01_noaug/logs/version_0
  curves: knowledge/runs/run-plcs-canonical-temporal-decomp-beta01-noaug/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_canonical_only_temporal_decomp_beta01_noaug/logs/version_0
parents:
- run-i530-direct-baseline
relations: []
tags:
- plcs
- canonical
- motion
- temporal-decomposition
- mean-collapse
- no-augmentation
---

## 考察 / Findings

### 要約

canonical pose だけを教師とする条件で、平均poseへ固定されるdirect readoutを、sequence-static poseとzero-mean motion residualに明示分解した。100 test sceneで予測の時間偏差は `0.096652 m`、GTは `0.082260 m`（振幅比 `1.174967`）、時間平均との差を除いたposeのPearson相関は `0.795146`、scene別cosine中央値は `0.849395` だった。したがって、50 epochで「時間によらない平均pose」から脱し、入力に対応したpose motionを推論する完了条件を満たした。

### アーキテクチャ詳細

`multiview_axial_base_temporal_pose` は既存のshared axial trunkを維持し、canonical readoutだけを `TemporalDecomposedCanonicalPoseHead` に置き換える。valid frameのfeature平均からstatic poseを予測し、各frameの平均との差をRMS正規化した別headからmotion residualを予測する。motion residualはvalid frame上で再度zero-mean化し、static poseと加算する。paddingされた可変長sequenceでも同じ制約になるよう、base / split / camtokenのreadoutへvalid frame maskを伝播した。

学習条件は `canonical_only`、Smooth-L1のメートル単位beta `0.1`、seed `42`、4 view、125 frame固定、augmentation無効、early stopping無効、50 epochである。`training.qualitative_logging.enabled=false` とし、学習中のanimation保存は行っていない。position / rotationのloss weightは0なので、frontmatterの位置・回転メトリクスは未学習headの値であり、このrunの成否判定には使用しない。

### メトリクスの解釈

canonical MPJPEは `0.091136 m`。平均pose固定ならほぼ0になるはずの予測時間偏差は `0.096652 m` で、GT `0.082260 m` の `117.5%`、scene別振幅比の中央値も `1.118058` だった。さらに、単なる無関係な揺れではなく、centered poseのPearson相関 `0.795146`、scene別centered cosine中央値 `0.849395`、9-frame low-pass後の速度相関 `0.668` を得たため、GTの動きの方向と時系列を追従していると解釈できる。

一方、raw速度比は `1.905053`、edge paddingした9-frame移動平均に対する `1 - low-pass速度 / raw速度` で定義したhigh-frequency fractionは、予測 `0.391067` に対してGT `0.068930` である。動きを復元した代償として高周波jitterを過剰生成しており、「動くこと」は達成したがtemporal smoothnessは未解決である。

### アーキテクチャ⇄メトリクスの因果考察

診断時の観測では、clean入力のraw UV frame差が約 `2.46e-4` なのに対し、既定augmentationはGaussian noise `0.003` とtemporal jitter `0.002` を加えていた。信号より1桁大きい摂動を切ると、direct readoutの50-epoch実験の先頭20 test sceneでも予測/GT速度比は `3.65%` から `12.79%` へ改善した。これはaugmentationが微細なmotion cueを壊す一因であるという観測を支持するが、それだけでは平均pose退化を解消しなかった。

また、trunk hookではembeddingの時間偏差/RMSが `4.44%` だったのに対し、axial block後の共通残差成分が大きく（state RMS `16.95`）、final RMSNorm後の時間偏差/RMSは `1.33%`、direct canonical出力では `1.06%` まで圧縮された。ここから、shared residual streamのsequence共通成分に小さな時間差が埋もれることを第二の原因と仮定した。temporal decompositionは、その小さな差分をstatic成分から分離してRMS正規化し、motion headの入力スケールを保証する。この機構と最終runの振幅比・centered相関の同時改善は因果仮説と整合する。ただし本run単独では、正規化・zero-mean制約・augmentation無効化それぞれの寄与を完全には分離していない。

### 既存実験との比較

親の [[run-i530-direct-baseline]] はcanonical poseをframeごとに直接回帰するreadoutであり、本runが置き換えた構造上のbaselineである。ただし親はEX10 split trunk、`canonical_rot`、100 epochのmulti-task条件なので、canonical MPJPEの絶対値を直接優劣比較してはいけない。本研究内の同一dataset・50 epoch・augmentation無効条件では、direct readoutの先頭20 sceneのraw速度比が `0.1279` に留まったのに対し、temporal decompositionは同じ20 sceneで `2.0930`（全100 sceneでは `1.9051`）まで動きを復元し、全100 sceneのcentered pose相関 `0.7951` を得た。主結論は静的精度の単純改善ではなく、平均pose固定から入力依存motionへ状態が変わったことである。

### 次に有効な実験

motion branchへvalid-mask対応の低域通過層を入れるか、canonical velocity lossを弱く追加し、振幅比を保ったままhigh-frequency fractionをGTの `0.069` に近づける。次に、Gaussian noise / temporal jitterをclean frame差より小さい範囲から段階的に戻し、motion相関を維持できるaugmentation上限を測る。最後にcanonical以外のlossを再導入し、position / rotationとのmulti-task条件でも退化が再発しないかを確認する。
