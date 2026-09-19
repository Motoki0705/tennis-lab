---
id: run-slcs-plcs-meiji-foot-e60-resume-v3
type: run
title: Meiji PLCS 60epoch完了・validation最良選定
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: multiview_axial_foot_residual
  loss: all_outputs_beta01_reprojection
  data: camera_view_real_rgb_ft_v1
  data.num_workers: 0
metrics:
  position_error_m: 0.25076
  angular_error_deg: 5.34375
  position_accuracy_0.5m: 0.933561
  angle_accuracy_15deg: 0.971509
  canonical_mpjpe_m: 0.142817
  canonical_pck_0.1m: 0.415702
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_meiji_foot_real_rgb
    data.num_workers=0 run.output_dir=plcs/train/meiji_foot_real_rgb/s42-001 run.init_weights=null
    run.resume=plcs/train/meiji_foot_real_rgb/s42-001/logs/version_2/checkpoints/last.ckpt
artifacts:
  run_dir: knowledge/runs/run-slcs-plcs-meiji-foot-e60-resume-v3
  predictions: knowledge/runs/run-slcs-plcs-meiji-foot-e60-resume-v3/pred_test.npz
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/train/meiji_foot_real_rgb/s42-001/logs/version_3
  tb_logdir: outputs/plcs/train/meiji_foot_real_rgb/s42-001/logs/version_3
  checkpoint_selection: knowledge/runs/run-slcs-plcs-meiji-foot-e60-resume-v3/selection.json
  curves: knowledge/runs/run-slcs-plcs-meiji-foot-e60-resume-v3/curves.png
  all_versions_curves: knowledge/runs/run-slcs-plcs-meiji-foot-e60-resume-v3/all_versions_curves.png
parents:
- run-slcs-plcs-meiji-foot-e60-resume-v2
relations: []
tags:
- slcs
- plcs
- meiji
- foot-residual
---

## 考察 / Findings

### 要約
3回の中断を経て総60epochを完了。version 0–3の保存checkpointをvalidation位置誤差のみで選び、epoch index 57（58番目）の0.218174353mを採用。testは選定に使用していない。

### アーキテクチャ詳細
78.4Mパラメータのfoot-residual PLCS。3視点、32–128 frame、batch4、lr5e-5、bf16。前区間のoptimizer/scheduler状態を復元し、num_workers=0へ変更してepoch47–59を継続。全再開区間と全保存候補のSHA256はselection.jsonを参照。

### メトリクスの解釈
frontmatterとpred_test.npzは終端epoch59の自動test値。位置誤差0.250760436m、回転5.34375deg、canonical MPJPE0.142817244m。これらはvalidation選定epoch57のtestではない。合成source-motion分離教師との一致であり、実RGBの独立実測3D精度を示さない。
validationはversion0の最良0.240329489mからversion1=0.223767430m、version2=0.220993042m、version3=0.218174353mへ低下した。後半の改善は小さく、epochごとの変動も残る。

### アーキテクチャ⇄メトリクスの因果考察
継続学習でvalidation位置誤差は低下した。ただし同条件の対照runが無いため、foot residualやnum_workers変更の個別寄与は断定できない。v0/v2はWSL再起動、v1はnative exit139で中断。num_workers=0で完走した事実だけから障害原因は確定しない。

### 既存実験との比較
前提runは同一60epoch実験の再開区間。中断区間はfailedのまま保持。last.ckptのcallback current_scoreは順序により前epochを指す場合があるため、選定では同epochのbest_k_modelsおよびTensorBoardを照合。中断直前のTB未flush epoch46はcallbackのepoch付き候補値を使用した。

### 次に有効な実験
選定重みのCPU testと終端testの全非予測array一致をrun-slcs-plcs-meiji-foot-e60-selected-testで確認済み。次に実RGBへの教師適用を評価する。独立測定3Dが無いため、実RGBの誤差は外部GT導入後に別評価する。
