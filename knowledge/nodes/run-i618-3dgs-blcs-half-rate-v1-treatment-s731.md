---
id: run-i618-3dgs-blcs-half-rate-v1-treatment-s731
type: run
title: 3DGS×BLCS 1/12 synthetic treatment seed 731
issue: 618
provider: codex
session: 019f984c-8bc1-7041-8e1d-362a5b11daa2
date: '2026-07-26'
status: done
config:
  model: conv_next_unet
  initialization: ckpt/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt
  initialization_sha256: cd7927ad27e53ddd6aa77df28eca3c5e674552461ccda083a41e99e629857892
  loss: focal_bce_gamma_2
  data: TrackNet games 1-8 with one C10 synthetic window every second batch, game
    9 validation
  batch_size: 6
  synthetic_per_batch: 1
  synthetic_batch_period: 2
  synthetic_windows: 2620
  sampled_windows: 31440
  synthetic_fraction: 0.08333333333333333
  steps_per_epoch: 655
  epochs: 8
  seed: 731
  learning_rate: 1.0e-05
metrics:
  best_validation_epoch: 4
  best_val_f1: 0.6913540959358215
  best_val_precision: 0.6567028760910034
  best_val_recall: 0.7298657894134521
  best_val_mean_distance_px: 2.2479147911071777
  best_val_loss: 0.0004066114779561758
  final_val_f1: 0.6562153697013855
  final_val_precision: 0.6212893724441528
  final_val_recall: 0.6953020095825195
  final_val_mean_distance_px: 2.312025547027588
  paired_control_best_val_f1: 0.6729001402854919
  paired_f1_delta: 0.0184539556503296
  qualification_passed: true
repro:
  commit: ac9e640903a6dfaecb65fc980f5dcf408bbcd589
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: .venv/bin/python -m src.tasks.ball_detection.scripts.train --config-name
    train_3dgs_blcs_v1 data.synthetic_batch_period=2 run.output_dir=outputs/ball_detection/3dgs_blcs_half_rate_v1/treatment/seed_731
artifacts:
  run_dir: knowledge/runs/run-i618-3dgs-blcs-half-rate-v1-treatment-s731
  log: .training_queue/logs/1785010253183180835_2991601_i618_3dgs_blcs_half_rate_v1_treatment_s731.log
  output_dir: outputs/ball_detection/3dgs_blcs_half_rate_v1/treatment/seed_731/logs/version_0
  checkpoint: outputs/ball_detection/3dgs_blcs_half_rate_v1/treatment/seed_731/logs/version_0/checkpoints/3dgs-blcs-epoch=04.ckpt
  checkpoint_sha256: 5aecfbcbe04eaea10e9fcefc13896621e3ae901945bdf2bbc871497a45383d28
  live_monitor: .codex-loop/C12_LIVE_MONITOR.json
  curves: knowledge/runs/run-i618-3dgs-blcs-half-rate-v1-treatment-s731/curves.png
  tb_logdir: outputs/ball_detection/3dgs_blcs_half_rate_v1/treatment/seed_731/logs/version_0
parents:
- run-i618-3dgs-blcs-half-rate-v1-control-s731
- run-i618-blcs-b00-full-scale-v1
relations:
- to: run-i618-3dgs-blcs-v1-treatment-s731
  rel: supersedes
- to: run-i618-3dgs-blcs-real-baseline-v1
  rel: compares
tags:
- ball_detection
- 3dgs-blcs
- paired-treatment
- simple-sphere
- synthetic-half-rate
- seed-731
- passed-qualification
- validation-only
---

## 考察 / Findings

### 要約

C10 simple-sphere syntheticの平均露出を1/12へ半減したseed 731 treatmentは、
game-9 best validation F1 **0.691354**で、paired real-only controlの
**0.672900**を`+0.018454`上回った。事前宣言したqualification gateを通過し、
同一mixをpredeclared seeds 1931/3253へ展開する。

### アーキテクチャ詳細

controlと同じConvNeXt U-Net初期値、AdamW、batch 6、655 step/epoch、
8 epoch、lr `1e-5`、augmentation、seedを使用した。唯一の機能差は
`synthetic_batch_period=2`でmixed batchとreal-only batchを交互にし、
mixed batchだけ1つのC10 synthetic windowを含めた点である。全8 epochでは
31,440 sampled windows中2,620がsyntheticで、厳密に1/12となる。
game 9だけでcheckpointを選択し、`run.test_after_fit=false`によりgame 10は
実行していない。

### メトリクスの解釈

epoch 0--7 F1は0.585012、0.588702、0.660237、0.648116、0.691354、
0.647816、0.645988、0.656215。best epoch 4はprecision 0.656703、
recall 0.729866、平均距離2.247915 px、loss 0.000406611だった。
paired control bestに対しprecision `+0.019165`、recall `+0.017450`、
F1 `+0.018454`、平均距離`-0.035249 px`で、分類とlocalizationの両方が
改善した。

### アーキテクチャ⇄メトリクスの因果考察

観測上、1/6 mixで低下したreal recallが1/12 mixではcontrolを上回った。
仮説として、real window置換量を半減したことでreal分布の学習量を保ちつつ、
3DGS背景・物理軌道・occlusionを持つsyntheticから得る追加variationが有効に
なった可能性がある。ただし単一seedではsampling varianceを排除できないため、
syntheticの因果効果確定には残りpaired seedsが必要である。

### 既存実験との比較

同一seedの親controlに対してbest F1は0.672900から0.691354へ改善した。
1/6 treatment `run-i618-3dgs-blcs-v1-treatment-s731`の0.663127に対しても
`+0.028227`高く、露出低減という単一変更が負の結果を反転した。C02配備モデル
とは評価集約が異なるため、このvalidation結果だけで最終改善は主張しない。

### 次に有効な実験

既に直列queueへ追加済みのseeds 1931/3253について、同一budgetのcontrolと
1/12 treatmentを完了する。3 seed中2 seed以上でpaired F1 deltaが正かつ
aggregate deltaが正というfrozen reproducibility gateをgame 9だけで判定し、
全checkpoint固定後にのみgame 10とpaired uncertainty/protected gatesへ進む。
