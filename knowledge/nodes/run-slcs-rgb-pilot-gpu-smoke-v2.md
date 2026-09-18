---
id: run-slcs-rgb-pilot-gpu-smoke-v2
type: run
title: 'SLCS実RGB pilot: 1epoch validation後のcheckpoint監視キー不一致'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: failed
config:
  model: SLCSFusionModel hidden_dim=128, shared_layers=4
  loss: SLCSLoss (resolved_config.yaml参照)
  data: slcs/real_rgb_pilot_v2
  augmentation_enabled: true
  max_epochs: 1
  warmup_steps: 0
  checkpoint_monitor: val/scene_position_error_m
metrics: {}
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
    .venv/bin/python -m src.tasks.slcs.scripts.train --config-name train_real_rgb_pilot
    training.trainer.max_epochs=1 training.warmup_steps=0 run.output_dir=slcs/train/real_rgb_pilot_gpu_smoke/s42-002
artifacts:
  run_dir: knowledge/runs/run-slcs-rgb-pilot-gpu-smoke-v2
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_pilot_gpu_smoke/s42-002/logs/version_0
  log: knowledge/runs/run-slcs-rgb-pilot-gpu-smoke-v2/queue.log
  curves: knowledge/runs/run-slcs-rgb-pilot-gpu-smoke-v2/curves.png
  tb_logdir: outputs/slcs/train/real_rgb_pilot_gpu_smoke/s42-002/logs/version_0
parents:
- run-slcs-rgb-pilot-gpu-smoke-v1
relations: []
tags:
- slcs
- real-rgb
- pilot
- failed
- checkpoint-monitor
---

## 考察 / Findings

### 要約
confidence生成契約を修正したpilot_v2で1epochの学習・validationへ進んだが、epoch末のModelCheckpoint保存時に監視キーを見つけられずexit_code=1で停止した。test評価は未完了。

### アーキテクチャ詳細
SLCS実RGB pilot profileの128次元・共有4層モデル、入力augmentationあり。Meiji 2 clipとbroadcast 5 clipのpilot_v2を使い、max_epochs=1・warmup_steps=0でGPU smokeを行った。splitはv1と同じ先行試験でMeiji test収録はない。生ViTPose peakを生成側で明示clip[0,1]しmetadataへ記録した新データ版であり、consumerのstrict検証は維持した。実行時設定はbundle内resolved_config.yamlに保存した。

### メトリクスの解釈
validationの記録はあるが、本runの目的は学習・logger・checkpoint境界の動作確認である。test精度は未測定のためfrontmatterのmetricsは空。1epochだけの曲線から収束やaugmentation改善を判断しない。疑似教師への一致度を実測3D精度と呼ばない。

### アーキテクチャ⇄メトリクスの因果考察
ModelCheckpointは`val/scene_position_error_m`を要求したが、実際のepoch集計名は`val/scene_position_error_m_epoch`だった。ログの候補一覧に後者だけがあり、保存時のMisconfigurationExceptionを直接説明する。モデル精度の失敗ではなく、profileとloggerの命名契約の不一致である。親はprofileのmonitorを修正済み。本物config・CPU Trainer.fit normal mode・TensorBoardLogger・ModelCheckpointを組み合わせる回帰試験は別workerが追加中で、この記録時点では完了を主張しない。

### 既存実験との比較
parentのgpu-smoke-v1はplayer_kp_visの[0,1]違反でepoch内停止した。本runはその検証を通過してvalidation後まで進んだが、checkpoint保存には失敗した。baseline/augmentation初回2 runのTensorBoard保存失敗からも、境界ごとの検証が必要と分かる。loggerやcheckpoint処理を省くCPU fast_dev_runは本学習経路を証明しない。

### 次に有効な実験
実profileでnormal modeのlogger・validation・checkpoint保存・再読込を回帰試験に含め、修正後の別GPU smokeを完走させてから60epoch比較へ進む。進行中のgpu-smoke-v3の成否は本runに合算せず、親が別ノードで記録する。
