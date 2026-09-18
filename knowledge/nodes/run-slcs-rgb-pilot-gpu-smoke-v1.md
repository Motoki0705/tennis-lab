---
id: run-slcs-rgb-pilot-gpu-smoke-v1
type: run
title: 'SLCS実RGB pilot: 1epoch GPU smokeの関節confidence契約違反'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: failed
config:
  model: SLCSFusionModel hidden_dim=128, shared_layers=4, dino_patch_downsample_factor=2
  loss: SLCSLoss (resolved_config.yaml参照)
  data: slcs/real_rgb_pilot_v1
  augmentation_enabled: true
  max_epochs: 1
  warmup_steps: 0
metrics: {}
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
    .venv/bin/python -m src.tasks.slcs.scripts.train --config-name train_real_rgb_pilot
    training.trainer.max_epochs=1 training.warmup_steps=0 run.output_dir=slcs/train/real_rgb_pilot_gpu_smoke/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-rgb-pilot-gpu-smoke-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_pilot_gpu_smoke/s42-001/logs/version_0
  log: knowledge/runs/run-slcs-rgb-pilot-gpu-smoke-v1/queue.log
parents:
- run-slcs-rgb-pilot-baseline-e60-v1
- run-slcs-rgb-pilot-augmented-e60-v1
relations: []
tags:
- slcs
- real-rgb
- pilot
- failed
- startup-contract
---

## 考察 / Findings

### 要約
TensorBoard設定保存の修正後、1epoch GPU smokeは学習epoch内のmodel I/O検証で停止した。`player_kp_vis values must lie in [0, 1]`が原因でexit_code=1。完走・test評価はしていない。

### アーキテクチャ詳細
SLCSFusionModelはhidden_dim=128、共有4層、DINO patch縮小factor=2。Meiji 2 clipとbroadcast 5 clipを統合した`slcs/real_rgb_pilot_v1`を使用。trainはvideo_000/shanghai/washington、valはvideo_001/indoorhard、testはeastbourneで、Meiji test収録はない先行試験。3D教師は独立実測GTではない。 augmentation.enabled=true、max_epochs=1、warmup_steps=0。実行時の設定正本をbundle内`resolved_config.yaml`へ保存した。現在のprofileはv2へ更新済みでも、この失敗runはv1である。 60epoch比較前の実GPU・logger経路の確認を目的に1epochとwarmup 0へ短縮した。

### メトリクスの解釈
test指標は得られていないためmetricsは空。失敗までのlogger記録があっても収束・汎化・RGB改善の証拠ではない。ログ上のtrain batchesは6でlog_every_n_steps=20より少なく、曲線の不在や疎さを学習安定性と解釈しない。

### アーキテクチャ⇄メトリクスの因果考察
ViTPoseの生heatmap peakは確率ではなく1を超え得る。失敗後のデータ監査では旧Meiji 2 clipに範囲超過が9件と46件、最大値1.03125と1.046875で確認された。consumerの[0,1]契約を緩めるのでなく、生成側で明示的にclip[0,1]し変換をmetadataへ記録、consumerの厳格な拒否を維持してpilot_v2へ再生成した（親作業で確認済みの修正経路）。このrun自体は修正前v1の失敗であり、v2による精度改善を示さない。

### 既存実験との比較
parentsの2 runは学習前に凍結dataclassのTensorBoard保存で停止した。本runは`save_hyperparameters({"config": config})`への修正後にその段階を通過したが、別の入力契約違反を検出した。モデル品質の優劣は比較できない。loggerを省くCPU fast_dev_runだけでは本学習経路を証明できず、全入力範囲の監査も必要だった。

### 次に有効な実験
再生成pilot_v2を全windowで検査し、実TensorBoardLoggerを使う回帰試験と1epoch GPU smokeの完走後に60epochのbaseline/augmentation比較へ進む。同じcheckpointの4入力条件評価はtarget/mask/weight対応を固定し、擬似教師一致度として報告する。
