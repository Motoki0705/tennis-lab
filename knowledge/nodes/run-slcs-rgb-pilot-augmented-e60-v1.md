---
id: run-slcs-rgb-pilot-augmented-e60-v1
type: run
title: 'SLCS実RGB pilot: augmentationあり60epoch起動のTensorBoard保存失敗'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: failed
config:
  model: SLCSFusionModel hidden_dim=128, shared_layers=4, dino_patch_downsample_factor=2
  loss: SLCSLoss (resolved_config.yaml参照)
  data: slcs/real_rgb_pilot_v1
  augmentation_enabled: true
  max_epochs: 60
  warmup_steps: 20
metrics: {}
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
    .venv/bin/python -m src.tasks.slcs.scripts.train --config-name train_real_rgb_pilot
    run.output_dir=slcs/train/real_rgb_pilot_augmented/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-rgb-pilot-augmented-e60-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_pilot_augmented/s42-001/logs/version_0
  log: knowledge/runs/run-slcs-rgb-pilot-augmented-e60-v1/queue.log
parents: []
relations:
- to: run-slcs-rgb-pilot-baseline-e60-v1
  rel: compares
tags:
- slcs
- real-rgb
- pilot
- failed
- startup-contract
---

## 考察 / Findings

### 要約
augmentationあり60epoch試験はTensorBoardのhyperparameter設定保存時に凍結dataclassを処理できず、学習前にexit_code=1で停止した。augmentationやRGB特徴の精度比較結果は得られていない。

### アーキテクチャ詳細
SLCSFusionModelはhidden_dim=128、共有4層、DINO patch縮小factor=2。Meiji 2 clipとbroadcast 5 clipを統合した`slcs/real_rgb_pilot_v1`を使用。trainはvideo_000/shanghai/washington、valはvideo_001/indoorhard、testはeastbourneで、Meiji test収録はない先行試験。3D教師は独立実測GTではない。 augmentation.enabled=true、max_epochs=60、warmup_steps=20。実行時の設定正本をbundle内`resolved_config.yaml`へ保存した。現在のprofileはv2へ更新済みでも、この失敗runはv1である。 対照との差は入力augmentationの有無であり、教師やsplitを変える設計ではない。

### メトリクスの解釈
学習前停止のためtest指標・収束結果はない。metricsは空とし、60epochは予定上限であって完了epoch数ではない。TensorBoard eventファイルの存在だけでは学習成功を意味しない。

### アーキテクチャ⇄メトリクスの因果考察
BaseLightningModuleの`save_hyperparameters("config")`が呼出しframeを探索し、SLCS runtimeの凍結dataclassを拾った。その後TensorBoardのYAML保存で`A frozen dataclass was passed to apply_to_collection`となった。これは設定保存境界の失敗であり、モデル構造やaugmentationの学習品質を原因とする証拠はない。親は明示mappingの`save_hyperparameters({"config": config})`へ修正し、本物のTensorBoardLoggerを用いた回帰試験を追加済み。

### 既存実験との比較
augmentationなしのbaselineも同じ凍結dataclassエラーで学習前停止したため、両条件の精度・収束差は判断できない。先行するCPU fast_dev_runはloggerを省く経路のため、この本学習時の保存境界を証明していなかった。

### 次に有効な実験
実TensorBoardLoggerで設定保存を検証し、共有queueで短いGPU smokeを完走してから60epoch比較へ進む。その後のsmokeで発見されたViTPose confidence範囲違反には、生成側の明示clipとmetadata記録、consumer strict拒否、pilot_v2再生成が必要となった。修正後runは別ノードで結果を記録し、この失敗runへ精度や改善効果を追記しない。
