---
id: run-court-scale-manifest-training-smoke-s42
type: run
task: court_detection
sequence: 49
recorded_at: '2026-10-23'
title: Courtデータ版manifestからの3更新GPU接続確認
issue: 1354
provider: codex
session: 01a16433-cb23-7453-8370-31ba2f87eb60
date: '2026-10-23'
status: done
config:
  model:
    backbone: dinov3_vitb16
    transformer_dim: 768
    frozen_backbone: true
  data:
    source: synthetic_court_dataset_version
    batch_size: 1
    input_long_side: 512
  training:
    max_steps: 3
    warmup_steps: 1
    validation_every_n_steps: 3
    compile: false
metrics:
  optimizer_updates: 3
  accepted_images: 298
  real_val_images: 2
repro:
  commit: 61c948c5338f18d620528f1ca6515cdbf9d69d63
  branch: codex/court-scale-training-eval
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: "#!/usr/bin/env bash\nset -euo pipefail\nTASK_ROOT=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-scale-training-eval\n\
    ASSET_ROOT=/home/kamimura/projects/tennis-lab\ncd \"$TASK_ROOT\"\nexport OMP_NUM_THREADS=2\
    \ MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n.venv/bin/python\
    \ -m src.tasks.court_detection.scripts.train --config-name train_data_scaling\
    \ \\\n  \"paths.data_root=$ASSET_ROOT\" \\\n  \"paths.checkpoint_root=$ASSET_ROOT/ckpt\"\
    \ \"paths.external_asset_root=$ASSET_ROOT/third_party\" \\\n  \"paths.output_root=$TASK_ROOT/tmp/court-scale-manifest-smoke/outputs\"\
    \ \\\n  \"paths.artifact_root=$TASK_ROOT/tmp/court-scale-manifest-smoke/outputs\"\
    \ \\\n  data.source.manifest=.claude/worktrees/court-scale-trajectory-strategies/tmp/court-scale-review/run-04/1bef5ceda3d453b269cbb99e6b0d422e1e8466ad4e89d527cfa12b67d361dcda.dataset.json\
    \ \\\n  mixed.sources.tennis_court_detector.root=.claude/worktrees/court-scale-training-eval/tmp/court-scale/smoke-data/real\
    \ \\\n  mixed.sources.tennis_court_detector.registry=court_detection/evaluate/smoke-real-video-v1/s42/registry.json\
    \ \\\n  data.batch_size=1 mixed.train_batch_counts.synthetic_court=1 data.num_workers=0\
    \ \\\n  training.compile.enabled=false training.trainer.enable_progress_bar=false\
    \ training.trainer.enable_model_summary=false \\\n  training.trainer.log_every_n_steps=1\
    \ training.warmup_steps=1 training.step_schedule.max_steps=3 \\\n  training.step_schedule.validation_every_n_steps=3\
    \ training.step_schedule.checkpoint_every_n_steps=3 \\\n  run.output_dir=court_detection/train/dataset-version-smoke/s42\n\
    CKPT=\"$TASK_ROOT/tmp/court-scale-manifest-smoke/outputs/court_detection/train/dataset-version-smoke/s42/logs/version_0/checkpoints/update-00000003.ckpt\"\
    \n.venv/bin/python -m src.tasks.court_detection.scripts.evaluate_ablation \\\n\
    \  --checkpoint \"$CKPT\" --expected-step 3 --split val --series smoke \\\n  --output\
    \ \"$TASK_ROOT/tmp/court-scale-manifest-smoke/outputs/court_detection/evaluate/dataset-version-smoke-val/s42\""
artifacts:
  run_dir: knowledge/runs/run-court-scale-manifest-training-smoke-s42
  config: knowledge/runs/run-court-scale-manifest-training-smoke-s42/config.yaml
  data_condition: knowledge/runs/run-court-scale-manifest-training-smoke-s42/data-condition.json
  evaluation: knowledge/runs/run-court-scale-manifest-training-smoke-s42/evaluation-val.json
  verification: knowledge/runs/run-court-scale-manifest-training-smoke-s42/verification.json
  validation: knowledge/runs/run-court-scale-manifest-training-smoke-s42/validation.json
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/0-8207316784400436991_2026-10-22T15-33-35-599563008Z_1964939_court-scale-manifest-training-smoke.log
  output_dir: tmp/court-scale-manifest-smoke/outputs/court_detection/train/dataset-version-smoke/s42/logs/version_0
  curves: knowledge/runs/run-court-scale-manifest-training-smoke-s42/curves.png
  tb_logdir: tmp/court-scale-manifest-smoke/outputs/court_detection/train/dataset-version-smoke/s42/logs/version_0
parents:
- run-court-data-scaling-smoke-s42
relations: []
papers: []
tags:
- court
- data-scaling
- integration-smoke
- synthetic-only
- manifest
---

## 観測

生成器の本番recipeによる4シーンの正式なデータ版を、学習側の
`synthetic_court_dataset_version` sourceから読み、GPUで3更新と実画像val評価を完了した。
採択済み298枚の内訳はB00 71・B01 80・B02 71・B03 76枚で、
`load_dataset_version(..., verify_images=True)`が採択集合、QA ledger、hash、
stable ID由来の幾何、train-only splitを検証した後のDataset件数と一致した。
データ版・生成seed 42・manifest来歴を学習記録とcheckpointに保存し、
3更新checkpointの厳格な再読込から実画像val 2枚を評価できた。
生成済みstoreは読み取りだけで、画像生成・既存sceneの書換えは行っていない。

モデルは凍結ViT-B＋幅768、長辺512、poseを含む共通教師・lossを使用した。
共有queueの`resource=all`、batch 1、compile無効、worker 0で実行した。
valは前回の接続確認で元実画像から抽出した固定2動画・2枚であり、
全体の実画像val 4,468枚を評価した結果ではない。testは実行していない。
初回登録は絶対パスの子要素をPathResolverがGPU起動前に拒否した。
DATA/ARTIFACT root相対の入力に直して再登録した実行を本ノードへ保存した。

## 解釈と次の確認

従来の8枚storeによる確認から、生成器と学習側の正式なmanifest境界の
接続まで確認範囲が広がった。3更新とval 2枚の指標から精度・収束・
本番batch 8の容量は判断しない。既存モデルの採用判断は維持する。
後続のColab試走では同じmanifest入口を使い、本番の更新数・評価間隔・
容量と実画像全体での評価を固定する。共有GPU時間の見積りには本確認を流用しない。

## 通常検証

CPU生成器の正式なデータ版を学習Datasetで読む結合テスト27件と、
既存のv3入力・混合設定・生成器の回帰176件が成功した。
採択集合との件数一致、storeと読み込み時のKP・可視性・pose・RGBの一致、
既存前処理後の教師、不正hash・seed・非train sampleの拒否、
CPUの6更新・checkpoint保存・再評価を確認した。全pytestは共有flock経由・4 workersで実行した。
初回のfixture前提と教師構造に関するassertionを修正して再実行し、
学習側の実装はGPU確認時から変更していない。ruff・mypyを含むpre-commitも成功した。
実行コマンド、fixture修正と検査基準はvalidation receiptを参照する。
