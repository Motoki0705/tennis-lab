---
id: run-court-data-scaling-smoke-s42
type: run
task: court_detection
sequence: 48
recorded_at: '2026-10-22'
title: Courtの合成専用・固定20更新と実画像評価のGPU起動確認
issue: 1354
provider: codex
session: 01a1628d-0e82-7393-8a00-582e63df0fb0
date: '2026-10-22'
status: done
config:
  preset: train_data_scaling
  backbone: dinov3_vitb16
  transformer_dim: 768
  train_batch_size: 1
  long_side: 512
  compile: false
  generation_seed: 695
  training_seed: 42
  max_steps: 20
  warmup_steps: 2
  validation_every_n_steps: 10
  checkpoint_every_n_steps: 20
  accepted_images: 8
metrics:
  completed_updates: 20
  final_learning_rate: 1.0e-06
repro:
  commit: 16162608ccd11e9ebfd8a5e0a4e0eee75bb86d49
  branch: codex/court-scale-training-eval
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: "#!/usr/bin/env bash\nset -euo pipefail\nTASK_ROOT=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-scale-training-eval\n\
    ASSET_ROOT=/home/kamimura/projects/tennis-lab\ncd \"$TASK_ROOT\"\nexport OMP_NUM_THREADS=2\
    \ MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n.venv/bin/python\
    \ -m src.tasks.court_detection.scripts.train --config-name train_data_scaling\
    \ \\\n  \"paths.data_root=$TASK_ROOT/tmp/court-scale/smoke-data\" \\\n  \"paths.checkpoint_root=$ASSET_ROOT/ckpt\"\
    \ \"paths.external_asset_root=$ASSET_ROOT/third_party\" \\\n  \"paths.output_root=$TASK_ROOT/outputs\"\
    \ \"paths.artifact_root=$TASK_ROOT/outputs\" \\\n  'data.source.store_roots=[synthetic]'\
    \ data.source.data_version=b00-eight-image-smoke data.source.generation_seed=695\
    \ \\\n  mixed.sources.tennis_court_detector.root=real \\\n  mixed.sources.tennis_court_detector.registry=court_detection/evaluate/smoke-real-video-v1/s42/registry.json\
    \ \\\n  data.batch_size=1 mixed.train_batch_counts.synthetic_court=1 data.num_workers=0\
    \ \\\n  training.compile.enabled=false training.trainer.enable_progress_bar=false\
    \ training.trainer.enable_model_summary=false \\\n  training.trainer.log_every_n_steps=1\
    \ training.warmup_steps=2 training.step_schedule.max_steps=20 \\\n  training.step_schedule.validation_every_n_steps=10\
    \ training.step_schedule.checkpoint_every_n_steps=20 \\\n  run.output_dir=court_detection/train/data-scaling-smoke/s42-final\n\
    CKPT=\"$TASK_ROOT/outputs/court_detection/train/data-scaling-smoke/s42-final/logs/version_0/checkpoints/update-00000020.ckpt\"\
    \nfor SPLIT in val test; do\n  .venv/bin/python -m src.tasks.court_detection.scripts.evaluate_ablation\
    \ \\\n    --checkpoint \"$CKPT\" --expected-step 20 --split \"$SPLIT\" --series\
    \ smoke \\\n    --output \"$TASK_ROOT/outputs/court_detection/evaluate/data-scaling-smoke-$SPLIT/s42-final\"\
    \ndone\n.venv/bin/python -m src.tasks.court_detection.scripts.compare_data_scaling\
    \ \\\n  --evaluations \"$TASK_ROOT/outputs/court_detection/evaluate/data-scaling-smoke-val/s42-final/evaluation.json\"\
    \ \"$TASK_ROOT/outputs/court_detection/evaluate/data-scaling-smoke-test/s42-final/evaluation.json\"\
    \ \\\n  --output \"$TASK_ROOT/outputs/court_detection/report/data-scaling-smoke/s42-final\""
artifacts:
  run_dir: knowledge/runs/run-court-data-scaling-smoke-s42
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/0-8207343476796513194_2026-10-22T08-08-43-203486805Z_2920561_court-data-scaling-final-smoke.log
  output_dir: outputs/court_detection/train/data-scaling-smoke/s42-final/logs/version_0
  verification: knowledge/runs/run-court-data-scaling-smoke-s42/verification.json
  real_evaluation_registry: knowledge/runs/run-court-data-scaling-smoke-s42/real-evaluation-registry.json
  validation_evaluation: knowledge/runs/run-court-data-scaling-smoke-s42/evaluation-val.json
  test_evaluation: knowledge/runs/run-court-data-scaling-smoke-s42/evaluation-test.json
  comparison: knowledge/runs/run-court-data-scaling-smoke-s42/comparison.md
  scaling_curve: knowledge/runs/run-court-data-scaling-smoke-s42/scaling.png
  configuration: knowledge/runs/run-court-data-scaling-smoke-s42/config.yaml
parents: []
relations: []
papers: []
tags:
- smoke
- synthetic-only
- fixed-updates
- video-disjoint-evaluation
---

## 確認したこと

#1354の学習・評価経路を共有training queueの全GPU予約で実行した。入力は、既存B00を読み取り専用にして別storeへコピーした8枚と、実画像の4動画・各1枚（除外注釈1枚も別途コピー）の起動確認用storeである。生成seed 695と学習seed 42を別記録した。

凍結ViT-B/16・LoRAなし、Transformer幅768・8層・12 heads・FFN 2048、DPT 512ch、dense head hidden 256・depth 2、長辺512を使用した。ローカルGPUの容量に配慮し、batch 1・compile無効にした。本設定の総batch 8を検証した結果ではない。

20更新を完了し、更新数に依存するschedulerの状態20・最終学習率1e-6をcheckpointから確認した。10・20更新で実画像valだけを監視し、20更新ちょうどのcheckpointから実画像val/testを別々に評価した。すべてのtrain loss・pose loss・validation lossは有限値であり、poseも学習した。主評価JSONはKP mean/median誤差・SEG/semantic LINE mIoU・LINE Diceだけを含み、poseを集計しない。val/testを混ぜずに比較表・対数軸の曲線を生成できた。

元のTensorBoardはrun出力に残し、主要scalarの全20更新をverification.jsonへ抽出した。短い動作確認であり、学習曲線の研究的解釈や従来モデルとの精度比較は行わない。

## 実画像全体の固定分割

別途、元の実画像データ全体に対してreal-video-v1・seed 42・動画数50:50の分割一覧を作った。退化注釈1件を除いた8,840枚・498動画を次のように固定し、動画重複は0だった。

| split | 動画数 | 枚数 | 旧train由来 | 旧val由来 |
|---|---:|---:|---:|---:|
| val | 249 | 4,468 | 3,359 | 1,109 |
| test | 249 | 4,372 | 3,270 | 1,102 |

全体の一覧とsource/hashはreal-evaluation-registry.jsonに固定した。この全体testは今回評価していない。GPU起動確認で評価したのは別版smoke-real-v1のval 2枚・test 2枚だけであり、本番の性能推定やcheckpoint選択には使わない。

## 再現と制約

run.json/repro.shは、全sourceがコミットされた状態の正確なqueue commandを保持する。prepare-smoke-inputs.pyは既存データから小さな入力を別出力へ作るCPU準備の記録で、元データを変更しない。再実行時は既存runを上書きせず、新しい出力を明示する。checkpointと生のTensorBoardは通常のoutputsに置き、Gitには小さな設定・指標・図・固定分割と再現情報だけを保存する。

既存のsize ablationや採用モデルの判断は変更しない。#1356で本番batch 8の容量・速度、共通更新数・warmup・評価/保存間隔をColab試走から固定する。条件manifestのschemaと正式adapterは#1352の責務であり、今回の最小入口は明示したstore root列である。

## 通常検証

実装のpre-commitはruff・mypyとも通過した。最終版の追加回帰は189件成功で、明示store・不正入力拒否、旧設定compose、CPU学習と再開、指定更新数評価、val/test別集計、seed/画像数の比較と対数軸図、既存の出力境界を確認した。直前の広い回帰は300件成功・1件skip（TENNIS_COURT_TEST_ROOT未指定の任意実データテスト）だった。実行した正確な対象と区別はvalidation.jsonを参照する。
