---
id: run-court-render-batches-v1
type: run
title: 本番NHTの常駐ロードとカメラバッチ検証
provider: codex
session: 01a0985c-eb87-7310-9657-5411e2818d4e
date: '2026-09-13'
status: done
config:
  renderer: NHT public render_scene, RTX 5060 Ti
  batch_sizes:
  - 1
  - 4
  - 8
  data: B00–B03 observed cameras, 8 per scene
metrics:
  scenes: 4
  cameras_per_scene: 8
  checkpoint_loads_per_public_call: 1
  max_abs_error_rgb_alpha_depth: 0.0
repro:
  commit: 21199b3e29746c9d6360f287462f1ca400865690
  branch: codex/court-storage-ondemand
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/benchmark_court_render_batches.py
    --nht-python /home/kamimura/.local/share/uv/tools/nht/bin/python --nht-repo /home/kamimura/projects/tennis-lab/.claude/worktrees/nht-court-storage-ondemand
    --scenes /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes
    --output outputs/storage-production/render-batches-v1
  nht_commit: 9493001f1f5650f62e8f9809b023618edcde3f03
artifacts:
  run_dir: knowledge/runs/run-court-render-batches-v1
  metrics: knowledge/runs/run-court-render-batches-v1/metrics.json
parents:
- run-court-storage-views-v1
relations: []
tags:
- court
- rendering
- storage
---

## 考察 / Findings

### 要約
B00–B03各8カメラで公開コマンドのcheckpointロードは1回。
バッチ1/4/8のRGB・alpha・depthの最大絶対差はすべて0。

### アーキテクチャ詳細
公開`render_scene`でcheckpointとshaderを一度初期化する。
同解像度のカメラをまとめてgsplatへ渡し、C=1を要求する固定shaderには
カメラ別に入力する。既定バッチは4、解像度変化で分割し順序を保持する。
NHT変更はreproのnht_commitで固定。学習時レンダリングは導入しない。

### メトリクスの解釈
8枚のwarm中央値はバッチ1で0.174–0.198秒、4で0.186–0.207秒、8で0.187–0.212秒。
CUDA peak allocatedはそれぞれ約0.45–0.46/1.79–1.80/3.31–3.33 GiB。
公開コマンドのロード・PNG/NPY保存・検証込みは1.80–2.03秒/8枚。
メモリ値はTorch allocatorの値で、プロセス全体やGPU全体の使用量ではない。

### アーキテクチャ⇄メトリクスの因果考察
今回、ロード済み状態でのバッチ化自体は単枚より4–7%程度遅い。
改善の確実な根拠は画像ごとのcheckpoint/shader再初期化を除去したこと。
旧公開経路との同条件end-to-end速度比はこのrunでは測定していない。
OOM等を単枚処理へ静かにフォールバックしない。必要なら明示バッチ1を指定する。

### 既存実験との比較
先行実験は研究workerで常駐化し公開CLIは変更していなかった。
今回は公開CLI経路を接続して実際のファイル出力も比較した。

### 次に有効な実験
より大きい解像度・異なるcheckpointでバッチ数とVRAMの関係を測る。
本採用は事前生成＋圧縮保存であり、学習オンデマンド方式の導入は不要。
