---
task: slcs
sequence: 105
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-v9-feature-reuse-v1
type: run
title: Meiji全56clipのRGB特徴を照合しv9へ再利用
provider: codex
date: '2026-09-19'
status: done
config:
  device: cpu
  source: slcs/meiji_rgb_v8
  target: slcs/meiji_rgb_v9
  spec:
    backbone: dinov3_vitb16
    patch_size: 16
    image_height: 256
    image_width: 448
    embed_dim: 768
    frame_stride: 10
  checkpoint_sha256: 73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c
metrics:
  clips: 56
  cameras: 168
  linked: 56
  already_valid: 0
  prepublication_input_files: 915
  final_input_and_output_files: 1139
  hash_mismatches: 0
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-feature-reuse-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_feature_reuse/s42-001
  audit: knowledge/runs/run-slcs-meiji-v9-feature-reuse-v1/audit.json
parents:
- run-slcs-meiji-v8-features-missing-v1
- run-slcs-meiji-video-alias-audit-v1
relations: []
tags:
- slcs
- meiji
- rgb-features
- reuse
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 91dc8895
  branch: codex/slcs-real-rgb
  command: env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python
    -B knowledge/runs/run-slcs-meiji-v9-feature-reuse-v1/reuse_features.py --original
    /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset
    --source /home/kamimura/projects/tennis-lab/data/slcs/meiji_rgb_v8 --target /home/kamimura/projects/tennis-lab/data/slcs/meiji_rgb_v9
    --checkpoint /home/kamimura/projects/tennis-lab/third_party/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
    --output-dir /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_feature_reuse/s42-001
---

## 考察 / Findings

### 要約
原本Meijiとv8のmedia・manifest・producer/spec・RGB配列を照合し、対象56clip168cameraの特徴をv9へ再利用した。公開前915入力と、公開した特徴を含む最終1139ファイルのSHA照合が通過した。旧失敗監査は保存したままで、間欠的なhash不一致の原因が解決したとは扱わない。

### アーキテクチャ詳細
固定DINOv3 ViT-B/16・256×448・patch16・embed768・stride10、CPUのみ。v9にはproduction materialize_datasetで原本manifestとimmutable mediaを配置し、全56source cacheをproduction readerで検証する。npzのfloat16/int64、有限値・shape、exact frame indicesとmarker sample countも検査。全preflight後、旧検証済みmigrate_clipのatomic no-replace hardlinkでRGB特徴だけを公開し、各targetも再検証した。3D教師はコピーしない。

### メトリクスの解釈
linked56、already_valid0、168camera。公開前は915ファイルのhash一致を確認。公開した224ファイルを初回取得して検証対象へ加え、最終1139の全digestが初回記録と一致した。エラー0。全体モデル精度を示す指標ではなく、入力同一性と特徴cacheの検査結果である。学習曲線は無い。

### アーキテクチャ⇄メトリクスの因果考察
RGB featureはCourtや3D教師に依存しないため、同一media・manifest・producer/specのままCourtをv9へ変更する場合に再利用可能である。hardlinkはimmutabilityを前提とし、変更があれば次の検証で拒否する。今回の成功は全ての実行環境・将来の読取の安定性を証明しない。

### 既存実験との比較
v8の全体監査は1動画の終了時SHA不一致でfailedし、別process snapshotは一致した。本runは両記録をSHA付き参照し、上書きせず別の入力照合を実施した。並行したv9人物観測再利用はViTPose hash不一致で停止しており、その失敗と本runの成功を区別する。

### 次に有効な実験
人物観測の読取不一致の切り分けとv9観測の生成を進める。全体教師が完成した後、production strict品質レポートと固定splitの統合で本特徴を再検証し、SLCS学習へ使用する。
