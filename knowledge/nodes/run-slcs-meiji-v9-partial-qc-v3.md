---
id: run-slcs-meiji-v9-partial-qc-v3
type: run
title: 'Meiji v9部分監査3: 修復3件を確認、後半3件の記録不一致'
provider: codex
date: '2026-09-19'
status: failed
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config: {data: slcs/meiji_rgb_v9, allow_incomplete: true, device: cpu}
metrics: {completed_clips: 45, missing_clips: 8, error_clips: 3, excluded_clips: 1}
repro:
  commit: 25f4beafe1156371d00b5bf70ae1847b31ab5eb5
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tennis_scene.scripts.report_slcs_dataset_quality
    allow_incomplete=true output_dir=tennis_scene/analyze/meiji_rgb_v9_quality/s42-takeover-003
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-partial-qc-v3
  output_dir: outputs/tennis_scene/analyze/meiji_rgb_v9_quality/s42-takeover-003
parents: [run-slcs-meiji-v9-repair-batch-v1]
relations: [{to: run-slcs-meiji-v9-partial-qc-v2, rel: compares}]
tags: [slcs, meiji, quality, provenance, cpu]
---

## 考察 / Findings

### 要約

45clipが監査成功。先行修復した3clipはいずれも本体datasetで成功し、後半の3clipで
PLCSのraw/producer checkpoint記録の不一致を検出した。全件生成中のsnapshotであり、全体採用ではない。

### アーキテクチャ詳細

前回と同じCPU品質監査で、公開済み教師・媒体・DINO特徴・観測receiptを照合した。
実行中のGPU生成は公開前guardを追加する前に開始したプロセスであり、途中のコード変更は適用されない。

### メトリクスの解釈

不足8clipはallow_incompleteにより許容したが、不正3clipで終了コード1。
全体producer設定の混在エラーはない。新規学習はなく、収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

`video_001/clip_020`、`video_002/clip_004`、`video_002/clip_015` が失敗した。
正しい固定pin `e851a8fe...` に対し、`d09130f1...` または `036ff869...` が
rawまたはproducerの片側に記録されている。原因を断定せず、metadataの手修正や例外許可はしない。

### 既存実験との比較

前回不正だった `video_001/clip_010`、`video_001/clip_011` と先行 `video_000/clip_009` は成功。
未修復の不正教師をモデル品質の根拠に含めない。

### 次に有効な実験

新たな3clipを同じ固定重みと観測でCPU再生成し、subset監査後に旧成果物を退避して差し替える。
全件生成後にはallow_incomplete=falseで改めて全体を監査する。
