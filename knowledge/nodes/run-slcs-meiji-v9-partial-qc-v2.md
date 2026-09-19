---
id: run-slcs-meiji-v9-partial-qc-v2
type: run
title: 'Meiji v9部分監査2: clip009修復を確認、後続2clipの記録不一致'
provider: codex
date: '2026-09-19'
status: failed
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config: {data: slcs/meiji_rgb_v9, allow_incomplete: true, device: cpu}
metrics: {completed_clips: 26, missing_clips: 28, error_clips: 2, excluded_clips: 1}
repro:
  commit: ab394121
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tennis_scene.scripts.report_slcs_dataset_quality
    allow_incomplete=true output_dir=tennis_scene/analyze/meiji_rgb_v9_quality/s42-takeover-002
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-partial-qc-v2
  output_dir: outputs/tennis_scene/analyze/meiji_rgb_v9_quality/s42-takeover-002
parents: [run-slcs-meiji-v9-clip009-repair-v1]
relations: [{to: run-slcs-meiji-v9-partial-qc-v1, rel: compares}]
tags: [slcs, meiji, quality, provenance, cpu]
---

## 考察 / Findings

### 要約

差し替え後のclip009は本体datasetでも監査成功。26clipが成功し、後続の
`video_001/clip_010` と `video_001/clip_011` はPLCSのraw/producer記録不一致で失敗した。
生成途中のsnapshotであり、全体教師は未採用。

### アーキテクチャ詳細

前回と同じread-only品質監査。公開済みclipを読むだけで、checkpointや教師metadataは変更しない。
継続中のGPUジョブは公開前guard追加前に開始したプロセスで、新しいguardはまだ適用されていない。

### メトリクスの解釈

未生成28clipはallow_incompleteにより許容するが、2件のerrorで終了コード1。
全体で異なるproducer設定が混在するエラーはない。学習を行っていないため収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

エラーdigestの組は前回と同じ `e851a8fe...` と `78ef9815...` であり、raw/producerのどちら側が
異なるかはclip間で逆。原因の断定や新しい例外許可はせず、指定重みでの再生成を必要とする。

### 既存実験との比較

前回の不整合clipは解消し、他clipを隠すことなく後続の不一致が検出された。
失敗した旧教師の数値をモデル品質の証拠として採用しない。

### 次に有効な実験

残clipの生成が終わってから、不一致をまとめてCPU再生成・退避付き差し替えし、全体監査を通す。
