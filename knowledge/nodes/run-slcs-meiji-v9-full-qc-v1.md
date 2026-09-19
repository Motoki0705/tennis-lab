---
id: run-slcs-meiji-v9-full-qc-v1
type: run
title: 'Meiji v9初回全件監査: 欠落0、成功53、記録不一致3'
provider: codex
date: '2026-09-19'
status: failed
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config: {data: slcs/meiji_rgb_v9, allow_incomplete: false, device: cpu}
metrics: {completed_clips: 53, missing_clips: 0, error_clips: 3, excluded_clips: 1}
repro:
  commit: 57726f2c8f38fe50ab3870a7a030fad78ab14b4b
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tennis_scene.scripts.report_slcs_dataset_quality
    output_dir=tennis_scene/analyze/meiji_rgb_v9_quality/s42-full-001
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-full-qc-v1
  output_dir: outputs/tennis_scene/analyze/meiji_rgb_v9_quality/s42-full-001
parents: [run-slcs-meiji-v9-full-build-v1, run-slcs-meiji-v9-repair-clip004-v1]
relations: [{to: run-slcs-meiji-v9-partial-qc-v3, rel: compares}]
tags: [slcs, meiji, quality, provenance, cpu]
---

## 考察 / Findings

### 要約

期待する全56clipの公開を確認したが、53clip成功・3clip不正で監査は失敗した。
先行修復済み6clipはいずれも本体dataset上で監査成功。欠落や全体producerの設定混在はない。

### アーキテクチャ詳細

部分監査と同じコード・閾値を使い、allow_incomplete=falseで媒体・教師・RGB特徴・観測receiptを確認した。
監査自体はcheckpointや既存の生成成果物を変更しない。

### メトリクスの解釈

終了コード1。`video_002/clip_017` はBLCS、`clip_018` と `clip_020` はPLCSの
raw/producer digest不一致。これらを成功集計へ含めず、学習用の全体採用も保留した。
学習を行っておらず、収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

guard追加前に起動した生成プロセスの最後の領域にも記録不一致が残った。
原因の断定や例外拡大はせず、固定pinと同じ採用済み学習checkpointの復元コピーで修復する。

### 既存実験との比較

部分監査3の未生成8clipがすべて公開され、その時点で不正だった3clipは修復成功を確認した。
今回の3件は別clipであり、先行修復が失敗したという意味ではない。

### 次に有効な実験

残存3clipを別dataset版でCPU再生成・監査し、旧成果物を退避して本体へ差し替える。
改めて全件監査を実行し、56成功・0欠落・0不正を確認する。
