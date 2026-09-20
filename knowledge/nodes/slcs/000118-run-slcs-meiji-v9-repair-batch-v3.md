---
task: slcs
sequence: 118
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-v9-repair-batch-v3
type: run
title: 'Meiji v9全件監査で残った3clipを修復・差し替え'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/meiji_rgb_v9_repair_batch_v3
  clips: [video_002/clip_017, video_002/clip_018, video_002/clip_020]
  stage: infer
  device: cpu
metrics:
  generated_clips: 3
  audited_clips: 3
  audit_error_clips: 0
  raw_ball_reprojection_mean_px: 12.307619045888037
  refined_ball_reprojection_mean_px: 6.771689268648406
  supported_player_speed_max_mps: 11.551239967346191
  supported_ball_speed_max_mps: 53.536407470703125
repro:
  commit: 57726f2c8f38fe50ab3870a7a030fad78ab14b4b
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tennis_scene.scripts.build_slcs_dataset device=cpu stage=infer
    'dataset_clip_ids=[video_002/clip_017,video_002/clip_018,video_002/clip_020]'
    dataset_output_directory=slcs/meiji_rgb_v9_repair_batch_v3
    output_dir=tennis_scene/generate/meiji_rgb_v9_repair_batch/s42-005
    plcs_checkpoint=plcs/real-rgb-meiji-foot-e60-v1.restore-20260919.ckpt
    blcs_checkpoint=blcs/real-rgb-meiji-e60-v1.restore-20260919.ckpt
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-repair-batch-v3
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v9_repair_batch/s42-005
parents: [run-slcs-meiji-v9-full-qc-v1]
relations: [{to: run-slcs-meiji-v9-repair-clip004-v1, rel: confirms}]
tags: [slcs, meiji, quality, provenance, cpu, repair]
---

## 考察 / Findings

### 要約

残存3clipをguard付きCPU経路で再生成し、媒体・RGB特徴を含むsubset監査が成功した。
旧教師・生成成果物を退避し、監査済み成果物を本体へ差し替えた。

### アーキテクチャ詳細

採用済み学習checkpointの固定pinを維持した復元コピーを明示指定。観測・幾何補正・閾値は同じ。
DINO特徴は同一媒体の本体cacheをコピーし、監査で検証した。

### メトリクスの解釈

3clipの同じ最終正weight maskで、ball再投影平均は12.3076→6.77169px。
ball支持率はclip017が96.59%、018が88.21%、020が84.96%。
これは擬似教師と観測の整合性であり、独立3D精度ではない。学習曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

pin・raw・producerが一致する成果物だけを公開し、記録を書き換える方法は使っていない。
重みの読取不一致の根本原因解決をこの成功から推測しない。

### 既存実験との比較

全件監査1の3件のerrorを修復した。これまでの合計9clipで旧成果物を保存している。
今回のsubset監査と、本体56clipの全件監査は区別する。

### 次に有効な実験

本体をallow_incomplete=falseで再監査し、成功後に固定split付きの全体データへ統合する。
