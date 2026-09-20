---
task: slcs
sequence: 117
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-v9-repair-batch-v2
type: run
title: 'Meiji v9後半3clip修復: 検証済みPLCSコピーで2件生成、BLCS照合で1件停止'
provider: codex
date: '2026-09-19'
status: failed
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/meiji_rgb_v9_repair_batch_v2
  clips: [video_001/clip_020, video_002/clip_004, video_002/clip_015]
  stage: infer
  device: cpu
metrics: {published_clips: 2, failed_clips: 1}
repro:
  commit: 25f4beafe1156371d00b5bf70ae1847b31ab5eb5
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tennis_scene.scripts.build_slcs_dataset device=cpu stage=infer
    'dataset_clip_ids=[video_001/clip_020,video_002/clip_004,video_002/clip_015]'
    dataset_output_directory=slcs/meiji_rgb_v9_repair_batch_v2
    output_dir=tennis_scene/generate/meiji_rgb_v9_repair_batch/s42-003
    plcs_checkpoint=plcs/real-rgb-meiji-foot-e60-v1.restore-20260919.ckpt
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-repair-batch-v2
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v9_repair_batch/s42-003
parents: [run-slcs-meiji-v9-repair-start-failure-v1]
tags: [slcs, meiji, provenance, checkpoint, cpu, repair]
---

## 考察 / Findings

### 要約

固定pinを照合済みのPLCS復元コピーで2clipのCPU再生成が成功。
`video_002/clip_004` はBLCSのproducer記録が固定pinと異なるため、公開前に拒否した。
成功分は保持したが、run全体は終了コード1であり成功扱いしない。

### アーキテクチャ詳細

採用済みPLCSの学習出力から別パスへコピーし、同じ固定pinを確認した重みを明示指定。
producer identityはcheckpointの場所でなく内容digestを使うため、モデル構成は変わらない。
BLCSと観測・幾何補正・品質閾値は従来の設定を維持した。

### メトリクスの解釈

`video_001/clip_020` と `video_002/clip_015` を生成し、1clipは生成前に停止。
subsetの全体監査は残り1clipの修復後に別runとして行う。学習・曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

開始時のBLCS照合は成功したが、clip単位の読み取りでは期待値 `dd0e54d2...` に対し
`500327c2...` が得られた。原因は未確定。失敗を隠す再試行・metadataの改変はしない。
採用済みBLCSの学習時保存元と別パス復元コピーを検証し、新しいrunで失敗clipだけを処理する。

### 既存実験との比較

先行の開始前PLCS照合失敗からは進んだが、すべての読取安定性が解決したとはいえない。
2clipが公開前guardを通過したことと、データセット全件の監査成功を区別する。

### 次に有効な実験

採用済みPLCS/BLCSの検証済み復元コピーで未生成1clipを明示的に再開し、3clipをまとめて監査する。
旧教師は退避を伴って差し替え、全体生成後に本体の全件監査を行う。
