---
id: run-slcs-meiji-v9-repair-start-failure-v1
type: run
title: 'Meiji v9後半修復: 開始前のPLCS固定pin照合で停止'
provider: codex
date: '2026-09-19'
status: failed
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/meiji_rgb_v9_repair_batch_v2
  clips: [video_001/clip_020, video_002/clip_004, video_002/clip_015]
  stage: infer
  device: cpu
metrics: {published_clips: 0}
repro:
  commit: 25f4beafe1156371d00b5bf70ae1847b31ab5eb5
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tennis_scene.scripts.build_slcs_dataset device=cpu stage=infer
    'dataset_clip_ids=[video_001/clip_020,video_002/clip_004,video_002/clip_015]'
    dataset_output_directory=slcs/meiji_rgb_v9_repair_batch_v2
    output_dir=tennis_scene/generate/meiji_rgb_v9_repair_batch/s42-002
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-repair-start-failure-v1
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v9_repair_batch/s42-002
parents: [run-slcs-meiji-v9-partial-qc-v3]
tags: [slcs, meiji, provenance, checkpoint, cpu]
---

## 考察 / Findings

### 要約

後半3clipのCPU再生成は、公開前のstage開始時にPLCS checkpointの固定pinと実測digestが
一致せず停止した。教師は1件も公開されず、チェックを無効化して続行していない。

### アーキテクチャ詳細

既存のCPU修復と同じ設定。開始時の `verify_checkpoint_integrity` が停止したため、
観測やモデル推論、品質評価には到達していない。

### メトリクスの解釈

期待値は `e851a8fe...`、実測値は `78ef9815...`。終了コード1。
精度や収束の数値を得たrunではなく、曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

この結果だけではファイルの永続的破損やハードウェア原因を断定できない。
別inodeの学習時保存元 `version_3/checkpoints/plcs-epoch=57.ckpt` は固定pinと一致し、
そこから別パスへコピーした940592845 byteの復元候補もdual SHA照合が成功した。
同じ失敗パスへの無条件再試行ではなく、この採用済み保存元を使う明示的な復旧へ進む。

### 既存実験との比較

部分監査3では公開後に不一致を発見したが、この試行ではguardが開始前に拒否した。
PLCS/BLCSへの例外拡大や固定pin変更はしていない。

### 次に有効な実験

検証済みの別パス復元候補を明示指定し、新しいrun出力先で3clipを再生成する。
実行中の全体生成とcheckpoint差し替えを競合させず、旧ファイルを保存してから復元する。
