---
id: run-i931-v2-dataset-meiji-dino-precompute-20260927
type: run
task: slcs
sequence: 139
recorded_at: '2026-09-27'
title: 再生成したv2 dataset（Meiji clip_000）のDINOv3 token precompute
issue: 931
provider: claude
session: abf284fe-984d-4b6c-afec-7375982b0bb0
date: '2026-09-27'
status: done
config:
  dataset: data/slcs/meiji_one_clip_scene_v2
  backbone: dinov3_vitb16
  checkpoint: third_party/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
  image_size: [256, 448]
  frame_stride: 10
metrics:
  clips_processed: 1
  clips_failed: 0
  samples_per_camera: 102
  slcs_train_windows_stride60: 48
repro:
  commit: 4f1001dca8232702bf1c549e095530f368e87664
  branch: campaign930/i931-4-imports
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.tasks.slcs.scripts.precompute_dino_tokens
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i931-v2-dataset-meiji-one-clip-20260927/dino_precompute
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/third_party/dinov3/checkpoints
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party precompute.checkpoint_path=dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
    data.dataset_root=slcs/meiji_one_clip_scene_v2
artifacts:
  run_dir: knowledge/runs/run-i931-v2-dataset-meiji-dino-precompute-20260927
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790477278322332172_3141623_i931-v2-dataset-meiji-dino-precompute-20260927.log
  dino_manifest: knowledge/runs/run-i931-v2-dataset-meiji-dino-precompute-20260927/dino_v3_annotation.json
parents:
- run-i931-v2-dataset-meiji-one-clip-20260927
relations: []
papers: []
tags:
- dataset_regeneration
- dino_precompute
---

## 要約

[run-i931-v2-dataset-meiji-one-clip-20260927](../tennis_scene/000023-run-i931-v2-dataset-meiji-one-clip-20260927.md) で再生成した v2 dataset `data/slcs/meiji_one_clip_scene_v2` に対し、DINOv3 patch token を作り直した。前の job は `paths.output_root` の末尾を `slcs` にしたため path contract で停止しており、これはその再実行である。今回は `.../dino_precompute` を output root にした。

- 結果: `processed=1 skipped_existing=0 failed=0`。`annotations/dino_v3/` に cam0〜cam2 の npz（各 102 sample、frame_stride=10、合計 186MB）と完了 marker ができた。marker は `dino_v3_annotation.json` として保存した。
- 読み戻し（CPU、本 run の後に手元で実施）: 全 video を train にする split を scratch に作り、`require_dino=True`・`on_incomplete=error` の `SLCSWindowDataset`（stride 60）で読んだ。clip は 1 件読めて skip は 0、window は 48 件、落とした window も 0 だった。batch には `dino_tokens` (12, 448, 768) が入っている。v2 annotation と DINO token を合わせて、SLCS の学習入力として使えることを確認した。

## 限界

1 clip だけの dataset であり、学習・評価はしていない。dataset の split file（`splits.json`）は作っていない。
