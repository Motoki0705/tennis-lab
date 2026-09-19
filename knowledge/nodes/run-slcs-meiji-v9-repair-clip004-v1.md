---
id: run-slcs-meiji-v9-repair-clip004-v1
type: run
title: 'Meiji v9残り1clip修復と後半3clipの監査・差し替え'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/meiji_rgb_v9_repair_batch_v2
  clip: video_002/clip_004
  stage: infer
  device: cpu
metrics:
  generated_clips: 1
  audited_clips: 3
  audit_error_clips: 0
  three_clip_raw_ball_reprojection_mean_px: 17.09714041101371
  three_clip_refined_ball_reprojection_mean_px: 7.07325220265147
  three_clip_supported_player_speed_max_mps: 11.572952270507812
  three_clip_supported_ball_speed_max_mps: 63.50231170654297
repro:
  commit: 25f4beafe1156371d00b5bf70ae1847b31ab5eb5
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tennis_scene.scripts.build_slcs_dataset device=cpu stage=infer
    'dataset_clip_ids=[video_001/clip_020,video_002/clip_004,video_002/clip_015]'
    'clip_ids=[video_002/clip_004]'
    dataset_output_directory=slcs/meiji_rgb_v9_repair_batch_v2
    output_dir=tennis_scene/generate/meiji_rgb_v9_repair_batch/s42-004
    plcs_checkpoint=plcs/real-rgb-meiji-foot-e60-v1.restore-20260919.ckpt
    blcs_checkpoint=blcs/real-rgb-meiji-e60-v1.restore-20260919.ckpt
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-repair-clip004-v1
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v9_repair_batch/s42-004
parents: [run-slcs-meiji-v9-repair-batch-v2]
relations: [{to: run-slcs-meiji-v9-repair-batch-v1, rel: confirms}]
tags: [slcs, meiji, quality, provenance, cpu, repair]
---

## 考察 / Findings

### 要約

未生成の `video_002/clip_004` が公開前guardを通過し、先行2clipと合わせて媒体・DINO特徴を含む
subset監査が成功した。3clipの旧成果物を削除せず退避し、監査済み成果物を本体へ差し替えた。

### アーキテクチャ詳細

採用済み学習出力から復元したPLCS/BLCSを別パスで明示指定し、固定pin・観測・補正条件は維持。
生成済み2clipは再実行せず、2つの生成runを監査へ明示した。DINO特徴は同じ媒体の本体cacheをコピー。
本体の各生成ディレクトリにCPU recipeを残し、annotation全ファイルのcopy元との一致を確認した。

### メトリクスの解釈

3clipの同じ最終正weight maskで、サンプル数加重ball再投影平均は17.0971→7.07325px。
支持区間の最大速度はplayer 11.573m/s、ball 63.502m/s。一方、非支持区間を含む全軌道には
140/205m/sのjumpが残る。`video_001/clip_020` のball支持率は54.10%、最長非支持区間420frame。
擬似教師の整合性であり独立3D精度ではない。新規学習はなく、曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

固定pinとraw/producerの一致を確認した成果物だけを採用した。重みの読み取り不一致の根本原因は
断定せず、例外範囲も広げていない。支持率が十分でも全時刻に有効な教師があるとは限らない。

### 既存実験との比較

部分監査3で検出された3件を解消し、先行修復と合わせて6件を退避付きで差し替えた。
3clip subset監査の成功であり、まだ全体56clipの成功を意味しない。

### 次に有効な実験

全体生成後に本体を監査する。重みを復元して1コマンド生成の再現性確認と固定split統合へ進む。
