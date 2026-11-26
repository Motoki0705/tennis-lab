---
id: run-meiji-multiview-scene
type: run
task: tennis_scene
sequence: 106
recorded_at: '2026-11-24'
title: Meiji 4cam全69本の3D時間IDからの身体復元と新旧比較
issue: 2446
provider: codex
session: 01a20be1-562e-7c01-8713-400bc0986f03
date: '2026-11-24'
status: done
summary_impact: none
config:
  recipe: multiview_3d_tracking
  dataset: meiji_4cam
  players_per_side: 2
metrics:
  clips: 69
  legacy_complete_clips: 52
  multiview_complete_clips: 69
  legacy_all_placed_frames: 29878
  multiview_all_placed_frames: 41949
  source_frames: 46228
  ball_3d_frames: 45958
  multiview_batch_seconds: 4541.326202505
  multiview_rss_peak_bytes: 5149065216
  meiji3_holdout_legacy_recall: 0.853017417799809
  meiji3_holdout_multiview_recall: 0.6725180772204284
repro:
  commit: 01421ac96bbf9472258a930d42c8688c1d546ae0
  branch: codex/parent-1739-issue-2446-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: '.venv/bin/python -m src.tennis_scene.multiview_scene run --dataset ../parent-2278-issue-2363-primary/outputs/i2363/i2363-all-v1/inputs/dataset
    --progress ../parent-2278-issue-2363-primary/outputs/i2363/i2363-all-v1/all/progress/d1df39596d4f47f1896cb535191e7040/progress.json
    --tracks ../parent-1739-issue-2445-primary/outputs/i2445/integrated/meiji4 --candidates
    ../parent-1739-issue-2444-primary/outputs/i2444/refine/timing/meiji4 --output
    outputs/i2446 --players-per-side 2 '
artifacts:
  run_dir: knowledge/runs/run-meiji-multiview-scene
  log: .training_queue/logs/0-8204491283477436990_2026-11-24T08-25-16-522563009Z_745087_meiji-multiview-scene-all.log
  legacy_completion: knowledge/runs/run-meiji-multiview-scene/legacy-completion.csv
  multiview_completion: knowledge/runs/run-meiji-multiview-scene/multiview-completion.csv
  meiji3: knowledge/runs/run-meiji-multiview-scene/meiji3.json
  meiji4: knowledge/runs/run-meiji-multiview-scene/meiji4.json
  verified_scenes: knowledge/runs/run-meiji-multiview-scene/verified-scenes.json
  comparison_page: knowledge/runs/run-meiji-multiview-scene/comparison-page.json
parents:
- run-i2363-trt-player-video
- run-i2445-player-slot-integration
relations:
- to: run-i2445-reserved-tracking
  rel: compares
papers: []
tags: [meiji_4cam, meiji_3cam, gvhmr, scene_result]
---

## 考察 / Findings

新経路はmeiji_4cam全69本で各側2人のGVHMR身体、身体配置、ボール3Dを含むSceneResultを生成した。clip単位の完了は52/69から69/69に、全4人の有効配置の割合は64.63%から90.74%になった。`clip_016`は198/1,214フレームで4人が揃い、フレーム単位の欠けは残る。ボール3Dは両方式とも45,958/46,228フレームで利用できる。

Meiji 3camの予約未見3本では、新経路の射影IDF1は80.42%、選別recallは67.25%だった。旧経路は92.07%、85.30%。新経路では未使用の対応視点への再投影残差の中央値も高かった。標準pipelineの採否はユーザーの確認事項のため、現在の判断と配備を維持する。 [比較表・再現手順](../../runs/run-meiji-multiview-scene/README.md)
