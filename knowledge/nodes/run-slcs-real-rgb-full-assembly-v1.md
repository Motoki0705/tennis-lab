---
id: run-slcs-real-rgb-full-assembly-v1
type: run
title: '実RGB全体版: Meiji 56 + broadcast 5を固定splitで統合'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/real_rgb_v1
  sources: [slcs/meiji_rgb_v9, slcs/broadcast_rgb_v4]
  split_unit: source recording / curated broadcast venue
  seed: 42
metrics:
  clips: 61
  recordings_or_venues: 7
  clip_camera_pairs: 173
  frames: 32973
  train_clips: 15
  val_clips: 22
  test_clips: 24
repro:
  commit: a4655661
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tennis_scene.scripts.assemble_slcs_dataset
    output_dir=tennis_scene/generate/assemble_real_rgb/s42-takeover-001
artifacts:
  run_dir: knowledge/runs/run-slcs-real-rgb-full-assembly-v1
  output_dir: outputs/tennis_scene/generate/assemble_real_rgb/s42-takeover-001
parents: [run-slcs-meiji-v9-full-qc-v2, run-slcs-plcs-broadcast-e60-v2, run-slcs-blcs-broadcast-e60-v1]
tags: [slcs, real-rgb, meiji, broadcast, dataset, split, cpu]
---

## 考察 / Findings

### 要約

全件監査済みMeiji 56clipと採用済みbroadcast 5clipを `slcs/real_rgb_v1` へatomicに公開した。
全61clip・173 clip-camera・32973frameを含み、従来pilotに無かったMeiji test収録を確保した。

### アーキテクチャ詳細

既定のassemble recipeに従い、全annotationとDINO特徴の完成・整合性を確認した上で統合。
媒体と配列はimmutableなhardlink、派生manifestとmarkerは新dataset用に分離される。
入力digest・派生manifest digest・split digestはassembly.jsonに保存されている。

### メトリクスの解釈

train 15clip/39 clip-camera/11456frame、val 22/64/13174、test 24/70/8343。
frameはclip内の時刻数で、カメラ数を掛けたサンプル数ではない。実際の学習窓数はquality filterとstrideに依存する。
本runはデータ統合であり、学習精度や収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

Meijiはvideo_000/001/002をtrain/val/testに分離。broadcastはShanghai・Washingtonをtrain、
indoor hardをval、Eastbourneをtestへ割り当てた。clipをランダムに混ぜて同じ収録がsplitを跨ぐことを避ける。
ただし3D教師は擬似ラベルであり、held-out教師との一致も独立実測3D精度ではない。

### 既存実験との比較

7clipの旧pilotから全体61clipへ進んだ。旧pilotのモデル評価を、この新しい教師版の評価と混同しない。
データ統合時に除外条件・教師weight・モデル重みを変更していない。

### 次に有効な実験

production datasetによる実window数と1バッチCPU smokeを確認する。共有queueで予定済み単独loss比較を行い、
その結果を踏まえた60epoch全体学習とvalidation選定checkpointのheld-out評価へ進む。
