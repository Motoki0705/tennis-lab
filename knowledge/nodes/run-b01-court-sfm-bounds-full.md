---
id: run-b01-court-sfm-bounds-full
type: run
title: B01：SfM外周制約＋広角化、全2,101視点のGPU検証
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: done
config:
  scene: B01
  dataset_selector: sfm_bounded
  hfov_degrees:
  - 45
  - 90
  boundary_margin_m: 0.5
  look_at_jitter_radius_m: 1.0
metrics:
  proposals: 2288
  rendered_frames: 2101
  accepted_frames: 2101
  accepted_fraction: 0.9182692307692307
  trajectory_groups: 47
  outside_sfm_hull_count: 0
  minimum_boundary_clearance_m: 0.6938413116579625
repro:
  commit: 96b98a17ee067d9956cafcfb65448d527910349b
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python outputs/b01-court-bounds-wide/full_render_check.py
artifacts:
  run_dir: knowledge/runs/run-b01-court-sfm-bounds-full
  output_dir: /home/kamimura/projects/tennis-lab-worktrees/court-sfm-bounded/outputs/b01-court-bounds-wide
parents:
- run-b01-court-sfm-bounds
relations: []
tags:
- court
- synthetic-data
- sfm
- camera-sampling
---

## 考察 / Findings

### 要約
B01の全2,288候補を検査し、描画前に187視点を除外。残る2,101視点を全てGPUで描画し、2,101視点が可視性条件を通過した（採用率91.83%）。47軌道すべてに採用視点があり、full/near_full/partialと14KP全クラスを確認した。

### アーキテクチャ詳細
前実験と同じSfM凸包内の円・楕円、中心周辺の乱数注視点を使用し、HFOVを45〜90度へ変更した。外周外0、最小余白0.694m。円20・楕円27、複合中心と各3面の中心、高さ1.5/2.25/3mと上下変動を保持。

### メトリクスの解釈
全視点を描画することで2,000枚以上・採用率90%以上・全軌道採用・coverage三種・14KP可視性を確認。これは画質の完全保証や下流モデル精度の検証ではない。実験はpublic nht-renderを256視点単位で9回呼び出し、PNGと集計を保持した。検証後の配列のみ削除した。

### アーキテクチャ⇄メトリクスの因果考察
画角を広げたことで、内側の小半径カメラからも必要なコート点が画面に入り、前実験の採用率不足が解消した。RGBの目視では従来の外周外視点で見られた大きな崩れが減少した一方、ネット・遠景にはartifactが残る。

### 既存実験との比較
前実験の幾何採用2,015/2,288（88.1%）から2,101/2,288（91.8%）へ改善。カメラ位置は同じ。元のv3計画は4,800候補中3,769が外周外だった。

### 次に有効な実験
実動画のpose分布・下流検出精度を用いて半径倍率、注視点ずれ、HFOVの分布を調整する。凸包は未観測の凹部を埋め、高さも制約していないため、別sceneでの品質確認が必要。

B01入力は `human_confirmed_court_geometry_v1` を明示的に解釈した座標検証用adapterで読み込んだ。canonical alignment ownerの自動fit/holdout受理を主張せず、元B01やcanonical dataset ownerには書き込んでいない。通常パイプラインの公開・性能ゲートまで実行した結果ではない。
