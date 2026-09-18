---
id: run-slcs-real-infer-pilot-v1
type: run
title: 'Meiji実動画: 既存PLCS/BLCS教師の整合性評価'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: existing camera_view_v2 PLCS + BLCS
  loss: inference only
  data: Meiji video_000/clip_000, development
metrics:
  observable/ball_reprojection_median_px: 82.2333659627376
  observable/player_reprojection_median_px: 60.28365094650789
  observable/foot_root_xy_median_m: 4.285474303868562
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m src.tennis_scene.scripts.build_slcs_dataset stage=infer clip_ids=[video_000/clip_000]
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
artifacts:
  run_dir: knowledge/runs/run-slcs-real-infer-pilot-v1
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v1/s42-002/video_000/clip_000
parents:
- run-slcs-real-court-probe-v3
relations: []
tags:
- slcs
- real-rgb
---

## 考察 / Findings

### 要約
既存教師は実動画で大きな再投影誤差を示した。SLCSの教師としてそのまま採用せず、追加学習と観測に基づく補正を行う。

### アーキテクチャ詳細
DINOの検出をコート半面ごとに選択し、ViTPoseで2選手を追跡。Courtはoutputsのdepth3 epoch17。ボールはoutsource注釈。PLCS/BLCSはcamera_view_v2、128フレーム、stride2、重複窓を統合。

### メトリクスの解釈
ボール再投影中央値82.233px、人物関節60.284px、足元とrootのXY差4.285m。近似カメラによる観測整合性であり独立3D正解誤差ではない。開発用clip000のみ。

### アーキテクチャ⇄メトリクスの因果考察
仮説: 合成→実動画の分布差に加え、足元・カメラ幾何の弱い利用が位置の縮みを生む。Court自体も未測定の近似校正なので系統誤差は残る。

### 既存実験との比較
Court選択は親runの9フレーム集約を使用。最初の人物top2面積選択で生じた同一半面の重複を解消した後の評価。

### 次に有効な実験
足元priorを持つPLCSを60 epoch学習し、品質検査した三角測量データでBLCSを60 epoch fine-tuneする。
