---
task: slcs
sequence: 24
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-meiji-observation-mask-probe-v1
type: run
title: Meijiの広い走行範囲と画面外欠損を実クリップで検証
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: DINO + ViTPose-H
  data: video_000/clip_001,video_001/clip_001
  court_half_width_m: 6.5
  long_gap_policy: mask
  max_gap_seconds: 1.0
  pose_precision: bfloat16
metrics:
  completed_clips: 2
  masked_frames_cam2_video001_clip001: 87
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'env TENNIS_RGB_GPU=0 bash /home/kamimura/projects/tennis-lab/.claude/worktrees/slcs-real-rgb/scripts/datasets/build_real_rgb.sh
    --execute meiji stage=observe clip_ids=\[video_000/clip_001\,video_001/clip_001\]
    output_dir=tennis_scene/generate/meiji_observations/s42-003 features.enabled=false '
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-observation-mask-probe-v1
  output_dir: outputs/tennis_scene/generate/meiji_observations/s42-003
parents:
- run-slcs-vitpose-precision-v1
relations: []
tags:
- slcs
- meiji
- observation
- quality
---

## 考察 / Findings

### 要約
過去に人物抽出が失敗した2クリップを再処理し完了した。幅6.5mで横移動する選手を残し、実際に画面外へ出る87フレームでは観測confidenceを0にする。

### アーキテクチャ詳細
DINOを4フレーム間隔で検出し、Court半面内の候補からViTPoseの入力boxを選択する。短い内部box欠損だけを補間し、ViTPoseは各RGBフレームを読む。1秒超の欠損と非観測の端をpose_supported_maskで区別する。検出済みobserved_masksは別に保持する。既存のraw detector出力はhash確認のうえ再利用した。

### メトリクスの解釈
video_000/clip_001の3カメラは両選手について全フレームが観測または短い内部補間で支持された。video_001/clip_001のcam2ではローカルnear選手の185–271フレーム（87/1300）が未支持となり、confidenceが厳密に0である。他カメラは利用できる。これはカバレッジの検証で、3D実測精度ではない。

### アーキテクチャ⇄メトリクスの因果考察
5.8mでは000clip001の本来の選手が選択範囲外となり別人を拾う場面があった。6.5mの設定とRGB上への重ね合わせで該当フレームの人物を確認した。画面外の長い欠損は補間で有効観測に見せず、他視点の支持へ依存させる。

### 既存実験との比較
ViTPose BF16の精度・速度評価を前提にし、今回は人物選択と欠損の意味だけを変更した。CPUの後続教師生成で同2クリップの固定品質閾値を通過し、cam2ローカルnearがreference上のP1へ正しく対応して欠測になることも確認した。観測・教師配列のSHAと検査結果はbundleのobservation_checks.jsonを参照。

### 次に有効な実験
同じ設定でMeiji全体を生成し、失敗を明示的に記録する。SLCSで長い検出欠損とRGBのみの条件を評価し、正しい教師支持と入力欠損耐性を分けて判断する。
