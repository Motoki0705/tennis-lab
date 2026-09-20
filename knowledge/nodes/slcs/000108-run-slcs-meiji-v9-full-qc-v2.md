---
task: slcs
sequence: 108
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-v9-full-qc-v2
type: run
title: 'Meiji v9全件監査成功: 56clip・29148frame、欠落/不正0'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config: {data: slcs/meiji_rgb_v9, allow_incomplete: false, device: cpu}
metrics:
  completed_clips: 56
  missing_clips: 0
  error_clips: 0
  excluded_clips: 1
  total_frames: 29148
  positive_weight_ball_frames: 25520
  positive_weight_player_slot_frames: 57316
  raw_ball_reprojection_mean_px: 14.75492000715478
  refined_ball_reprojection_mean_px: 6.107228436448117
  raw_pose_all_joint_reprojection_mean_px: 27.278577832992433
  refined_pose_all_joint_reprojection_mean_px: 16.548231673572133
  refined_supported_player_speed_max_mps: 11.972556114196777
  refined_supported_ball_speed_max_mps: 63.941856384277344
repro:
  commit: a4655661
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tennis_scene.scripts.report_slcs_dataset_quality
    output_dir=tennis_scene/analyze/meiji_rgb_v9_quality/s42-full-002
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-full-qc-v2
  output_dir: outputs/tennis_scene/analyze/meiji_rgb_v9_quality/s42-full-002
parents: [run-slcs-meiji-v9-repair-batch-v3]
relations: [{to: run-slcs-meiji-v9-full-qc-v1, rel: supersedes}]
tags: [slcs, meiji, quality, provenance, cpu]
---

## 考察 / Findings

### 要約

対象56clipの教師・実媒体・DINO特徴・生成記録が全件監査を通過した。欠落・不正・設定混在は0。
外注ボールの多視点支持が足りない1clipの理由付き除外は維持する。broadcastとの統合へ進める状態になった。

### アーキテクチャ詳細

先行監査と同じ品質条件で、修復済み9clipを含む本体 `slcs/meiji_rgb_v9` を検査した。
raw/refinedには同じ最終正weight frame maskを適用。速度は隣接両端が支持される区間のみで、欠損を跨がない。
windowの選択・重複回数はこのframe監査では再現していない。

### メトリクスの解釈

29148frame中、ballの正weightは25520frame。playerの2slot合計は57316/58296frame。
サンプル数加重のball再投影平均は14.7549→6.10723px、全関節の平均は27.2786→16.5482px。
再投影は補正にも使った2D観測との整合であり、実測3D精度の指標ではない。新規学習はなく曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

幾何補正と品質maskで平均整合性は改善したが、正weight frameでも非参加カメラなどの大きな残差は残る。
refined ball残差最大485.69px、全関節最大2712.41pxを隠していない。品質合格を全カメラ・全関節の高精度と
解釈しない。DINO/ViTPoseの明示的なdigest例外によるbytes認証の限界もレポートに残す。

### 既存実験との比較

初回全件監査の残存3件を修復して全56件が成功した。最もball支持率が低いclipは54.10%で、
その長い欠損も維持されている。未修復版の不一致を期待hashへ書き換えて通したものではない。

### 次に有効な実験

broadcastの採用済み5clipと固定splitで統合し、実際の学習window数とCPU dry-runを確認する。
予定済みの60epoch・ball smoothness単独比較と、全体版によるSLCS学習・held-out評価を進める。
