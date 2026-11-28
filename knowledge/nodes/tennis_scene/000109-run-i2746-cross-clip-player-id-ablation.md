---
id: run-i2746-cross-clip-player-id-ablation
type: run
task: tennis_scene
sequence: 109
recorded_at: '2026-11-29'
title: 明治4カメラ全69クリップのクリップ間プレイヤーID連結アブレーション（E0-E7・Axis G・Axis P）
issue: 2746
provider: gemini
date: '2026-11-29'
status: done
summary_impact: updated
config:
  dataset: meiji_4cam
  video: video_000
  benchmark_version: multiview-v4
  benchmark_manifest_sha256: 8e012944b1407d696a1ab1d325ae3c6b60cf26f071752510ea2fc688b031893c
  clips_total: 69
  warmup_clip: clip_002
  dev_clips: 34
  test_clips: 34
  crops_per_camera: 4
  total_crops: 4416
  base_encoder: clipreid_vitb16_market1501
  followup_axes: [G_multiclip_clustering, P_kpr_2d_prompted]
  axis_g_max_nodes_per_window: 28
  axis_g_time_limit_s: 30.0
metrics:
  e1_test_pair_f1: 0.844828
  e2_test_pair_f1: 0.800283
  e2_test_false_merge_rate: 0.162963
  e3_test_pair_f1: 0.800283
  e4_row_test_pair_f1: 0.837607
  e4_row_test_false_merge_rate: 0.02
  e5_smpl_beta_dev_auroc: 0.6913
  e5_test_pair_f1: 0.837607
  e6_short_gap_eligible_rate: 0.0
  e7_row_test_pair_f1: 0.854701
  e4_swap_dev_pair_f1: 1.0
  e4_swap_test_pair_f1: 0.970149
  e4_swap_test_false_merge_rate: 0.029851
  e4_swap_test_clip_exact: 0.970588
  e7_swap_dev_pair_f1: 1.0
  e7_swap_test_pair_f1: 1.0
  e7_swap_test_false_merge_rate: 0.0
  e7_swap_test_clip_exact: 1.0
  e4_swap_strict_open_set_unreg_fa_rate: 0.0
  g_greedy_test_pair_f1: 0.854701
  g_milp_test_pair_f1: 0.854701
  g_milp_timeouts: 0
  p_unprompted_swap_test_pair_f1: 0.866154
  p_prompted_swap_test_pair_f1: 0.969711
  p_prompted_swap_test_false_merge_rate: 0.0
  clip_reid_cache_wall_sec: 328.161
  kpr_cache_wall_sec: 405.094
  e7_swap_test_wall_ms: 84.503
repro:
  commit: 755941017c00360bb79f4e8e047faa0c085110b1
  branch: research/i2746-cross-clip-player-id
  queue_job: 0-8204117508277649875_2026-11-28T16-14-51-722350124Z_654123_i2746-extract-clip-reid-cache-v3.job
  command: bash knowledge/runs/run-i2746-cross-clip-player-id-ablation/repro.sh
artifacts:
  run_dir: knowledge/runs/run-i2746-cross-clip-player-id-ablation
  report: knowledge/runs/run-i2746-cross-clip-player-id-ablation/README.md
  gt: knowledge/runs/run-i2746-cross-clip-player-id-ablation/gt_cross_clip.json
  ablation_summary: knowledge/runs/run-i2746-cross-clip-player-id-ablation/ablation_summary.csv
  metrics: knowledge/runs/run-i2746-cross-clip-player-id-ablation/metrics.json
  links_sample: knowledge/runs/run-i2746-cross-clip-player-id-ablation/player_identity_links_sample.json
  comparison_figure: knowledge/runs/run-i2746-cross-clip-player-id-ablation/ablation_comparison.png
  montage_figure: knowledge/runs/run-i2746-cross-clip-player-id-ablation/montage_cross_clip_consistency.png
parents:
- run-i2733-meiji-event-labels-v4
relations: []
papers: []
tags:
- meiji_4cam
- cross_clip_reid
- player_association
- ablation
- kpr
---

## 考察 / Findings

### 要約

`meiji_4cam/video_000`（`multiview-v4` 全69クリップ・276ノード）に対し、Re-ID 非依存の同期4視点目視による独立 cross-clip GT（純粋275ノード＋上流混合1ノード `clip_062/p0`）と、`clip_002` を固定ウォームアップとした時系列 dev/test 分割（各34クリップ・元フレーム重複0件）を構築し、既存 `SceneResult` を改変しない後段 `player_identity_links` 契約の検証と E0〜E7・Axis G・Axis P の統制比較を完了した。

- **実データのドメインシフト**: 7回のエンド交代（32/69クリップ反転）に加え、`clip_019` で前衛2名が上着を脱ぎ、test 全34クリップが「4名中3名が黒系トップス」となる難条件だった。
- **独立最近傍の破綻と Hungarian 1-to-1＋Swap-Margin の決定的寄与**: 排他制約なしの `E2/E3` は test で10件の同時出現重複違反（`FMR=0.1630`、Pair F1 `0.8003`）を起こした。クリップ内1対1割当と割当交換マージンを用いた `E4_swap_margin` は固定アンカーのみで dev `1.0000`・test `0.9701`（`ClipEx=0.9706`）、高確信因果ギャラリー更新を加えた `E7_swap_margin` は誤更新0/28件で dev・test とも Pair F1 `1.0000`・`FMR=0.0000`・`ClipEx=1.0000`（既存 CLIP-ReID キャッシュ再利用、CPU 84.5 ms）を達成した。
- **無効／未検証要素の切り分け**: SMPL `beta`（E5）は有効入力率100%だが dev AUROC `0.6913` で E4 比改善ゼロ。短ギャップ 3D 連続性（E6）は全68隣接ペアが `7.34–41.35s`（`<=3.0s` は `0/68`）のため本データでは未検証。
- **条件付き追加軸（G / P）**: Axis G（最大28ノード/窓、timeout 0/14）では `G_greedy` と `G_milp` が完全一致（test F1 `0.8547`, `FMR=0.0000`）で MILP の優位はなかった。Axis P（KPR 6-part）では COCO17 正＋他者負の 2D プロンプトが `unprompted` 比で test F1 を `0.8662 → 0.9697`（+0.1035、重なり層 +0.1006、`FMR=0.0000`）へ押し上げ、更新なし・重なり多発条件での有力代替であることを示した。
- **開集合の安全性**: leave-one-out 未登録選手テストでは `E4_swap_margin`（`min_score=0.70`）の誤受入率が test `24.4%` に上がる一方、`min_score=0.84, min_margin=0.02` の厳格ゲートは dev/test とも未登録誤受入率 `0.0%`（`0/136`, `0/135`）を達成した。

### 限界と今後の課題

- 本検証は単一試合（ダブルス4名、固定4カメラ）の結果であり、短ギャップ連続映像（E6）、別試合・別日の完全着替え、上流トラック内途中スイッチ（`clip_062/p0`）の分割は未解決である。学習ランではないため TensorBoard 曲線は該当しない。
- 次段階として、`SceneResult` を不変に保ったまま `player_identity_links` v1 サイドカーと永続 `IdentityRegistry` を出力する本番後段モジュール（既定 `E7_swap_margin`、開集合厳格ゲートおよび KPR 2D-prompted オプション付き）を別 Issue で実装する。詳細数値・層別表・図表・再現手順は [詳細レポート](../../runs/run-i2746-cross-clip-player-id-ablation/README.md) を参照。
