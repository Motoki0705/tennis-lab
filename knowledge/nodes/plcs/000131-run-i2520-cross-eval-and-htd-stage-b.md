---
id: run-i2520-cross-eval-and-htd-stage-b
type: run
task: plcs
sequence: 131
recorded_at: '2026-11-25'
title: 単眼GVHMR・HTD-Refine Stage A/Bの合成3splitおよびmeiji_4cam 6クリップ横断評価
issue: 2520
provider: gemini
session: ec99afa3-0ddf-4af9-b92f-9e3912cd1273
date: '2026-11-25'
status: done
summary_impact: updated
config:
  eval_splits_synthetic:
  - single_object
  - single_object_broadcast_eval
  - single_object_mixed
  eval_real_benchmark: meiji_4cam/multiview-player-evaluation-v1
  eval_real_manifest_sha256: 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580
  htd_stage_b_steps: 25
  eval_device: cpu
  cpu_threads: 4
metrics:
  b0_mixed_position_error_m: 0.5146
  b0_mixed_angular_error_deg: 49.79
  b0_mixed_pa_mpjpe_m: 0.0918
  m0_mixed_position_error_m: 0.8404
  m0_mixed_pa_mpjpe_m: 0.0945
  m0_meiji_vs_dlt_mpjpe_m: 4.1306
  m0_meiji_jerk_mps3: 198069.2
  m1_mixed_position_error_m: 0.9404
  m1_mixed_pa_mpjpe_m: 0.0915
  m1_meiji_lovo_reproj_px: 612.57
  m1_meiji_vs_dlt_mpjpe_m: 3.7178
  m1_meiji_jerk_mps3: 163335.5
  m2_sv_mixed_position_error_m: 0.9495
  m2_sv_mixed_angular_error_deg: 13.97
  m2_sv_mixed_pa_mpjpe_m: 0.0699
  m2_sv_mixed_mpjae_mps2: 64.86
  m2_sv_mixed_jerk_mps3: 7209.4
  m2_sv_meiji_same_view_reproj_median_px: 1.27
  m2_sv_meiji_vs_dlt_mpjpe_m: 3.7016
  m2_sv_meiji_vs_dlt_pa_mpjpe_median_m: 0.1111
  m2_sv_meiji_jerk_mps3: 20311.1
  m2_sv_meiji_dlt_incomplete_recovery_rate: 1.0
  m2_sv_meiji_latency_ms_per_frame: 1.5405
  m2_mv_meiji_fused_vs_dlt_mpjpe_m: 1.1656
  m2_mv_meiji_fused_vs_dlt_mpjpe_median_m: 1.0414
  m2_mv_meiji_fused_all_cam_reproj_median_px: 19.23
  m2_mv_meiji_fused_jerk_mps3: 48336.6
repro:
  commit: 59faa32a28d2f54e6492abdba2429846fa0397e3
  branch: main
  command: bash knowledge/runs/run-i2520-cross-eval-and-htd-stage-b/repro.sh
artifacts:
  run_dir: knowledge/runs/run-i2520-cross-eval-and-htd-stage-b
parents:
- run-i2520-b0-multiview-axial-e100-s42
- run-i2520-v2-m0-mono-gvhmr-base-e100-s42
- run-i2520-v2-m1-mono-gvhmr-htd-e100-s42
- run-i2520-v2-a1-no-cal-noise-e100-s42
- run-i2520-v2-a2-no-bbox-aug-e100-s42
- run-i2520-v2-a3-world3d-only-e100-s42
relations:
- to: run-i2520-b0-multiview-axial-e100-s42
  rel: evaluates
  note: 合成3 splitで多視点参照基準B0を同一指標で再評価
- to: run-i2520-v2-m0-mono-gvhmr-base-e100-s42
  rel: evaluates
  note: 合成3 split＋実動画meiji_4cam 6クリップでM0を評価
- to: run-i2520-v2-m1-mono-gvhmr-htd-e100-s42
  rel: evaluates
  note: M1単体およびStage B HTD-Refine単視点(M2-SV)/多視点(M2-MV)を合成3 split＋meiji_4camで評価
- to: run-i2520-v2-a1-no-cal-noise-e100-s42
  rel: evaluates
  note: 校正摂動なしアブレーションA1を合成3 split＋meiji_4camで評価
- to: run-i2520-v2-a2-no-bbox-aug-e100-s42
  rel: evaluates
  note: bbox増強なしアブレーションA2を合成3 split＋meiji_4camで評価
- to: run-i2520-v2-a3-world3d-only-e100-s42
  rel: evaluates
  note: World-3DのみアブレーションA3を合成3 split＋meiji_4camで評価
papers:
- paper-2024-gvhmr
- paper-2026-htd-refine
tags:
- plcs
- monocular-gvhmr
- htd-refine
- stage-b-refiner
- cross-evaluation
- meiji-4cam
- i2475
- i2520
---

## 考察 / Findings

### 評価プロトコル
- `knowledge/runs/run-i2520-cross-eval-and-htd-stage-b/eval_i2520_plcs.py` を用い、全8構成（多視点参照 `B0`、単眼基本 `M0`、単眼+Stage A `M1`、`M1`+Stage B単視点 `M2-SV`、`M1`+Stage B多視点 `M2-MV`、およびアブレーション `A1`/`A2`/`A3`）を同一CPU条件（4スレッド、長系列は2048フレーム・128オーバーラップ分割推論）で評価した。
- **合成データ 3 split（各100テストシーン）**:
  - `single_object`（fixed固定視点, 66,927フレーム）
  - `single_object_broadcast_eval`（broadcast俯瞰・望遠視点, 70,613フレーム）
  - `single_object_mixed`（混在, 65,346フレーム）
- **実動画ベンチマーク**: `meiji_4cam/multiview-player-evaluation-v1`（ピン留め済み6クリップ、`manifest_sha256 = 3dc49651...`、50,297選手視点フレーム、4カメラDLT欠損 392フレーム）。単眼入力は校正済みカメラ内外パラメータ＋DINO/StrongSORT++追跡bbox＋ViTPose COCO17であり、HMR2・SMPL-Xは一切使用しない。

### 観測1: 合成3 split 横断結果（`single_object_mixed` / `fixed` / `broadcast`）

| 構成 | Split | Pos Err (m) | Ang Err (°) | World MPJPE (m) | PA-MPJPE (m) | MPJVE (m/s) | MPJAE (m/s²) | Jerk (m/s³) | Bone Std (mm) | CPU (ms/f) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** (多視点4cam) | fixed / bc / mixed | 0.361 / 0.558 / 0.515 | 45.1 / 57.4 / 49.8 | 0.416 / 0.601 / 0.529 | 0.116 / 0.093 / 0.092 | 0.78 / 0.86 / 0.99 | 32.2 / 24.7 / 39.9 | 4,390 / 2,415 / 5,799 | 4.68 / 4.49 / 4.83 | 3.22 / 1.50 / 3.10 |
| **M0** (単眼Base) | fixed / bc / mixed | 1.454 / 1.910 / 0.840 | 34.1 / 15.5 / 17.9 | 1.401 / 1.896 / 0.823 | 0.123 / 0.100 / 0.095 | 2.21 / 4.86 / 2.56 | 88.9 / 169.9 / 119.3 | 14,398 / 27,428 / 21,094 | 16.5 / 18.8 / 18.4 | 0.92 / 1.01 / 0.94 |
| **M1** (+Stage A) | fixed / bc / mixed | 1.591 / 2.035 / 0.940 | 27.3 / 14.9 / 17.8 | 1.555 / 2.016 / 0.920 | 0.122 / 0.096 / 0.092 | 2.26 / 4.89 / 2.52 | 90.5 / 170.4 / 118.1 | 14,394 / 27,334 / 20,472 | 17.8 / 18.5 / 18.4 | 1.16 / 1.17 / 1.11 |
| **M2-SV** (+Stage B) | fixed / bc / mixed | 1.543 / 2.016 / 0.949 | 26.9 / 14.9 / **14.0** | 1.538 / 1.990 / 0.907 | **0.100 / 0.088 / 0.070** | 2.03 / 4.50 / 2.27 | **51.7 / 100.4 / 64.9** | **5,179 / 10,658 / 7,209** | **2.56 / 3.41 / 8.47** | 1.18 / 1.40 / 1.39 |
| **A1** (校正雑音OFF) | fixed / bc / mixed | 1.894 / 2.617 / 1.028 | 26.5 / 15.2 / 20.2 | 1.850 / 2.597 / 1.009 | 0.122 / 0.100 / 0.095 | 2.42 / 5.67 / 2.72 | 94.2 / 192.2 / 128.3 | 15,082 / 30,119 / 22,502 | 17.6 / 18.9 / 18.4 | 1.24 / 1.21 / 1.25 |
| **A2** (BBox増強OFF) | fixed / bc / mixed | 1.776 / 2.130 / 0.949 | 36.0 / 13.2 / 18.2 | 1.735 / 2.125 / 0.926 | 0.119 / 0.092 / 0.088 | 2.88 / 5.65 / 2.93 | 116.8 / 210.2 / 156.7 | 18,342 / 33,672 / 28,333 | 18.2 / 17.0 / 18.2 | 1.09 / 1.02 / 0.86 |
| **A3** (World3Dのみ) | fixed / bc / mixed | 1.895 / 2.194 / 1.030 | **20.2 / 11.7 / 13.6** | 1.861 / 2.156 / 0.980 | 0.131 / 0.143 / 0.107 | 2.28 / 4.79 / 2.51 | 87.1 / 163.7 / 111.9 | 13,406 / 25,996 / 18,958 | 14.4 / 14.4 / 16.7 | 1.30 / 1.15 / 1.12 |

### 観測2: 実動画 `meiji_4cam` 6クリップ評価結果

| 構成 | LOVO Reproj mean/med (px) | Same-View Reproj mean/med (px) | Cross-Cam 3D (m) | vs DLT MPJPE mean/med (m) | vs DLT PA-MPJPE mean/med (m) | vs GVHMR MPJPE (m) | DLT欠損復元 | Jerk (m/s³) | Bone Std (mm) | Valid率 | CPU (ms/f) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **M0** (単眼Base) | 885.8 / 115.6 | 6.39 / 4.63 | 6.44 | 4.131 / 3.456 | 0.123 / 0.122 | 4.113 | 392/392 (100%) | 198,069 | 24.4 | 100% | 1.05 |
| **M1** (+Stage A) | 612.6 / 109.1 | 6.43 / 4.80 | 5.83 | 3.718 / 3.278 | 0.124 / 0.126 | 3.684 | 392/392 (100%) | 163,336 | 23.9 | 100% | 0.99 |
| **M2-SV** (+Stage B 単視点) | **581.6 / 108.3** | **3.13 / 1.27** | **5.81** | 3.702 / 3.268 | **0.120 / 0.111** | 3.693 | 392/392 (100%) | **20,311** (-87.6%) | 23.4 | 100% | 1.54 |
| **M2-MV** (+Stage B 多視点融合) | *(全カメラ再投影 54.2 / **19.2 px**)* | — | — | **1.166 / 1.041** (-68.6%) | 0.141 / 0.130 | **1.175** | 392/392 (100%) | 48,337 (-70.4%) | **20.4** | 100% | 1.07+0.61 |
| **A1** (校正雑音OFF) | 646.6 / 115.9 | 6.71 / 5.09 | 6.15 | 3.958 / 3.525 | 0.124 / 0.127 | 3.936 | 392/392 (100%) | 188,540 | 23.8 | 100% | 1.17 |
| **A2** (BBox増強OFF) | 684.7 / 112.8 | 6.42 / 4.72 | 5.93 | 3.832 / 3.272 | 0.122 / 0.121 | 3.798 | 392/392 (100%) | 226,325 | 24.6 | 100% | 0.93 |
| **A3** (World3Dのみ) | 705.8 / 121.6 | 5.72 / 3.92 | 6.70 | 4.263 / 3.755 | 0.123 / 0.118 | 4.234 | 392/392 (100%) | 164,078 | 20.0 | 100% | 1.34 |

### 考察（観測と仮説の分離）
1. **Stage A（M1 vs M0）**: 実動画 `meiji_4cam` において `LOVO -30.8%`、`vs DLT MPJPE -10.0%`、`jerk -17.5%` と未見カメラドメインでの一貫した改善を確認した。
2. **Stage B 単視点（M2-SV vs M1）**: 合成全3 splitで **jerk `-61〜-65%`・MPJAE `-41〜-45%`・骨長分散 `-54〜-86%`・PA-MPJPE `-9〜-24%`** を同時達成し、`mixed` の PA-MPJPE `0.070 m` および方位角 `13.97°` は4カメラ参照 `B0`（`0.092 m` / `49.79°`）を上回った。実動画でも **jerk `-87.6%`（`163k → 20.3k`）、同一視点再投影中央値 `1.27 px`、DLT比PA-MPJPE中央値 `11.1 cm`、DLT欠損392フレームの100%復元** を `1.54 ms/frame`（CPU約650 FPS）で達成した。
3. **実動画の単眼絶対深度誤差の原因と Stage B 多視点（M2-MV）**: `M2-SV` は同一視点再投影 `1.27 px`・PA-MPJPE `11.1 cm` と極めて正確である一方、未見スタジアムカメラでの `vs DLT MPJPE` が `3.70 m` 残る。これはカメラ光線方向のスケール/深度ずれが主因である（観測: 同一視点再投影は1px級なのに他視点LOVOが108px）。4カメラの2D観測を同時に最適化する `M2-MV` を適用すると `vs DLT MPJPE` が **`3.72 m → 1.17 m`（中央値 `1.04 m`, -68.6%）**・全カメラ再投影中央値 `19.2 px` まで急減し、光線深度の曖昧性が解消することを実証した。
4. **データ増強と損失のアブレーション（A1/A2/A3）**: 校正雑音OFF（A1）は全split位置誤差 `+9〜+29%` 悪化、bbox増強OFF（A2）は全splitの jerk/MPJAE `+23〜+39%` 悪化、canonical/角度損失OFF（A3）は PA-MPJPE と実動画深度（`4.26 m`）が悪化し、3要素すべての有効性を確認した。
5. **限界**: 全学習ランは単一seed（42）・単一混在データセット（`single_object_mixed`）での比較であり、`M1`（Epoch 59）・`A1`（Epoch 54）は `torsion_angle` 共線例外により100エポック未満で停止したチェックポイントを使用している。実動画の独立3D真値は存在しないため、DLTおよびGVHMR+DLTとの一致度とLOVO再投影誤差を代理指標としている。
6. `summary_impact: updated` の理由: 単眼GVHMR＋HTD-Refineの合成・実動画横断結果に基づき、`knowledge/summary.md` の PLCS 節の現在判断・未解決課題・次の実験を更新した。
