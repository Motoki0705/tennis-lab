---
id: group-i593-physics-prior
type: group
title: 物理prior campaign + 実クリップA/B (#593)
issue: 593
members:
- run-mono3d-blcs-bcast-v4-physics
- run-mono3d-blcs-fta-s03g015
- run-mono3d-blcs-ftb-s06g03
- run-mono3d-blcs-ftc-axis-s04g03
- run-mono3d-plcs-ft-s15-safe
parents:
- run-mono3d-blcs-bcast-v3-simfix
- run-mono3d-plcs-bcast-v2-simfix
provider: claude
session: dd9d56e0-5f2d-4d9d-a8c7-c80704a806b3
date: '2026-07-06'
tags:
- monocular
- broadcast
- sim-to-real
- physics-prior
---

## まとめ

### 動機
[[run-mono3d-blcs-bcast-v3-simfix]] / [[run-mono3d-plcs-bcast-v2-simfix]] の予測は、合成testで**加速度がGTの20〜70倍ジッター**していた（BLCS 自由飛行 a_z std≈73 vs GT≈-9.6、PLCS 位置 accel std がGTの20〜52倍）。教師あり位置lossは低周波軌道を合わせるが高周波の非物理ジッターを拘束しない。実クリップのボール軌道ガタつき・遠側プレーヤー速度スパイクの根本原因。

### 対策
共有ユーティリティ `src/utils/losses/temporal.py` を新設: (1) **jerk(3階差分)平滑化prior**（等加速度=弾道/滑らかな移動を許し重力/drag/走行加速度をバイアスしない）、(2) **弾道重力prior**（正規化高さの2階差分を -g·dt²/scale に固定 → 再投影と結合して単眼depthスケールを拘束）。BLCSは両方、PLCSは位置jerkのみ（yawはGT同等なので触らない）。

### 学習の教訓（重要）
**スクラッチ学習は失敗** ([[run-mono3d-blcs-bcast-v4-physics]]): 初期に物理lossが支配し、ジッター最小化の安易解として**振幅を圧縮**（std比 X 0.71 まで縮小）、in-dist位置 1.85→2.44m 悪化。corr と jerk は改善していたので「prior自体は機能、from-scratch動学が問題」と判断。→ **`run.init_weights`（重みのみfine-tune）を追加**し、正しい振幅を持つ baseline から緩くrefine する方式に変更。fine-tune bracket（ftA/ftB/ftC）+ 軸重み [1,1,0]（z jerkを切り重力に任せる, ftC）を比較。

### 実クリップA/Bが決定打（load_path再利用, 上流stage固定）
in-dist metricsでは fine-tune 群は誰も baseline を超えない（prior はジッター削減と引き換えに合成L2をわずかに落とす）。しかし**実映像=本タスクの評価軸**では明確に改善:

| 指標(実clip) | v3/v2 baseline | ftC BLCS + ft PLCS | 
|---|---|---|
| ボール jerk 平均 | 0.280 | **0.106**(-62%) |
| 重力整合 a_z∈[-15,-4] 割合 | 0.11 | **0.30**(≈3x) |
| ボール高さmax | 3.22m | 4.17m(圧縮せず現実的) |
| \|Y\|>15m 外れ | 10 | 7 |
| P1(遠側)速度スパイク>10m/s | 20フレーム | **9フレーム**(-55%) |
| P1 jerk平均 | 0.083 | 0.062 |

可視化mp4でも滑らかな連続弾道・両者正しいベースラインを確認。

### 決定
**ftC BLCS (axis[1,1,0], s0.4/g0.3) と ft PLCS (position_smoothness 1.5) を採用しデプロイ** (`ckpt/{blcs,plcs}/last.ckpt`)。最終成果物 `outputs/tennis_scene/tennis_clip_full_v4phys/`。in-dist合成L2の微増(+5%)より実映像の物理妥当性(ジッター・重力整合・スパイク)を優先。

### 次に有効な実験
(1) 振幅圧縮を根絶する定式化（scale不変jerk、または振幅保存項）で in-dist も超える。(2) best-ckpt評価（現状test=last-epoch重み、fine-tuneはdriftしうる）。(3) 物理prior weightのwarmupでスクラッチ学習を救済し、fine-tune依存を外す。(4) ボール速度の外れ値(>50m/s)はdetector欠損由来のteleportが主 → BLCS前段の補間/棄却で対処。
