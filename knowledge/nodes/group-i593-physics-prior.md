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

### 決定（PR #605 と統合）
物理prior + `run.init_weights` は**基盤機能として採用**（default off の任意ノブ）。ただし現時点で **default ckpt は置換しない**（`ckpt/{blcs,plcs}/last.ckpt` は v3/v2 baseline のまま）。理由: (1) in-dist が回帰（BLCS 1.845→1.947m、**PLCS 0.345→0.471m と +37%** は default 差し替えには過大）、(2) gravity curvature がまだ baseline より平坦（median Δ²z -0.0097→-0.0054）、(3) 速度外れ値(>50m/s)は detector 欠損由来 teleport で prior 未対処。実クリップは明確改善だが「厳密な上位互換」ではないため、feature として温存し次段で clean win を狙う。最終評価出力は `outputs/tennis_scene/tennis_clip_physics_final/`（別セッション作成、canonical）。

（注: 本 session は一度 ftC/ft を deploy したが、PR #605 の結論に合わせ baseline へ revert 済み。）

### 次に有効な実験（clean winへ）
(1) **gravity term を free-flight aware に**（bounce/occlusion/補間フレームに弾道拘束をかけない）→ in-dist回帰とcurvature平坦を同時解消（PR #605 の第一推奨）。(2) PLCSは一律training lossより **confidence-aware post-filter / masked temporal regularizer**（極端max-speed外れ値が残るため）。(3) 振幅圧縮を根絶する定式化（scale不変jerk / 振幅保存項）。(4) best-ckpt評価（現状test=last-epoch、fine-tuneはdriftしうる）。(5) 物理prior weightのwarmupでスクラッチ学習を救済しfine-tune依存を外す。(6) ボール>50m/s外れはdetector欠損teleportが主 → BLCS前段の補間/棄却。
