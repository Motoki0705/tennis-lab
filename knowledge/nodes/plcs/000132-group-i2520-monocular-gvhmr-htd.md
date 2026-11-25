---
id: group-i2520-monocular-gvhmr-htd
type: group
task: plcs
sequence: 132
recorded_at: '2026-11-25'
title: PLCS単眼GVHMR Early Fusion・HTD-Refine段階アブレーションと実動画評価 (#2475/#2520)
issue:
- 2475
- 2520
provider: gemini
summary_impact: updated
members:
- run-i2520-b0-multiview-axial-e100-s42
- run-i2520-v1-m1-mono-htd-collapse-e100-s42
- run-i2520-v2-m0-mono-gvhmr-base-e100-s42
- run-i2520-v2-m1-mono-gvhmr-htd-e100-s42
- run-i2520-v2-a1-no-cal-noise-e100-s42
- run-i2520-v2-a2-no-bbox-aug-e100-s42
- run-i2520-v2-a3-world3d-only-e100-s42
- run-i2520-cross-eval-and-htd-stage-b
parents:
- group-i545-loss-head-tuning
papers:
- paper-2024-gvhmr
- paper-2026-htd-refine
tags:
- plcs
- monocular-gvhmr
- htd-refine
- ablation
- meiji-4cam
- i2475
- i2520
---

## 考察 / Findings

### 実験群の総括（親Issue #2475 / 子Issue #2520）
本実験群では、GVHMR型Early Fusion（カメラ内外パラメータ + COCO17 + CLIFF合成bbox、HMR2特徴なし、dim=512・12層・8head・RoPE）と HTD-Refine 拡張（Stage A PVA-Heads・Stage B `PLCSHTDRefiner`）を PLCS および `tennis_scene` パイプラインへ統合し、同一データ（`single_object_mixed`）・同一予算（100エポック・実効バッチ16・seed 42）で多視点参照基準（`B0`）および段階アブレーション（`M0`/`M1`/`M2-SV`/`M2-MV`/`A1`/`A2`/`A3`）を実施した。詳細表と採用・整理判断材料は [結果レポート](../../reports/plcs-monocular-gvhmr-htd.md) にまとめた。

### 確立した主要結論
1. **Stage A + Stage B（`M2-SV`）で単眼高周波ジッタと局所3D姿勢が大幅改善**:
   - 従来の canonical pose 課題（高周波jitter過多）に対し、`M2-SV`（`M1` + Stage B 25step）は合成3 split で **jerk `-61〜-65%`、MPJAE `-41〜-45%`、骨長分散 `-54〜-86%`、PA-MPJPE `0.070〜0.100 m`**（`mixed` では4カメラ `B0` の `0.092 m` を凌駕）を達成した。
   - 実動画 `meiji_4cam` 6クリップでも **jerk `-87.6%`（`163k → 20.3k m/s³`）、同一視点再投影中央値 `1.27 px`、DLT比PA-MPJPE中央値 `11.1 cm`、DLT欠損392フレームの `100%` 復元** を HMR2・SMPL-X 非依存かつ CPU `1.54 ms/frame` で確認した。
2. **実動画における単眼絶対深度の限界と多視点Stage B（`M2-MV`）による解消**:
   - 単眼のみ（`M2-SV`）では未見実カメラ（`meiji_4cam`）での光線方向深度ずれが残り、`vs DLT MPJPE = 3.70 m`（中央値 `3.27 m`）であった。
   - 一方、4視点の2D観測を同時最適化する多視点 Stage B（`M2-MV`, CPU追加 `0.61 ms/frame`）では `vs DLT MPJPE` が **`1.17 m`（中央値 `1.04 m`, -68.6%）**、全カメラ再投影中央値 `19.2 px` まで急減した。
3. **各データ増強・損失の寄与を確認**:
   - カメラ校正雑音（`A1` vs `M1`）: 無効化で全split位置誤差 `+9〜+29%`、実動画 `+6.5%` 悪化。
   - 合成bboxゆらぎ増強（`A2` vs `M1`）: 無効化で合成・実動画の jerk/MPJAE が `+23〜+39%` 悪化。
   - canonical/骨格角度監視（`A3` vs `M1`）: 無効化で PA-MPJPE と実動画深度（`4.26 m`）が悪化。
4. **発見した実装・設定上の不具合（後続コードPRでの修正対象）**:
   - **既定損失重みによる深度崩壊（v1診断）**: `loss/monocular_htd.yaml` の `kinematic_acceleration_consistency_weight=0.1` が学習時bboxゆらぎ×`fps²` で総損失の `93.6%` を支配し CLIFF深度崩壊（`25.3 m`）を引き起こす。また `rotation_weight=1.0 + angle_weight=0.5` が初期エポックの股関節退化を誘発する。v2バランス（`kinematic_*=0.0`, `rotation=0.02`, `angle=0.0`）を既定へ反映すべきである。
   - **`torsion_angle` 共線クラッシュ**: `losses.py:271` の `pred.require_valid` が完全伸展腕の1フレーム共線で学習を停止させた（`M1` Ep 68、`A1` Ep 55）。共線フレームの安全マスク化が必要である。

### 採用・既存モデル整理の推奨判断（人間判断待ち）
- **標準多カメラ `tennis_scene` 既定**: **現行 4カメラ DLT + GVHMR を維持**（未見実カメラでの単眼絶対深度 `3.70 m`・多視点Stage B `1.17 m` はDTL既定を置き換える水準に未達）。
- **単眼・ブロードキャスト・DLT欠損補完用途**: **明示プロファイル `MonocularPLCSReconstructionModule`（`M1` + Stage B `PLCSHTDRefiner`）を採用**（HMR2/SMPL-X非依存、DLT欠損100%復元、CPU `1.54 ms/f`）。
- **既存 `PLCSMultiViewAxialModel`（`B0`）**: **現時点では保持（削除しない）**。多視点コート絶対定位（`0.36〜0.56 m`）では依然として単眼より優位であり、次期イテレーション（実スタジアム焦点距離分布へのデータ拡張・既定loss修正・多seed検証）後に再判断することを推奨する。
- `summary_impact: updated` の理由: PLCSの単眼GVHMR＋HTD-Refine実装・評価結果と新規の未解決課題（既定loss修正・torsion共線・実カメラ光線深度校正）を `knowledge/summary.md` へ反映した。
