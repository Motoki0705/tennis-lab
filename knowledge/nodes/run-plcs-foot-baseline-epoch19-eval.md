---
id: run-plcs-foot-baseline-epoch19-eval
type: run
title: 'PLCS epoch19基準モデル: 同条件test・明治実クリップ再評価'
provider: codex
date: '2026-09-14'
status: done
config:
  model: plcs_multiview_axial_split
  checkpoint_epoch: 19
  test_seed: 1234
  test_scenes: 998
  camera_reference: camera_0
metrics:
  position_error_m: 0.6024814071915311
  position_xy_error_m: 0.5794727269724125
  angular_error_deg: 15.681732177734375
  real_root_reprojection_px: 105.58882620344812
artifacts:
  run_dir: knowledge/runs/run-plcs-foot-baseline-epoch19-eval
  predictions: knowledge/runs/run-plcs-foot-baseline-epoch19-eval/baseline_test.npz
  output_dir: /home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-foot-residual/outputs/plcs/foot_residual/comparison
  checkpoint: /home/kamimura/projects/tennis-lab/ckpt/plcs/plcs-multiview-axial-split-camera-view-v2-epoch19.ckpt
  metrics: knowledge/runs/run-plcs-foot-baseline-epoch19-eval/baseline_test_metrics.json
parents: []
relations: []
tags:
- plcs
- foot-residual
- baseline
- real-clip
session: 01a09b38-6993-7e23-9122-9482895fe0b4
repro:
  checkpoint_sha256: 8ab943011c0c3249dda51dad57f77406cf87ade9a589b57187fe6e3a685382f3
  command: scripts/plcs_foot_residual/evaluate.py; test seed1234; real clip stride2/window128/overlap64
---

## 考察 / Findings

### 要約
指定されたconfigと一致するepoch19 checkpointを再評価した。合成testの3D平均誤差0.602481m、実クリップのroot再投影平均105.589px。保存済み本番位置との平均差は約7.6e-7mで、推論経路を再現できた。

### アーキテクチャ詳細
512次元のgroup MLP、共有trunkなし、位置/回転の各axial branch 6段。位置は直接回帰。可視性でUVをゼロ化するが点ごとの可視性を独立featureとして埋め込まない。canonical headは存在するが、このcheckpointのcanonical教師損失は0である。

### メトリクスの解釈
元のtest split 998 scene、中心128フレーム、seed1234、reference camera_0を固定した再評価。実クリップに独立した3D GTはない。root再投影は左右hip中点との整合性であり、コートからの近似カメラ・2D検出・root定義の差を含む。court fit RMSEはcam0/1/2で約4.01/3.42/2.16px。

### アーキテクチャ⇄メトリクスの因果考察
合成testで直接回帰の水平誤差0.579mは、学習なしの足首ground priorの水平誤差0.459mを上回る。仮説として、観測から得られる幾何を位置出力に明示すれば、絶対位置の学習負担を減らせる。ただしprior誤差が大きい少数例があり、残差学習が必要。

### 既存実験との比較
ユーザー指定configとcheckpointのmodel/data/loss/trainingが完全一致することを確認。旧scene.npzの予測値をGTには使用していない。GPU test評価後、実クリップは本番と同じfloat32でCPU再実行した（最初のautocast推論はcanonical出力のNumPy変換で失敗）。

### 次に有効な実験
同じtrunkを初期値に、可視性・身体相対座標・足首ground priorの埋め込みと残差出力を導入し、trainの大誤差例を増やす。改善の各要因は複合変更として評価し、個別因果を断定しない。
