---
id: paper-2026-htd-refine
type: paper
title: Natural Human Motion Recovery by Aligning High-Order Temporal Dynamics from
  Monocular Videos
year: 2026
authors:
- Dingkun Wei
- Zehong Shen
- Yan Xia
- Georgios Pavlakos
- Yujun Shen
- Xiaowei Zhou
tasks:
- plcs
- tennis_scene
source: https://arxiv.org/abs/2605.26879v1
summary: 単眼動画から推定した人体姿勢の高周波jitterや過剰平滑化を解消するため、時系列モデルから3D関節速度・3D関節加速度の高次時間動態（PVA）を直接推定し、2D再投影整合・jerk正則化と共に大域軌道最適化でアライメントする研究。
reason: 'Issue #2475において、単眼PLCSの3D骨格推定における奥行きjitter・足滑りと急峻な加減速保存のトレードオフを解決するため、人間がGVHMRと共に導入を指定した基礎研究。'
external_ids:
  arxiv: '2605.26879'
---

## このプロジェクトとの関係

Issue #2475 において、GVHMR（`paper-2024-gvhmr`）型の単眼PLCSアーキテクチャへ高次時間動態（3D速度・3D加速度）の学習と大域軌道最適化を導入するための一次根拠文献。既存PLCSの時間分解readout実験（`run-plcs-canonical-temporal-decomp-beta01-noaug`）で観測された「運動振幅の復元と高周波速度jitter（GT `0.069` に対し予測 `0.391`）のトレードオフ」を、単純な低域通過平滑化によるピーク加速度潰れを起こさずに解決する設計指針として採用する。

## 手法の要点

1. **問題意識**: 単眼動画からの人体動作復元（GVHMRやWHAMなど）では、位置レベルのフレーム間揺れ（特にカメラ光線・奥行き方向のjitterや足滑り）と、平滑化フィルタによる高速動作の鈍化（over-smoothing）が両立しにくい。一方で、時系列モデルから直接推定した1次微分（3D速度 $\hat{V}_t$）および2次微分（3D加速度 $\hat{A}_t$）は運動のリズム・加減速ピークの位相を正確に捉えやすい。
2. **PVA-Net（Position-Velocity-Acceleration Network）**: 時系列特徴から2D関節位置に加えて3D関節速度 $\hat{V}_t$ と3D関節加速度 $\hat{A}_t$ を専用ブランチ（時間畳み込み＋時間Attention）で直接回帰し、位置との有限差分整合および教師あり損失で学習する。
3. **HTD-Refine（High-Order Temporal Dynamics Alignment）**: 推論時のplug-and-play大域最適化として、初期3D予測から出発し、予測速度・予測加速度への整合項（$\mathcal{L}_V + \mathcal{L}_A$）、信頼度重み付き2D再投影項（$\mathcal{L}_{\text{reproj}}$）、3階差分jerk正則化（$\mathcal{L}_{\text{jerk}}$）を同時最小化する。これにより急峻な加減速波形を保ったまま高周波振動と接地破綻を抑える。

## PLCS・tennis_sceneへの適合方針と適用限界

- **RGB非依存・3D COCO17骨格出力への拡張**: 原論文は凍結ViTPose画像特徴マップからPVAを推定しSMPLパラメータを最適化するが、PLCSはRGB画像を持たない合成動作学習であり、出力もSMPLではなく3D COCO17骨格である。そのため、(Stage A) 単眼PLCSのRoPE時系列特徴上に3D速度・3D加速度・接地静止確率を推論する `PVA-Heads` を構築し、(Stage B) 自由関節最適化で生じ得る骨伸縮を防ぐ「骨長時系列不変・左右対称制約」と既知コート平面 `z=0` の「接地・足滑り抑制制約」を加えた `PLCSHTDRefiner` として実装する。
- **検証を要する点**: ViTPose実測ゆらぎを含む2D観測と近似コートカメラ校正誤差の下で、PVA予測自体にどの程度ノイズが混入するか、およびテニスのスプリットステップ・切り返し動作のピーク加速度保持率とW-MPJPE・LOVO再投影誤差が段階アブレーション（Baseline / +Stage A / +Stage B）で一貫して改善するかを固定splitと実クリップで検証する。

