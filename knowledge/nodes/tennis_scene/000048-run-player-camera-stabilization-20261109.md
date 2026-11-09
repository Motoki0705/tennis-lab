---
id: run-player-camera-stabilization-20261109
type: run
task: tennis_scene
sequence: 48
recorded_at: '2026-11-09'
title: 頭部回転補正とGoPro参考レンズの比較
issue: 1776
provider: codex
status: running
summary_impact: none
config:
  algorithm: gaussian_markley_quaternion_v1
  sigma_seconds: [0, 0.06, 0.08, 0.12]
  truncate_sigma: 3
  translation: unchanged
  horizon_lock: false
  camera_context_frames: [0, 1010]
  comparison_source_frames: [0, 240]
  lens: equidistant_diagonal_159_approximation
  panel_resolution: [640, 360]
  pinhole_render_resolution: [2560, 1440]
  warp_samples_per_axis: 2
  depth_of_field: false
  colab_gpu_time_limit_hours: 4
metrics: {}
artifacts:
  lens_survey: knowledge/runs/run-player-camera-stabilization-20261109/lens-survey.md
  prepare_script: knowledge/runs/run-player-camera-stabilization-20261109/prepare_inputs.py
  render_script: knowledge/runs/run-player-camera-stabilization-20261109/render_comparison.py
parents:
- run-player-view-blender-full-20261109
relations: []
papers:
- paper-2007-markley-quaternion-averaging
tags: [player-view, camera-stabilization, diagnostic]
---

人間の[強度を動画で選ぶ・rollと大きな動きを残す判断](https://github.com/Motoki0705/tennis-lab/issues/1776#issuecomment-6730136323)に基づく比較。現在は実装・CPU検証の段階で、Colab動画はまだ生成していない。statusはrunningであり、人間の受入や強度の採用を示さない。

Markleyの平均をGaussian時間重みで使い、連続有効区間を跨がず回転だけを処理する。生のC2は保持し、σ=0は元の数値のコピー。全列を補正してから4秒を選ぶため、後の全区間版と同じ文脈で比較する。

[レンズ調査](../../runs/run-player-camera-stabilization-20261109/lens-survey.md)は、最新固定レンズMISSION1系を参考にした等距離魚眼の近似を選んだ。公式native159度は画角の軸が明記されていないので、設計上の対角159度と区別する。F2.8は参考値でDOFは無効、顔offsetは維持する。C5の共通PINHOLE上で深度合成してからlinear RGBを一度だけ写像する。実機の完全再現や実C5性能の検証とは称さない。

現行summaryの保存3D再構築・表示品質の人間確認という判断は変わらないためsummary_impactはnone。Colab終了後に実行の成否・Drive保存・動画decodeと比較指標を追記する。TensorBoardは学習を行わないため対象外。
