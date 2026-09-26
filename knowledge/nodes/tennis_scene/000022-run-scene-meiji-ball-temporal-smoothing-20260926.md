---
id: run-scene-meiji-ball-temporal-smoothing-20260926
type: run
task: tennis_scene
sequence: 22
recorded_at: '2026-09-26'
title: Meiji clip_000の3方式球軌道平滑化
provider: codex
date: '2026-09-26'
status: done
config:
  clip: video_000/clip_000
  frames: 1010
  fps: 59.94006
  raw_ball_artifact_id: 337c149a57a3e95374d15ebfb79633d6edecd3919d5773bdff3b4aa02ad1b537
  methods: [savgol, robust_spline, ballistic_rts]
  default_method: none
  segmentation: valid_runs_y_reversal_low_z_minimum
  event_positions: held_at_raw_triangulation
  gap_policy: no_fill
  player_source: confirmed_target_2_players
metrics:
  valid_ball_frames: 987
  protected_event_frames: 12
  player_smpl_valid_frames: [1004, 995]
  flight_acceleration_p95_mps2:
    raw: 1686.695
    savgol: 207.419
    robust_spline: 29.070
    ballistic_rts: 19.782
  reprojection_p95_px:
    raw: 7.945
    savgol: 8.274
    robust_spline: 8.458
    ballistic_rts: 8.368
  reprojection_over_20px_of_2718:
    raw: 0
    savgol: 7
    robust_spline: 11
    ballistic_rts: 6
  displacement_from_raw_p95_m:
    savgol: 0.173
    robust_spline: 0.223
    ballistic_rts: 0.237
repro:
  commit: b2c79f56603e64e33fad499fd456a39f06f06123
  branch: codex/ball-trajectory-smoothing
  command: .venv/bin/python -m scripts.compare_ball_smoothing --scene-index
    /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/tennis_scene/scene.json
    --output-dir /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/ball_smoothing_meiji_20260926/variants
artifacts:
  run_dir: knowledge/runs/run-scene-meiji-ball-temporal-smoothing-20260926
  gallery: knowledge/runs/run-scene-meiji-ball-temporal-smoothing-20260926/index.html
  repro_script: knowledge/runs/run-scene-meiji-ball-temporal-smoothing-20260926/repro.sh
  report: knowledge/runs/run-scene-meiji-ball-temporal-smoothing-20260926/report.json
  comparison_figure: knowledge/runs/run-scene-meiji-ball-temporal-smoothing-20260926/trajectory_comparison.png
  mesh_videos:
    savgol: knowledge/runs/run-scene-meiji-ball-temporal-smoothing-20260926/savgol_mesh_full.mp4
    robust_spline: knowledge/runs/run-scene-meiji-ball-temporal-smoothing-20260926/robust_spline_mesh_full.mp4
    ballistic_rts: knowledge/runs/run-scene-meiji-ball-temporal-smoothing-20260926/ballistic_rts_mesh_full.mp4
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/ball_smoothing_meiji_20260926/variants
parents:
- run-scene-component-meiji-target2-20260925
relations: []
papers: []
tags: [real_clip, ball_triangulation, temporal_smoothing, mesh_visualization, diagnostic]
---

## 観測と比較

[確認済み2人軸のMeiji clip_000](000021-run-scene-component-meiji-target2-20260925.md)の保存済み`ball_triangulation`を固定し、3方式を同じ987有効frameに適用した。`savgol`は11frame局所2次多項式、`robust_spline`はHuber重み付き加速度正則化、`ballistic_rts`は重力を含む位置・速度状態のKalman/RTS平滑化。Y方向反転と低いZ極小で検出した12候補frameを元の三角測量位置に固定し、無効frameを補間しない。新しい`ball_smoothing` componentは元の三角測量artifactを別に保持し、各方式を設定で選ぶ。既定の`none`は元3D位置を変更しない。

飛行中の加速度ノルムp95は、raw `1686.695`、局所多項式`207.419`、ロバスト正則化`29.070`、重力付きRTS`19.782 m/s²`。この指標は検出したイベント付近の3frameを除外し、同じ有効な3連続frame上で計算した。位置のrawからの移動p95は順に`0.173`、`0.223`、`0.237 m`。採用camera観測2718件での再投影誤差p95はraw `7.945 px`から順に`8.274`、`8.458`、`8.368 px`へ増えた。20px超もraw `0`から`7/11/6`件となった。滑らかさは改善したが、入力2Dとの一致には小さな悪化と一部大きな外れがある。指標と全時系列は[比較図](../../runs/run-scene-meiji-ball-temporal-smoothing-20260926/trajectory_comparison.png)と[数値JSON](../../runs/run-scene-meiji-ball-temporal-smoothing-20260926/report.json)に固定した。

各方式のscene archiveを作り、元sceneとの差分が`ball_3d`配列だけであることを照合した。3本のmeshフルシーンはすべて1200×800、1010frame、59.94006fps、16.850167秒のH.264で、2名のSMPL mesh・コート・カメラを共通にした。[局所多項式](../../runs/run-scene-meiji-ball-temporal-smoothing-20260926/savgol_mesh_full.mp4)、[ロバスト正則化](../../runs/run-scene-meiji-ball-temporal-smoothing-20260926/robust_spline_mesh_full.mp4)、[重力付きRTS](../../runs/run-scene-meiji-ball-temporal-smoothing-20260926/ballistic_rts_mesh_full.mp4)。ローカルの`artifacts.output_dir/index.html`では3本と比較図を同時に参照できる。

## 判断と限界

これは1クリップ上の平滑さ・再投影・見た目の比較で、人手3D正解がないため絶対位置精度の改善は主張しない。raw球2Dは外部確認済み注釈、人物対応とsideは確認済み入力であり、検出モデル/Re-IDの独立精度評価でもない。12候補のイベント時刻は軌道からのヒューリスティック検出で真値ではなく、支持のない23frameには3D球を描かない。現時点では既定`none`を維持し、選択するなら再投影外れと接触時刻の目視確認が必要。次は人手で確認したバウンド・打球frameと独立3D基準、別clip/会場で、イベント近傍の誤差と再投影tailを比較する。学習runではなくTensorBoard曲線はない。
