---
task: slcs
sequence: 92
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-observation-review-v4
type: run
title: Meiji第3収録の画像確認とview-local人物番号の確認
provider: codex
date: '2026-09-19'
status: done
config:
  observations: tennis_scene/precompute/meiji_dino_vitpose/s42-004
  clips:
  - video_002/clip_000
  - video_002/clip_002
  device: cpu
  diagnostic_only: true
metrics:
  clips: 2
  camera_streams: 6
  reviewed_frame_camera_images: 52
  min_detection_sample_coverage: 0.9746835443037974
  min_pose_supported_fraction: 1.0
  min_torso_confident_fraction: 0.8269230769230769
  max_adjacent_box_center_previous_diagonal: 0.03471110016107559
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-observation-review-v4
  output_dir: outputs/tennis_scene/analyze/meiji_observation_review/s42-004
  diagnostics: knowledge/runs/run-slcs-meiji-observation-review-v4/results.json
  provenance: knowledge/runs/run-slcs-meiji-observation-review-v4/analysis_provenance.json
parents:
- run-slcs-meiji-observation-review-v3
relations: []
tags:
- slcs
- meiji
- observation
- cross-view
- cpu-diagnosis
- visual-review
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 740e788877a00971a7ab3d913458c8328379815e
  branch: codex/slcs-real-rgb
  command: env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    .venv/bin/python -B knowledge/runs/run-slcs-meiji-observation-review-v1/probe.py
    --project-root /home/kamimura/projects/tennis-lab --observation-root /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004
    --clip video_002/clip_000 --clip video_002/clip_002
    --output-dir outputs/tennis_scene/analyze/meiji_observation_review/REPRO_NEW_RUN_ID
---

## 考察 / Findings

### 要約
第3収録で最初に観測生成が完了した適格2clipについて、全6枚のcontact sheet・52画像を親が確認した。各視点内では対象選手を追跡して見える。cam1のP0は水色シャツのレシーバー、逆端から撮るcam2のP0は濃青シャツのサーバーだが、この番号は各視点のnear/farを表す。教師の共通人物軸への並べ替えは別工程にあり、画像の番号反転を教師の誤対応とは判定しない。

### アーキテクチャ詳細
既存CPU probeを変更せず実行した。各視点の均等5frameと各選手の最大相対box移動区間の両端を選定し、重複frameを除いた。動画・人物NPZ・metadataの前後dual SHA照合と、動画の記録済みSHA照合を通過。画像・実行コード・reportのhashをprovenanceに保存した。教師の採用閾値・model・splitは変更していない。

### メトリクスの解釈
最小検出sample率97.47%、pose_supported率100%、腰肩支持率82.69%、最大隣接box移動は前frameのbox対角長の3.47%。これらは視点内の検出・支持の値で、別視点の同じ番号が同一人物であることを証明しない。clip_000のcam1は8画像、cam0/cam2は各9画像、clip_002のcam2は8画像、cam0/cam1は各9画像。学習runではないため収束曲線なし。

### アーキテクチャ⇄メトリクスの因果考察
cam2は画像内の建物・コート背景からcam0/cam1とは逆側の撮影に見える。`740e7888` のコード確認では、`reference_pipeline/reconstruction.py:associate_people` が `view_half_turns=[false,false,true]` を使ってcam2の地面座標を反転し、共通座標のcourt Y中央値で人物を並べ替えていた。raw cacheをそのまま描く本probeはこの変換前であり、両clipの衣服と動作は期待されるview-local番号の反転と整合する。選定画像やコード確認だけから全frameの対応や3D精度は結論しない。

### 既存実験との比較
v1–v3と合わせて10clip・267画像・3収録を確認した。既存のvideo_000/clip_000とvideo_001/clip_002のcam1/cam2画像も再参照し、同様のview-local番号の反転を確認したが、再参照分を画像数に加算しない。以前の「明白な隣接コートへの切替えなし」は視点内の時間的追跡の所見であり、カメラ間対応を検証したものではない。v3のraw P0/P1を視点間で集計した腰肩支持数も、canonical人物ごとの支持とは区別して再確認する。

### 次に有効な実験
保存されたcourt校正とpeople cacheへのproductionの `associate_people` 適用は、後続の[CPU照合run](000088-run-slcs-meiji-canonical-association-check-v1.md)で全10clipの配列一致まで確認した。教師生成後には `player_association_result.json` とcanonical軸の再構成画像を確認する。今後の観測probeの表示は `local P0/P1` と明示し、観測生成の完了率や高い関節confidenceを教師完成の証拠としない。
