---
task: slcs
sequence: 60
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-full-no-smooth-broadcast-visual-v1
type: run
title: 'SLCS全体版baseline: broadcast val clipの予測・教師可視化'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel validation-selected epoch56
  data: slcs/real_rgb_v1
  clip: broadcast_indoor_hard/clip_001
  camera: cam0
  device: cpu
  frame_step_3d: 3
metrics:
  prediction_frames: 1172
  rendered_3d_frames: 391
  overlay_frames: 1172
  frames_without_homography: 0
  exit_code: 0
repro:
  commit: 0e1e32c19841f3a594f9e1c0d7b812ab54e8df5f
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m src.tasks.slcs.scripts.predict_clip paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs 'predict.checkpoint="slcs/train/real_rgb_no_ball_smooth/s42-takeover-003/logs/version_0/checkpoints/slcs-epoch=56.ckpt"' data.dataset_root=slcs/real_rgb_v1 data.split_file=slcs/real_rgb_v1/splits.json data.quality.min_window_label_ratio=0.5 predict.clip_id=broadcast_indoor_hard/clip_001 predict.camera_id=cam0 predict.device=cpu predict.frame_step=3 predict.output_dir=slcs/visualize/real_rgb_no_ball_smooth_broadcast/s42-takeover-003
artifacts:
  run_dir: knowledge/runs/run-slcs-full-no-smooth-broadcast-visual-v1
  predictions: knowledge/runs/run-slcs-full-no-smooth-broadcast-visual-v1/predictions.npz
  output_dir: outputs/slcs/visualize/real_rgb_no_ball_smooth_broadcast/s42-takeover-003
  pr_preview: knowledge/runs/run-slcs-full-no-smooth-broadcast-visual-v1/pr_preview
parents: [run-slcs-full-real-rgb-no-ball-smooth-val-v3]
tags: [slcs, real-rgb, visualization, cpu]
---

## 考察 / Findings

### 要約

broadcast validationの唯一のclipを、validation選定epoch56でCPU推論・可視化した。CLI終了コード0、
全1172frameのoverlayと391frameの3D動画を確認した。性能の採用判定ではなく定性的な確認である。

### アーキテクチャ詳細

既存predict_clip CLI、CPU float32、window120/eval stride120、augmentationなし。
入力root・checkpoint root・出力rootを明示し、保存checkpointのSHAは
`870407e89ca0ff31d07a33395acf3a3538f49146274f3566bd36ff969b6733a2`。
3Dは3frame間隔で描画し、2D overlayは原動画の全frameを保持する。

### メトリクスの解釈

全予測配列は有限、coverage最小1。overlayは640×360・30fps・1172frame、3Dは1200×600・10fps・391frame。
投影用homographyがないframeは0。6時点のcontact sheetも保存し目視した。
遠側選手の地面投影に実画像の足元からのずれが残る。ball shadowは高さ0への投影なので、画像上のballとの一致度には使わない。

### アーキテクチャ⇄メトリクスの因果考察

平均誤差で隠れる選手位置のずれを確認できるが、単眼homographyと擬似教師を用いた表示であり独立3D正解ではない。
見た目が成立することを頑健性の達成と呼ばない。

### 既存実験との比較

親runのbroadcast10窓の定量結果と同じ選定重み・clipを使う。clip推論では重なるwindowを平均するため、
window occurrencesの定量評価をこの動画から再計算したとは扱わない。

### 次に有効な実験

全体版gap48の学習完了後も同じclipを比較する。full/gapの位置・motion・裾誤差を主な採否根拠とする。

PR向けにはMeijiと同じ3.0–11.0秒を10fpsで同期合成した。既存の予測動画を使い、時間補間・軌道の外れ値除去はしない。
1440px MP4・3時点PNG・生成コマンドとSHAはpr_preview/へ保存した。GIFは表示用の縮小・色量子化のみ。
これは定性的な例で、遠側選手の地面投影ずれなどの失敗も残している。teacherは実測3D正解ではない。

![broadcastのRGB観測とSLCSの3D予測・疑似教師](../../runs/run-slcs-full-no-smooth-broadcast-visual-v1/pr_preview/comparison.gif)
