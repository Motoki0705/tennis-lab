---
id: run-slcs-full-no-smooth-meiji-visual-v2
type: run
title: 'SLCS全体版baseline: Meiji val clipのCPU可視化完了'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel validation-selected epoch56
  data: slcs/real_rgb_v1
  clip: video_001/clip_000
  camera: cam0
  device: cpu
  frame_step_3d: 3
metrics:
  prediction_frames: 1316
  rendered_3d_frames: 439
  overlay_frames: 1316
  frames_without_homography: 0
  exit_code: 0
repro:
  commit: 071e500fd86f530715b6633a15e6fc2dd23c5ebc
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m src.tasks.slcs.scripts.predict_clip paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs 'predict.checkpoint="slcs/train/real_rgb_no_ball_smooth/s42-takeover-003/logs/version_0/checkpoints/slcs-epoch=56.ckpt"' data.dataset_root=slcs/real_rgb_v1 data.split_file=slcs/real_rgb_v1/splits.json data.quality.min_window_label_ratio=0.5 predict.clip_id=video_001/clip_000 predict.camera_id=cam0 predict.device=cpu predict.frame_step=3 predict.output_dir=slcs/visualize/real_rgb_no_ball_smooth_meiji/s42-takeover-004
artifacts:
  run_dir: knowledge/runs/run-slcs-full-no-smooth-meiji-visual-v2
  predictions: knowledge/runs/run-slcs-full-no-smooth-meiji-visual-v2/predictions.npz
  output_dir: outputs/slcs/visualize/real_rgb_no_ball_smooth_meiji/s42-takeover-004
parents: [run-slcs-full-no-smooth-meiji-visual-interrupted-v1]
tags: [slcs, real-rgb, visualization, cpu]
---

## 考察 / Findings

### 要約

Meiji valのclip_000/cam0を新しい出力先へ再実行し、CLI終了コード0で完走した。
全1316frameの1920×1080 overlay、439frameの3D動画、全長予測を保存した。旧中断runは保全した。

### アーキテクチャ詳細

既存predict_clip CLI、CPU float32、window120/eval stride120、augmentationなし。
入力root・checkpoint root・出力rootを明示し、保存checkpointのSHAは
`870407e89ca0ff31d07a33395acf3a3538f49146274f3566bd36ff969b6733a2`。
3Dは3frame間隔で描画し、2D overlayは原動画の全frameを保持する。

### メトリクスの解釈

overlayは1920×1080・59.94006fps・1316frame、3Dは1200×600・約19.98fps・439frame。
投影用homographyがないframeは0。6時点のcontact sheetを確認し、近/遠選手の位置のずれとballの変化を表示できた。
これは成功例だけを誤差で選んだものではなく、事前に指定したclip_000である。学習曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

今回はGPU学習と同時実行せず完走したが、環境条件を制御した比較ではなく前回停止原因の確定には使わない。
画像上のball shadowは地面への投影で、実際の高さをもつballの再投影ではない。

### 既存実験との比較

親runと同じcheckpoint・clip・camera・設定で、出力先だけを変更した。
教師はPLCS/BLCSに幾何補正を加えた推定であり、動画のlabel表示を実測GTと呼ばない。

### 次に有効な実験

学習中の不必要なメモリ競合を避けるため、overlayの全frame保持を逐次writerへ移す。
次の選定モデルも同じ固定clipと定量評価を併用して比較する。
