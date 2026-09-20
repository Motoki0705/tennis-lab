---
task: slcs
sequence: 63
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-full-no-smooth-meiji-visual-interrupted-v1
type: run
title: 'SLCS全体版baseline: Meiji描画のprocess消失、overlay未完了'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: failed
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
repro:
  commit: 0e1e32c19841f3a594f9e1c0d7b812ab54e8df5f
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m src.tasks.slcs.scripts.predict_clip paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs 'predict.checkpoint="slcs/train/real_rgb_no_ball_smooth/s42-takeover-003/logs/version_0/checkpoints/slcs-epoch=56.ckpt"' data.dataset_root=slcs/real_rgb_v1 data.split_file=slcs/real_rgb_v1/splits.json data.quality.min_window_label_ratio=0.5 predict.clip_id=video_001/clip_000 predict.camera_id=cam0 predict.device=cpu predict.frame_step=3 predict.output_dir=slcs/visualize/real_rgb_no_ball_smooth_meiji/s42-takeover-003
artifacts:
  run_dir: knowledge/runs/run-slcs-full-no-smooth-meiji-visual-interrupted-v1
  predictions: knowledge/runs/run-slcs-full-no-smooth-meiji-visual-interrupted-v1/predictions.npz
  output_dir: outputs/slcs/visualize/real_rgb_no_ball_smooth_meiji/s42-takeover-003
parents: [run-slcs-full-real-rgb-no-ball-smooth-val-v3]
tags: [slcs, real-rgb, visualization, cpu]
---

## 考察 / Findings

### 要約

Meiji valのclip_000/cam0はpredictions.npzとscene_3d.mp4を保存したが、overlayの完了前にprocessが消失した。
環境再起動後にexec handleと対象processがなく、overlayも存在しなかった。終了コードは取得できていない。

### アーキテクチャ詳細

既存predict_clip CLI、CPU float32、window120/eval stride120、augmentationなし。
入力root・checkpoint root・出力rootを明示し、保存checkpointのSHAは
`870407e89ca0ff31d07a33395acf3a3538f49146274f3566bd36ff969b6733a2`。
3Dは3frame間隔で描画し、2D overlayは原動画の全frameを保持する。

### メトリクスの解釈

保存予測は1316frameで全値有限、coverage最小1。3D動画は439frameでffprobeによる読取が成功した。
overlayの成功・完全なCLI完走は未証明であり、このrunはfailedとする。

### アーキテクチャ⇄メトリクスの因果考察

描画実装は全RGB frameをlistに保持してnp.stackするため、1920×1080・1316frameでは両配列だけで約15.25GiBとなる。
同時に実行していた学習のRSSは約16GiBで、メモリ圧迫のリスクがある。ただしこのrunのpeak RSSや停止原因は未測定で、
メモリ不足を原因と断定しない。学習側のexit143をこのCPU processの終了コードへ転用しない。

### 既存実験との比較

同時に開始したbroadcast clipの可視化は小さい640×360画像で完走した。
媒体サイズ以外にも処理時間が違うため、停止原因の統制比較ではない。

### 次に有効な実験

既存成果物を上書きせず新しいrun-idでMeiji可視化を再実行し、全frameの動画を確認する。
既存video writerを再利用した逐次保存を調べ、frame数・解像度を維持してメモリ保持を減らす。
