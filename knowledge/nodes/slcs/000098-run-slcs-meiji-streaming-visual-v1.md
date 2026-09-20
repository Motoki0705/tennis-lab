---
task: slcs
sequence: 98
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-streaming-visual-v1
type: run
title: 'SLCS Meiji可視化: 全長逐次保存・予測不変・最大RSS1.58GiB'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel validation-selected epoch56
  loss: inference only
  data: slcs/real_rgb_v1
  clip: video_001/clip_000
  camera: cam0
  device: cpu
  frame_step_3d: 3
metrics:
  exit_code: 0
  prediction_frames: 1316
  overlay_frames: 1316
  rendered_3d_frames: 439
  frames_without_homography: 0
  peak_rss_kib: 1659224
  elapsed_seconds: 151.10
  prediction_arrays_identical: true
  decoded_videos_identical: true
repro:
  commit: d6f29c8081f7fd664c2555bd318eda27e12efe99
  branch: codex/slcs-real-rgb
  command: >-
    /usr/bin/time -v env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m src.tasks.slcs.scripts.predict_clip paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs 'predict.checkpoint="slcs/train/real_rgb_no_ball_smooth/s42-takeover-003/logs/version_0/checkpoints/slcs-epoch=56.ckpt"' data.dataset_root=slcs/real_rgb_v1 data.split_file=slcs/real_rgb_v1/splits.json data.quality.min_window_label_ratio=0.5 predict.clip_id=video_001/clip_000 predict.camera_id=cam0 predict.device=cpu predict.frame_step=3 predict.output_dir=slcs/visualize/real_rgb_no_ball_smooth_meiji/s42-streaming-001
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-streaming-visual-v1
  log: knowledge/runs/run-slcs-meiji-streaming-visual-v1/run.log
  predictions: knowledge/runs/run-slcs-meiji-streaming-visual-v1/predictions.npz
  output_dir: outputs/slcs/visualize/real_rgb_no_ball_smooth_meiji/s42-streaming-001
  pr_preview: knowledge/runs/run-slcs-meiji-streaming-visual-v1/pr_preview
parents:
- run-slcs-full-no-smooth-meiji-visual-v2
relations: []
tags:
- slcs
- real-rgb
- visualization
- streaming
- cpu
---

## 考察 / Findings

### 要約

全frame保持から既存VideoWriterによる逐次保存へ切り替えた公開predict_clipを実動画で確認した。
全1316frameのoverlayと439frameの3D動画を完走し、推論を含む最大RSSは1659224KiB（約1.58GiB）。
修正前runの予測配列および両動画の全decoded frameと完全一致した。

### アーキテクチャ詳細

推論checkpoint・clip・設定は親と同じで、checkpoint SHAは
`870407e89ca0ff31d07a33395acf3a3538f49146274f3566bd36ff969b6733a2`。
overlay/3D画像を描画した順に既存writerへ渡し、frame画像の保持をO(1)にした。
同一directoryの一時MP4へ保存し、frame数検証とencoder close成功後にatomic replaceする。
失敗時は一時fileだけを削除して既存の最終動画を保全し、matplotlib figureもfinallyで閉じる。

### メトリクスの解釈

CLI終了コード0、経過151.10秒。ffprobeの全decode計数でoverlayは1920×1080・1316frame・59.94006fps、
3Dは1200×600・439frame・19.98002fps。欠損homographyは0。
旧run `s42-takeover-004`と新runの15個のNPZ配列/metadataを`np.array_equal`で照合した。
動画は`ffmpeg -i <video> -map 0:v:0 -f hash -hash sha256 -`で全decoded frameのhashを比較し、両方一致した。
比較結果とresource_usage.txtをbundleへ保存した。学習ではないため収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

旧overlayのlist+stackはこの動画の画像配列だけで約15.25GiBを必要とする実装だった。
旧runのpeak RSSは測定していないため、この試算と新しい実測値から厳密な削減率は算出しない。
今回の完走と低RSSは逐次保存の効果を支持するが、過去のsignal停止原因が解決したことの証明ではない。

### 既存実験との比較

親と同じ推論・描画内容を維持し、解像度・frame数・CRF17・fpsも変えていない。
統合先でSLCS unit/integrationとVideoWriter計241 testsが通過。新規17件は実動画、read/write順序、
空入力・短decode・draw/write/close失敗時のcleanupと既存動画保全を含む。対象Ruff/mypyも通過した。

### 次に有効な実験

以後のモデル比較ではこの逐次描画経路を使用し、定量評価と固定clipの目視を併用する。
省メモリ化そのものは位置精度・欠損時の頑健性を改善しないため、モデル学習の採否とは分ける。

PR向けには既存動画の3.0–11.0秒を10fpsで同期合成した。元の異なるFPSに対し同じwall-timeの直前frameを使い、補間はしない。
推論はやり直さず、全画角・未加工の軌道を維持する。GIFは表示用の縮小・色量子化のみ。1440px MP4・3時点PNG・生成コマンドとSHAはpr_preview/へ保存した。
この区間は定性的な例であり、全体の性能を代表する統計的な抽出ではない。ball影はz=0の投影で、画像ballとの一致には用いない。

![MeijiのRGB観測とSLCSの3D予測・疑似教師](../../runs/run-slcs-meiji-streaming-visual-v1/pr_preview/comparison.gif)
