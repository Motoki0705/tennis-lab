---
id: run-b00-clay-flare-50-v1
type: run
title: B00 クレー化 Flare 50視点
provider: codex
date: '2026-09-20'
status: done
session: 01a0bd70-e78d-7732-ba19-1a4595e91a32
config:
  model: gpt-image-2.5-flare-2026-09-08
  quality: high
  generation_size:
  - 1536
  - 864
  source_interval:
  - 0
  - 248
  sample_count: 50
  reference_sha256: e938c6671d26263ccdea93472f475e9344e8f25f3586cbec0b88a6edc0c18428
metrics:
  selected_images: 50
  new_api_images: 49
  reused_images: 1
  new_api_calls: 49
  api_error_files: 0
  mean_request_seconds: 17.106455362408028
  wall_seconds_including_pilot_review_gap: 776.0532147884369
  new_input_tokens: 134015
  new_output_tokens: 52822
  visual_reviewed_images: 50
artifacts:
  run_dir: knowledge/runs/run-b00-clay-flare-50-v1
  output_dir: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001
  images: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/generation/images
  review: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/review/index.html
  overview: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/review/overview-50.jpg
parents:
- run-b00-clay-flare-high-v1
tags:
- synthetic-data
- image-api
- clay
---

## 考察 / Findings

### 要約
選択した50視点のクレー画像が揃った。比較で採用した1枚を入力・プロンプト・モデルの完全一致を確認して再利用し、新規49枚はAPIエラー・再送なしで生成できた。

### アーキテクチャ詳細
固定参照と原画像を1536×864で入力し、同一プロンプト・Flareの2026-09-08版・quality=highで編集した。2並列、送信開始間隔13秒。学習用には原寸1920×1080へ戻し、元SfMをコピーしてカメラと座標正規化を維持する。

### メトリクスの解釈
新規APIの平均要求時間は約17.1秒、最初の要求から最後の応答まで約776秒（パイロットレビューの間隔を含む）。50枚を原画像・生成画像・50%重ね合わせで目視確認した。これは画素単位の幾何誤差を定量測定したことを意味しない。

### アーキテクチャ⇄メトリクスの因果考察
コート境界・ネット・建物の主要配置は接触シート上で保たれて見える。一方で細かい植生、空、線幅は再描画されている。視点間の整合性への影響は3DGS再学習で検証する必要がある。

### 既存実験との比較
1視点のSunburst/Flare比較で使用した参照をそのまま固定した。50枚の出力条件と解像度は同じで、未変換画像の混入はない。

### 次に有効な実験
生成50枚（学習42／評価8）で7,000ステップのNHTを実行し、白線・ネットの二重化や背景の不一致をレンダリングで確認する。
