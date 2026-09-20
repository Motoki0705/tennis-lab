---
id: run-b00-clay-flare-100-v1
type: run
title: B00 Flare 50視点を追加し100枚へ拡張
provider: codex
session: 01a0bd70-e78d-7732-ba19-1a4595e91a32
date: '2026-09-20'
status: done
config:
  model: gpt-image-2.5-flare-2026-09-08
  quality: high
  generation_size:
  - 1536
  - 864
  reference_sha256: e938c6671d26263ccdea93472f475e9344e8f25f3586cbec0b88a6edc0c18428
  selection: retain 50 parent views; repeatedly bisect largest interval; earlier interval
    on ties
  train_images: 86
  validation_images: 14
metrics:
  selected_images: 100
  new_api_images: 50
  reused_images: 50
  new_api_calls: 50
  api_error_files: 0
  mean_request_seconds: 16.97115678272
  wall_seconds: 676.3797569274902
  new_input_tokens: 136750
  new_output_tokens: 53900
  new_visual_reviewed_images: 50
  inherited_reviewed_images: 50
artifacts:
  run_dir: knowledge/runs/run-b00-clay-flare-100-v1
  output_dir: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-100-30k-v001
  review: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-100-30k-v001/review/index.html
  overview: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-100-30k-v001/review/overview-100.jpg
  archive: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-100-30k-v001/generation/flare-100-api-png-1536x864.zip
parents:
- run-b00-clay-flare-50-v1
relations: []
tags:
- synthetic-data
- image-api
- clay
---

## 考察 / Findings

### 要約
既存の採用済み50枚を保持し、同じFlare・固定参照・プロンプトで新たに50枚を生成した。APIエラー・再送は0件、合計100枚を採用した。

### アーキテクチャ詳細
対象は元のframe 0〜248の範囲。既存の視点間の最大間隔を二分し、同間隔なら早い側を優先して追加50視点を決定した。全100のフレーム番号をvariant.yamlに保存。生成1536×864、quality=high、2並列・送信間隔13秒。元順位の分割を保持し、学習86枚／評価14枚となる。

### メトリクスの解釈
追加分はAPI 50回で50枚。平均応答16.971秒、batch開始から最後の応答保存まで676.380秒。新規入力136,750 tokens・出力53,900 tokens。これらはAPIが返した使用量であり金額換算はしていない。

### アーキテクチャ⇄メトリクスの因果考察
原画像・生成画像・50%重ね合わせで追加50枚を目視し、白線・ネット・背景の主要構図を確認した。細かい線の描画、植生、色・テクスチャの差は残る。幾何の画素単位一致や視点間の完全な整合性を確認したものではない。

### 既存実験との比較
既存50枚は生成原本と学習用JPEGのハッシュが親と一致する実コピー。既存結果の再課金はない。50枚版3万ステップのGPU学習とAPI生成を並行し、100枚版3万ステップを同じ共有queueの後続へ登録した。

### 次に有効な実験
50枚／100枚を同じ3万ステップ・seed 42・factor 2・固定カメラで学習し、共通する8評価視点の指標と白線・ネットのレンダリングを比較する。
