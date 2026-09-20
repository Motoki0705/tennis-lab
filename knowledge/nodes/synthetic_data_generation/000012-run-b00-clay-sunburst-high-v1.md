---
id: run-b00-clay-sunburst-high-v1
type: run
task: synthetic_data_generation
sequence: 12
recorded_at: '2026-09-20'
title: B00クレー化 sunburst high 1視点
provider: codex
session: 01a0bd70-e78d-7732-ba19-1a4595e91a32
date: '2026-09-20'
status: done
config:
  model: gpt-image-2.5-sunburst-2026-09-08
  quality: high
  size: 1536x864
  target: frame_000000.jpg
  reference: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/comparison-sunburst-flare-v002/inputs/reference.png
  prompt_sha256: 133e4f1a7ce9339bf236512e1c1bbac4457d671fef6dd9f2136548752fd6e2ea
  input_sha256:
  - e938c6671d26263ccdea93472f475e9344e8f25f3586cbec0b88a6edc0c18428
  - 2b32079c47bd56d52bcd0d90d0604ec1db28d413a1a672c74ebc2d88812f701d
metrics:
  elapsed_seconds: 24.24534948300061
  input_tokens: 2735
  output_tokens: 1078
  total_tokens: 3813
artifacts:
  run_dir: knowledge/runs/run-b00-clay-sunburst-high-v1
  output_dir: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/comparison-sunburst-flare-v002/sunburst
  image: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/comparison-sunburst-flare-v002/sunburst/api-result.png
  comparison: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/comparison-sunburst-flare-v002/comparison.png
parents: []
relations: []
papers: []
tags:
- synthetic-data
- image-api
- clay
---

## 考察 / Findings

### 要約
参照画像と対象画像を同一条件で入力し、地表のクレー化を確認した。対象はB00の1視点のみ。

### アーキテクチャ詳細
OpenAI Images edits APIへ固定参照、原画像の順で入力。両方を1536×864へ事前変換し、同じ固定プロンプトとquality=highを使用した。モデル内部の構成は未確認。

### メトリクスの解釈
所要時間はHTTP要求単位の1回の実測で、並行要求の待ち時間も含む。入力2735・出力1078トークンは応答の実測値。視点間整合性や3DGS品質を表す指標ではない。

### アーキテクチャ⇄メトリクスの因果考察
モデル以外の入力条件を同じにした。単発の速度差の原因をモデル内部の構成に帰属することはできない。両者とも空や植生の細部が再描画され、厳密な画素幾何の保存は未保証。

### 既存実験との比較
Sunburstは24.245秒、Flareは17.613秒。今回のクレー化の見た目には大きな差がなく、Flareの地面はより粒状感が強い。最初のHTTP 400はinput_fidelity非対応によるもので、comparison-sunburst-flare-v001に残している。

### 次に有効な実験
モデルの選定後、別の2視点でも白線・ネット・背景構造の維持を確認し、採用画像50枚で7,000ステップのNHT再学習を行う。

画像APIの生成記録であり、TensorBoardの学習曲線は生成されない。
