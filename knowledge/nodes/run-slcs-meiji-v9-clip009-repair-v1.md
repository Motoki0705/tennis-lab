---
id: run-slcs-meiji-v9-clip009-repair-v1
type: run
title: 'Meiji v9 clip009: 公開前整合性チェック付きCPU再生成'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/meiji_rgb_v9_repair_clip009_v1
  source_clip: video_000/clip_009
  stage: infer
  device: cpu
metrics:
  audited_clips: 1
  audit_error_clips: 0
  player_0_positive_weight_fraction: 1.0
  player_1_positive_weight_fraction: 0.9610215053763441
  ball_positive_weight_fraction: 0.8723118279569892
  raw_positive_weight_ball_reprojection_mean_px: 14.920834094323938
  refined_positive_weight_ball_reprojection_mean_px: 4.192707020530974
  supported_player_speed_max_mps: 8.416019439697266
  supported_ball_speed_max_mps: 61.464561462402344
  unsupported_ball_longest_gap_frames: 56
repro:
  commit: ab394121
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tennis_scene.scripts.build_slcs_dataset
    device=cpu stage=infer 'dataset_clip_ids=[video_000/clip_009]'
    dataset_output_directory=slcs/meiji_rgb_v9_repair_clip009_v1
    output_dir=tennis_scene/generate/meiji_rgb_v9_repair_clip009/s42-001
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-clip009-repair-v1
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v9_repair_clip009/s42-001
parents:
- run-slcs-meiji-v9-partial-qc-v1
relations: []
tags: [slcs, meiji, quality, provenance, cpu, repair]
---

## 考察 / Findings

### 要約

checkpoint記録が一致しなかった1clipを、既存2D観測・固定PLCS/BLCS重みからCPUで再生成した。
新しい公開前チェックと、実際の媒体・RGB特徴を含む品質監査が成功した。
元のraw出力・公開annotationは削除せず退避し、検証済みの再生成物をMeiji v9へ差し替えた。
差し替え元、退避先、前後のdigestは `promotion.json` に保存した。

### アーキテクチャ詳細

Meiji v9の推論・幾何補正設定は同一。実行deviceだけCPUとし、別dataset/runへ生成した。
`stage=infer` はRGB特徴を生成しないため、同じclip/mediaの既存DINO cacheを修復datasetへコピーし、
manifest・映像digest・特徴仕様を監査した。モデルや重みの再学習、metadataの手修正はしていない。

### メトリクスの解釈

同じ最終正weight maskのball観測1,821件で再投影平均14.9208→4.1927px。
これは観測と擬似教師の整合性であり、独立3D正解に対する精度ではない。
ballの正weightは87.23%、最長unsupported区間は56frame（約0.93秒）。
支持された連続点の最大速度はplayer 8.42m/s、ball 61.46m/sだが、unsupported端点も含めると
51.43m/s、139.52m/sに達する。学習maskと可視化でこの区別を維持する。
新規学習がないため収束曲線は対象外。監査は選択した1clipだけの成功で、全56clip完了ではない。

### アーキテクチャ⇄メトリクスの因果考察

公開前に既に取得済みのpin・producer・raw記録を照合することで、前回の不一致を公開段階で検出できる。
今回一致したことは以前の不整合原因を確定するものではない。ハードウェア原因の探索を進行条件にせず、
指定重みでの再生成とデータ監査により学習に進める状態へ戻した。

### 既存実験との比較

親の部分監査ではこのclipがerror、今回はerror 0。
不整合だった旧結果の数値との優劣比較は行わない。比較対象は今回の同じ観測から得たraw/refinedのみ。
DINO/ViTPoseだけの明示例外は従来どおりで、PLCS/BLCSには広げていない。

### 次に有効な実験

継続中の残clip生成が終わった後、差し替え先で全体品質監査・RGB重畳表示・warm再利用を確認する。
mask対象外の飛びを学習成功と混同せず、SLCS評価では同じmask上の平均値baselineと動的軌跡を比較する。
