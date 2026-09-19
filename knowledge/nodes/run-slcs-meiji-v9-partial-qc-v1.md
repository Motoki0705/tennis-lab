---
id: run-slcs-meiji-v9-partial-qc-v1
type: run
title: 'Meiji v9引継ぎ時の部分教師監査: 1クリップのcheckpoint記録不整合'
provider: codex
date: '2026-09-19'
status: failed
config:
  data: slcs/meiji_rgb_v9
  generation_run: tennis_scene/generate/meiji_rgb_v9/s42-002
  allow_incomplete: true
  device: cpu
metrics:
  completed_clips: 19
  missing_clips: 36
  error_clips: 1
  excluded_clips: 1
  raw_positive_weight_ball_reprojection_mean_px: 14.116881185376313
  refined_positive_weight_ball_reprojection_mean_px: 5.470675908640705
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
repro:
  commit: 767ac67c43c35f65dd294ddf69a0b6a25d519656
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tennis_scene.scripts.report_slcs_dataset_quality
    allow_incomplete=true
    output_dir=tennis_scene/analyze/meiji_rgb_v9_quality/s42-takeover-001
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-partial-qc-v1
  output_dir: outputs/tennis_scene/analyze/meiji_rgb_v9_quality/s42-takeover-001
parents:
- run-slcs-meiji-v9-observation-reuse-v4
relations: []
tags:
- slcs
- meiji
- quality
- provenance
- cpu
---

## 考察 / Findings

### 要約

生成中のMeiji v9をCPU監査し、19clipが完了扱い、36clipは未生成、1clipは理由付き除外、
`video_000/clip_009` はcheckpoint記録不一致で失敗した。`allow_incomplete=true` は
未生成だけを許すため、ジョブ終了コードは1。全体教師の採用・学習開始はしていない。

### アーキテクチャ詳細

通常の品質レポートCLIを使用し、媒体・観測・教師とRGB特徴の由来、支持mask、
同じ最終正weight mask上のraw/refined数値を確認した。モデル推論・最適化はしていない。
監査時点で別の共有queueジョブが後続clipを生成しているため、これは固定時点の部分結果である。

### メトリクスの解釈

成功19clipの正weight ball観測34,253件で、再投影平均はraw 14.1169px、refined 5.4707px。
これは同じ観測から作る擬似教師の整合性であり、独立3D精度ではない。未完成のtest収録を含む
全体データへの一般化も評価していない。新規学習がなく、収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

失敗clipのraw PLCS digestは `78ef9815c8752c8d262cb2340a72c3c46ef53723ae2252d3c9d17bf20ce9f9ca`、
producer側は固定pinの `e851a8fe3fbc8273ed86fcf7876c34e941b77761e4ab2f264495228b5c12919a`。
追加のmetadata照合ではBLCSにも差があり、rawは固定pinの
`dd0e54d296604f43f52fd33ebf53abc4ac71cbc441f86d49cea2d1dd17e0bf32`、producer側は
`d66c43d213b48064ddee210b0b4e8ecdb08a40bd521d5cbd96e77fe40551ec67` だった。
これは記録の不整合を示す。実際のモデル出力異常やハードウェア原因はこの監査だけでは確定しない。
開始前pin確認に加え、既に取得したproducer/raw digestを公開前に照合する必要がある。

### 既存実験との比較

前段の人物観測再利用142cameraの成功を取り消すものではなく、新しい3D教師段階での問題。
DINO/ViTPoseだけの明示例外はPLCS/BLCSに適用されていない。元metadataは書き換えず保全する。

### 次に有効な実験

実行中の残clip生成を継続し、公開前の既存digest間照合を補う。完了後に不整合clipを
保全して再生成し、全体品質レポートを改めて通す。ハードウェア原因の特定を進行条件にしない。
