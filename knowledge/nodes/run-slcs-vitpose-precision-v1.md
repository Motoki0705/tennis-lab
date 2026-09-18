---
id: run-slcs-vitpose-precision-v1
type: run
title: 'ViTPose: 同一人物cropのfloat32/bfloat16比較'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: ViTPose-H float32 vs bfloat16
  loss: inference only
  data: Meiji clip000 four person crops x 128 frames
metrics:
  timing/speedup: 2.174037611385163
  difference/max_series_p95_px: 0.17132039368152618
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m scripts.analysis.benchmark_vitpose_precision --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --observations /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-001/video_000/clip_000
    --checkpoint /home/kamimura/projects/tennis-lab/third_party/GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
    --output /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/vitpose_precision/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-vitpose-precision-v1
  output_dir: outputs/tennis_scene/evaluate/vitpose_precision/s42-001
parents: []
relations: []
tags:
- slcs
- real-rgb
---

## 考察 / Findings

### 要約
4人物crop系列×128フレームでbfloat16を比較。平均的な座標差が小さく、処理時間を短縮できたため採用した。

### アーキテクチャ詳細
同じViTPose-H重み・bbox・flip testでautocast精度だけ変更。heatmapはfloat32へ戻してUDP decoderへ入力。

### メトリクスの解釈
float32合計56.393秒、bfloat16 25.939秒（2.17倍）。系列別座標差95%点の最大0.171px。最大差29.272pxの例外があり、全点等価ではない。同時GPU負荷があるため速度は参考値。

### アーキテクチャ⇄メトリクスの因果考察
仮説: 大きな例外は近いheatmap peak間の選択差。モデル精度差の評価ではなく数値精度変更による出力安定性検査である。

### 既存実験との比較
同一モデル・同一入力を比較し、異なるクリップ間の精度差を混ぜていない。

### 次に有効な実験
bfloat16を明示した別キャッシュで人物観測を生成し、再投影・人物対応の検査を継続する。
