---
id: run-person-model-speed-low-precision
type: run
task: tennis_scene
sequence: 86
recorded_at: '2026-11-25'
title: Meiji 4camの現行人物モデルPT16・TRT16速度再計測
issue: 2496
provider: codex
date: '2026-11-25'
status: done
summary_impact: updated
config:
  gpu: NVIDIA GeForce RTX 5060 Ti
  driver: '621.24'
  cuda: '13.0'
  pytorch: 2.13.0+cu130
  tensorrt: 10.14.1.48.post1
  pytorch_matmul_and_cudnn_fp32_precision: ieee
  input_id: meiji_4cam/person-model-speed-v1
  input_manifest_sha256: 64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62
  timing_protocol_sha256: 39e362cb87c7c373f8c1ad25e5e28025aab89b1467be21ca3139c2fb456136a2
  warmup: 30
  measured_blocks: 3
  repetitions_per_block: 100
  parity_status: not_evaluated
metrics:
  completed_conditions: 22
  infeasible_conditions: 0
  timed_repetitions: 6600
  verified_input_files: 728
  matched_asset_inventories: 22
  wall_p50_change_median_percent: -1.564
repro:
  commit: 13e508cb78c46663fbe58ad8be03bcf9c96816a6
  command: >-
    env PYTHONPATH="$PWD" TENNIS_LAB_DINO_EXTENSION="$TENNIS_LAB_DINO_EXTENSION"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    .venv/bin/python -B -u knowledge/runs/run-person-model-speed-baseline/run.py
    --output "$PERSON_RUN_OUTPUT" --input-dir "$PERSON_FIXED_INPUTS"
    --pose-engine-run "$PERSON_POSE_ENGINE_RUN" --trt-root "$PERSON_TRT_ROOT"
    --external-assets "$PERSON_EXTERNAL_ASSETS" suite "$PERSON_BASELINE_GROUP"
artifacts:
  run_dir: knowledge/runs/run-person-model-speed-low-precision
  case_table: knowledge/runs/run-person-model-speed-low-precision/cases.csv
  case_records: knowledge/runs/run-person-model-speed-low-precision/cases.json
  timing_comparison: knowledge/runs/run-person-model-speed-low-precision/timing-comparison.csv
  queue_receipts: knowledge/runs/run-person-model-speed-low-precision/queue-receipts.json
  output_dir: .worktree-outputs/parent-2492-issue-2496-primary/person-model-speed-low-precision
parents:
- run-person-model-speed-baseline
relations:
- to: run-person-model-speed-baseline
  rel: compares
papers: []
tags: []
---

## 考察 / Findings

DINO 4-scale Swin-L、ViTPose-Hのflipあり・なし、HMR2特徴＋GVHMRをPT16・TRT16で再計測した。RTX 5060 Tiの排他queue 3件が完了し、22条件で各300反復を取得した。全条件の計時状態は `ok`、parityは `not_evaluated`。OOM・変換失敗・TensorRT context生成失敗は0件だった。固定入力728ファイルと22条件の各反復のID順、同条件の重み・planのSHA-256を初回の記録と照合した。 [全条件の速度・VRAM](../../runs/run-person-model-speed-low-precision/cases.csv) · [環境・条件別の証拠](../../runs/run-person-model-speed-low-precision/cases.json)

### Batch 1の速度とVRAM

GPUはモデルforwardのCUDA event、wallは前後処理・転送・バッチ終端の同期を含む。items/sはDINOで画像、ViTPoseで人物crop、身体でperson-frameを数える。VRAMはPyTorchのallocated peak、reserved peak、NVMLのデバイス全体のサンプル最大値。Δは初回からの実測差（再計測 − 初回）。 [全batchの初回・再計測の照合](../../runs/run-person-model-speed-low-precision/timing-comparison.csv)

| モデル | backend | GPU P50 / P95 ms | wall P50 / P95 ms | items/s | torch A / R / NVML MiB | wall P50 Δ ms |
|---|---|---:|---:|---:|---:|---:|
| DINO | PT16 | 146.47 / 149.93 | 165.69 / 170.71 | 6.02 | 2102 / 2366 / 2806 | -1.99 |
| DINO | TRT16 | 75.12 / 75.59 | 94.79 / 101.53 | 10.47 | 12 / 14 / 1384 | +0.02 |
| ViTPose-H flipあり | PT16 | 18.00 / 24.12 | 22.17 / 28.72 | 43.13 | 3764 / 3950 / 4388 | -2.34 |
| ViTPose-H flipあり | TRT16 | 9.28 / 9.97 | 12.64 / 13.47 | 78.12 | 2 / 2 / 1752 | -0.25 |
| ViTPose-H flipなし | PT16 | 15.79 / 21.86 | 19.43 / 25.97 | 49.66 | 3755 / 3944 / 4382 | -2.63 |
| ViTPose-H flipなし | TRT16 | 5.66 / 6.80 | 8.89 / 10.18 | 108.86 | 1 / 2 / 1694 | -0.26 |
| HMR2特徴＋GVHMR | PT16 | 140.60 / 144.34 | 492.28 / 558.10 | 31.82 | 4338 / 4484 / 4926 | -17.38 |
| HMR2特徴＋GVHMR | TRT16 | 57.73 / 58.64 | 408.72 / 436.20 | 38.85 | 50 / 54 / 1994 | -18.20 |

### 初回と先行runとの照合

- 同じ22条件のwall P50は初回比で−11.91%〜+0.02%、条件間の中央値は−1.56%だった。wall P95は18条件で短く、4条件で長かった。初回にparity不成立だった11条件は初回の生データから時刻の統計量を復元した。再計測はバッチ終端を同期し、初回からの差には実行ごとの変動も含まれる。初回のVRAMはparityまで、再計測のVRAMは最後の計時ブロックまでを含む。 [初回記録](000085-run-person-model-speed-baseline.md) · [対応する計時値](../../runs/run-person-model-speed-low-precision/timing-comparison.csv)
- DINOの同期込みforward P50はB=1でPT16 146.56 ms、TRT16 75.21 msだった。#2336の同じ750×1333の動的profile計測は150.43 ms、74.98 ms。#2304のTRT16 56.98 msは静的profile、別入力で得た値。 [batch掃引](000080-run-i2336-vision-batch-sweep.md) · [静的plan](000078-run-i2304-dino-tensorrt.md)
- ViTPose-HのflipありTRT16はB=4の同期込みforward P50が28.77 msで、#2336の28.64 msに近い。B=1ではPT16 18.04 ms、TRT16 9.32 msで、#2303では27.197 ms、8.468 msだった。B=1同士でも固定crop集合と実行時設定が異なる。 [batch掃引](000080-run-i2336-vision-batch-sweep.md) · [先行の単体計測](000077-run-i2303-vit-tensorrt.md)
- 身体B=1のHMR2（内部16 crop）は同期込みforward P50がPT16 124.94 ms、TRT16 55.18 ms。#2336では145.81 ms、55.07 msだった。身体のwallには同じ反復のGVHMR推論とカメラ補正・IKも含む。 [HMR2単体batch掃引](000080-run-i2336-vision-batch-sweep.md)

新規の検出・姿勢の比較図は、疑似GTで実測した精度とこの規約の速度を用いる。身体の表は速度の基準値を示す。PT32とDINO＋ViTPose-H合計の速度は、修正前の計時プログラムで得た暫定値として初回ノードに保存した。 [初回の暫定値とparity](000085-run-person-model-speed-baseline.md) · [再現手順と生データ](../../runs/run-person-model-speed-low-precision/README.md)
