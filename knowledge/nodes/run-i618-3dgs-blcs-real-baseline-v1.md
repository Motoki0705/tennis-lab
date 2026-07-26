---
id: run-i618-3dgs-blcs-real-baseline-v1
type: run
title: 3DGS×BLCS 固定実データ配備モデル baseline v1
issue: 618
provider: codex
date: '2026-07-25'
status: done
config:
  model: conv_next_unet
  checkpoint: ckpt/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt
  checkpoint_sha256: cd7927ad27e53ddd6aa77df28eca3c5e674552461ccda083a41e99e629857892
  loss: evaluation-only
  data: TrackNet game9 val / game10 test
  heatmap_size:
  - 72
  - 128
  peak_threshold: 0.5
  ball_distance_threshold_px: 4.0
  nms_kernel: 3
  subpixel_refine: true
metrics:
  val_f1: 0.7126280665397644
  val_precision: 0.7250000238418579
  val_recall: 0.7006711363792419
  val_mean_distance_px: 2.3024067878723145
  val_negative_frame_fpr: 0.11666666666666667
  test_f1: 0.7217894196510315
  test_precision: 0.7356557250022888
  test_recall: 0.708436131477356
  test_mean_distance_px: 2.1762075424194336
  test_negative_frame_fpr: 0.08771929824561403
repro:
  commit: ac9e640903a6dfaecb65fc980f5dcf408bbcd589
  branch: main
  command: .venv/bin/python -m src.tasks.ball_detection.scripts.evaluate_manifest
    manifest_path=src/tasks/ball_detection/configs/evaluation/3dgs_blcs_real_v1.yaml
  manifest_sha256: 3b91d9e5be6a343794c5e7f01ed50dfd7e69a78558a9e6860ac74b169a3721e0
  evaluator_code_sha256: a0e39a4a1dd6df74a46e4da65c8ca89385ec6558e62d42b2e6473befc868685e
  tracked_diff_sha256: 5acfcd41030379b80b15567779c155dd6a260c06f2f947aafe6a919407c8eec6
  untracked_tree_sha256: 77744c67582968809e92cd684fa61b56d35f2ad613a968e83aa03141d5b8203b
artifacts:
  log: .codex-loop/logs/C02-baseline-evaluation.log
  output_dir: outputs/ball_detection/3dgs_blcs/evaluation_real_v1
  summary: outputs/ball_detection/3dgs_blcs/evaluation_real_v1/summary.json
  comparison: outputs/ball_detection/3dgs_blcs/evaluation_real_v1/comparison.csv
  provenance: outputs/ball_detection/3dgs_blcs/evaluation_real_v1/run_provenance.json
parents:
- run-i618-convnext-v2-ft
relations: []
tags:
- ball_detection
- 3dgs-blcs
- real-baseline
- fixed-evaluation
---

## 考察 / Findings

### 要約

配備済み `run-i618-convnext-v2-ft-epoch13.ckpt` を凍結済み TrackNet
game9/game10 protocol で再評価し、val F1 **0.712628**、test F1
**0.721789**を再現した。履歴値との差は全主要 metric で実質 0 であり、
3DGS synthetic treatment の比較基準として固定できる。

### アーキテクチャ詳細

学習は行わない evaluation-only run。親ノード
`run-i618-convnext-v2-ft` の ConvNeXt U-Net checkpoint を byte hash で固定し、
72x128 heatmap、peak threshold 0.5、NMS kernel 3、原画像 4 px matching、
subpixel refine を全 split で共通使用した。game9 は validation 専用、
game10 は final test 専用で、games 1--8 だけが将来の実データ学習集合である。

### メトリクスの解釈

game9 は precision/recall/F1 = 0.725000/0.700671/0.712628、平均距離
2.302407 px。3,040 frame 中 negative 60、false-positive negative 7
（FPR 0.116667）。game10 は 0.735656/0.708436/0.721789、2.176208 px。
4,168 frame 中 negative 114、false-positive negative 10（FPR 0.087719）。
GPU throughput は val 205.45 fps、test 201.20 fps、peak allocated VRAM は
約 1,370 MiB。

### アーキテクチャ⇄メトリクスの因果考察

今回の目的は因果改善ではなく再現性確認である。同一 checkpoint、split、
decode/matching protocol で履歴値と一致したため、後続 synthetic treatment
の差を評価器やデータ split の drift と誤認するリスクを減らせた。推測を
含む新しい性能因果は主張しない。

### 既存実験との比較

親 `run-i618-convnext-v2-ft` に記録された配備 checkpoint の native
protocol test 値（F1 0.7218、precision 0.7357、recall 0.7084、距離
2.18 px）を完全に確認した。履歴と今回の val/test 数値に観測上の差はない。

### 次に有効な実験

game9 だけで synthetic mixing/checkpoint を選択し、game10 は最終 A/B
まで設定選択へ使わない。control/treatment 3 seed の最終比較前に、同一
評価器へ immutable per-frame/per-clip trace を追加し、paired bootstrap
と negative-frame regression を算出可能にする。
