---
id: group-i983-court-dinov3-shared1024-s42
type: group
task: court_detection
sequence: 44
recorded_at: '2026-10-13'
title: 凍結DINOv3 4サイズと共通後段の比較 (#983)
issue: 983
members:
- run-i983-court-dinov3-s-s42
- run-i983-court-dinov3-splus-s42
- run-i983-court-dinov3-b-s42
- run-i983-court-dinov3-l-s42
parents: []
papers: []
tags:
- issue-983
- colab-l4
- frozen-backbone
- shared-downstream
config:
  backbone_train_mode: frozen
  transformer_dim: 1024
  transformer_depth: 8
  heads: 16
  ffn_dim: 2752
  dpt_channels: 512
  long_side: 512
  batch_size: 8
  mixed_batch_counts:
  - 4
  - 4
  seed: 42
  max_epochs: 20
repro:
  commit: 75e30a565b1f7a5f7fd4f08cea2daf458d377653
  command: .venv/bin/python -m src.tasks.court_detection.scripts.compare_ablation
    --evaluations knowledge/runs/run-i983-court-dinov3-s-s42/evaluation/evaluation.json
    knowledge/runs/run-i983-court-dinov3-splus-s42/evaluation/evaluation.json knowledge/runs/run-i983-court-dinov3-b-s42/evaluation/evaluation.json
    knowledge/runs/run-i983-court-dinov3-l-s42/evaluation/evaluation.json --output
    knowledge/runs/group-i983-court-dinov3-shared1024-s42/comparison
artifacts:
  run_dir: knowledge/runs/group-i983-court-dinov3-shared1024-s42
  report: knowledge/runs/group-i983-court-dinov3-shared1024-s42/comparison/comparison.md
  analysis: knowledge/runs/group-i983-court-dinov3-shared1024-s42/README.md
  provenance: knowledge/runs/group-i983-court-dinov3-shared1024-s42/comparison/provenance.json
  input_evidence: knowledge/runs/group-i983-court-dinov3-shared1024-s42/run.json
  output_dir: gdrive:tennis_lab/outputs/court_detection/report/i983-shared1024/comparison-s42
  manifest: knowledge/runs/group-i983-court-dinov3-shared1024-s42/artifact-manifest.json
relations:
- to: group-i983-court-throughput-s42
  rel: related
---

#983で合意した4条件をすべて20epoch学習し、各val/loss最小checkpointを同じ合成test886画像・実画像val2211画像で評価した。入力版・順序・前処理・学習条件・共通後段と定性サンプルを照合し、[比較表と36パネル](../../runs/group-i983-court-dinov3-shared1024-s42/comparison/comparison.md)、[条件・考察・制約](../../runs/group-i983-court-dinov3-shared1024-s42/README.md)、再現記録を保存した。

この単一seedではBが合成KP/SEG、LがLINE/semantic LINE/pose再投影で最良であり、全出力に共通する一方的な優越は見られなかった。実画像valでは差が小さい。Lの中断再開、選択にも用いた実画像val、Sの2点に支配されたpose平均を明示し、元の指標を置き換えていない。次の検証候補はB/Lの複数seedとpose投影の安定性で、採用モデル変更や追加学習は未実施である。
