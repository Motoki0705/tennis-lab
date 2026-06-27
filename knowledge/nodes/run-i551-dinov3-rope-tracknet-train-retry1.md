---
id: run-i551-dinov3-rope-tracknet-train-retry1
type: run
title: i551_dinov3_rope_tracknet_train_retry1
issue: 551
provider: codex
session: 019eed5d-33b5-7720-849c-b6ea3e0e9d30
date: '2026-06-26'
status: done
config:
  model: dinov3_rope
  data: rgb_sequence
metrics:
  f1: 0.080299
  loss: 0.000871
  mean_distance_px: 2.752451
  precision: 0.078611
  recall: 0.082061
repro:
  commit: ed6eef1b4fbfa5431d1ce40f3a010e90115b09fb
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.ball_detection.scripts.train
    model=dinov3_rope data=rgb_sequence run.output_dir=outputs/ball_detection/dinov3_rope_tracknet
    && .venv/bin/python -c "from pathlib import Path; import sys; root=Path(\"outputs/ball_detection/dinov3_rope_tracknet\");
    ckpts=sorted(root.glob(\"logs/version_*/checkpoints/last.ckpt\"), key=lambda p:
    p.stat().st_mtime); ckpts or sys.exit(f\"No last.ckpt found under {root}\"); ckpt=ckpts[-1];
    target=root/\"latest\"/\"checkpoints\"/\"last.ckpt\"; target.parent.mkdir(parents=True,
    exist_ok=True); target.unlink() if target.exists() or target.is_symlink() else
    None; target.symlink_to(ckpt.resolve()); print(f\"latest checkpoint -> {ckpt}\")"'
artifacts:
  run_dir: knowledge/runs/run-i551-dinov3-rope-tracknet-train-retry1
  log: .training_queue/logs/1782464653858008549_1862554_i551_dinov3_rope_tracknet_train_retry1.log
  output_dir: outputs/ball_detection/dinov3_rope_tracknet/logs/version_1
  curves: knowledge/runs/run-i551-dinov3-rope-tracknet-train-retry1/curves.png
  tb_logdir: outputs/ball_detection/dinov3_rope_tracknet/logs/version_1
parents: []
relations: []
tags:
- ball_detection
- dinov3
- rope
- tracknet
- temporal-detector
- num_frames_1
---

## 考察 / Findings

### 要約
Issue #551 で実装した DINOv3 ViT-B/16 + 3 軸 RoPE bidirectional Transformer 検出器を `num_frames=1`（静止画）で tracknet 上に学習した最初の baseline run。loss は収束するが test/f1=0.080 と非常に低く、現状の構成（frozen backbone・20ep・focal_bce）では peak 検出品質が不足することを示す。

### アーキテクチャ詳細
- model=`dinov3_rope`, data=`rgb_sequence`, `num_frames=1`。入力 `(B,1,3,288,512)` → frame-wise DINOv3 ViT-B/16 patch token `(18,32)` → decoder(dim=256, layers=4, heads=8, ffn=1024, rope_dim=32, rope_base=10000×3軸) → heatmap head で `288×512` 復元 → logits `(B,1,1,288,512)`。
- backbone は `train_mode=frozen`（`last_n_blocks=0`）で完全凍結。学習対象は decoder + heatmap head のみ。
- loss=`focal_bce`(gamma=2.0)、metrics は peak_threshold=0.5 / ball_distance_threshold=4.0px / nms_kernel=9。
- training: 20ep, AdamW lr=1e-4 wd=1e-4 warmup=200, precision=bf16-mixed, batch=4, sample_stride=4, val/loss monitor + early_stopping patience=5。

### メトリクスの解釈
- test: f1=0.080, precision=0.079, recall=0.082, mean_distance_px=2.75, loss=0.00087。
- val/f1 は 20ep を通じて 0.06–0.08 で頭打ち。train/f1≈0.23 に対し val/f1≈0.06 で学習-検証ギャップがあり、低水準ながら過学習傾向もある。
- loss が極端に小さいのは focal_bce が背景優位な heatmap で支配されるため。loss 収束と検出品質（f1）が乖離している。
- mean_distance_px≈2.75 は threshold 4.0px 内で、検出できた時の位置精度は悪くない。問題は recall が低く大半のフレームで peak が 0.5 を超えないこと。

### アーキテクチャ⇄メトリクスの因果考察
- 仮説: frozen DINOv3 backbone + 浅い decoder(4層) + 20ep では、heatmap peak を 0.5 閾値まで鋭く立てるだけの容量・学習量が不足。loss は下がるが peak が鈍く recall が伸びない。
- 仮説: focal_bce の背景支配により勾配が peak 強調に十分回らず、precision/recall がともに 0.08 付近に留まる。閾値・loss 重み・backbone 解凍（`last_n_blocks>0`）が改善余地。

### 既存実験との比較
- ball_detection の検出器を knowledge graph に登記した最初の run のため、直接比較できる先行ノードは無い（i524 系は court segmentation で別タスク）。
- 同一アーキで `num_frames=3` にした [[run-i551-dinov3-rope-tracknet-t3-train]] が f1=0.094 と僅かに上回り、時系列文脈が小幅に効くことを示す。

### 次に有効な実験
- backbone 部分解凍（`last_n_blocks=2〜4` または `full`）と学習エポック増で容量・適合を底上げ。
- focal_bce の loss 重み / peak_threshold の見直し、sigma_ratio 調整で peak を鋭くする。
- まず Issue #553 の評価パイプラインで pred_test を取得し、定量比較の基盤を整える（本 run は eval pipeline 整備前のため pred_test.npz 無し）。
