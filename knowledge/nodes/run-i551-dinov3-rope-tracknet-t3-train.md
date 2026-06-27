---
id: run-i551-dinov3-rope-tracknet-t3-train
type: run
title: i551_dinov3_rope_tracknet_t3_train
issue: 551
provider: codex
session: 019f0329-2f83-7e22-83f8-10d706257993
date: '2026-06-26'
status: done
config:
  model: dinov3_rope
  data: rgb_sequence
metrics:
  f1: 0.094238
  loss: 0.000928
  mean_distance_px: 2.566375
  precision: 0.084435
  recall: 0.106615
repro:
  commit: ed6eef1b4fbfa5431d1ce40f3a010e90115b09fb
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.ball_detection.scripts.train
    model=dinov3_rope data=rgb_sequence model.num_frames=3 run.output_dir=outputs/ball_detection/dinov3_rope_tracknet_t3
    && .venv/bin/python -c "from pathlib import Path; import sys; root=Path(\"outputs/ball_detection/dinov3_rope_tracknet_t3\");
    ckpts=sorted(root.glob(\"logs/version_*/checkpoints/last.ckpt\"), key=lambda p:
    p.stat().st_mtime); ckpts or sys.exit(f\"No last.ckpt found under {root}\"); ckpt=ckpts[-1];
    target=root/\"latest\"/\"checkpoints\"/\"last.ckpt\"; target.parent.mkdir(parents=True,
    exist_ok=True); target.unlink() if target.exists() or target.is_symlink() else
    None; target.symlink_to(ckpt.resolve()); print(f\"latest checkpoint -> {ckpt}\")"'
artifacts:
  run_dir: knowledge/runs/run-i551-dinov3-rope-tracknet-t3-train
  log: .training_queue/logs/1782482830449695828_1983238_i551_dinov3_rope_tracknet_t3_train.log
  output_dir: outputs/ball_detection/dinov3_rope_tracknet_t3/logs/version_0
  curves: knowledge/runs/run-i551-dinov3-rope-tracknet-t3-train/curves.png
  tb_logdir: outputs/ball_detection/dinov3_rope_tracknet_t3/logs/version_0
parents:
- run-i551-dinov3-rope-tracknet-train-retry1
relations:
- to: run-i551-dinov3-rope-tracknet-train-retry1
  rel: compares
tags:
- ball_detection
- dinov3
- rope
- tracknet
- temporal-detector
- num_frames_3
---

## 考察 / Findings

### 要約
[[run-i551-dinov3-rope-tracknet-train-retry1]] と同一アーキ・同一 config で `num_frames=3` にした時系列版 run。Phase 1 (`T=1`) から `T>1` への移行が同一パラメータで成立し、test/f1=0.094 と T=1(0.080) を僅かに上回る。時系列文脈が小幅に効くことを示すが、絶対水準は依然低い。

### アーキテクチャ詳細
- model=`dinov3_rope`, data=`rgb_sequence`, `num_frames=3`。入力 `(B,3,3,288,512)` → frame-wise DINOv3 → patch token を flatten し decoder の Q/K に `(time,y,x)` 3 軸 RoPE 適用 → logits `(B,1,3,288,512)`。
- T=1 run との差分は `model.num_frames=1→3` と output_dir のみ（config 差分はこの 2 点だけ）。backbone frozen・decoder(4層/8head/256dim)・focal_bce・20ep・batch=4 は同一。

### メトリクスの解釈
- test: f1=0.094, precision=0.084, recall=0.107, mean_distance_px=2.57, loss=0.00093。
- T=1 比で recall(0.082→0.107) と mean_distance_px(2.75→2.57) が改善、precision は微増(0.079→0.084)。
- val/f1 は 0.07 前後で頭打ち、train/f1≈0.20。T=1 と同様に loss 収束と f1 が乖離し、絶対水準は低いまま。

### アーキテクチャ⇄メトリクスの因果考察
- 仮説: 3 フレーム文脈と 3 軸 RoPE により時間方向の手掛かりが加わり、ボールの軌跡上で peak がやや立てやすくなって recall が伸びた。
- ただし改善幅は小さく、frozen backbone・浅い decoder・短い学習という T=1 と共通のボトルネックが支配的と考えられる。

### 既存実験との比較
- [[run-i551-dinov3-rope-tracknet-train-retry1]] (`T=1`) に対し全 detection 指標で僅かに優位。同一重み構造で `T=1`→`T=3` がロードできる checkpoint 契約（受け入れ条件）が実運用上も機能していることを確認。

### 次に有効な実験
- num_frames をさらに増やす（T=5/8）前に、backbone 解凍・エポック増・loss/閾値調整で絶対水準を引き上げる方が費用対効果が高い。
- Issue #553 の評価パイプラインで pred_test を取得し、T=1 と T=3 を同一プロトコルで定量比較する。
