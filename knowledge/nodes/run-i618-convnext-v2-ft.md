---
id: run-i618-convnext-v2-ft
type: run
title: i618_convnext_v2_ft
issue: 618
provider: claude
session: 147b5124-0694-4620-bb75-11cb13e410c0
date: '2026-07-07'
status: done
config:
  model: conv_next_unet
metrics:
  precision: 0.607843
  recall: 0.688209
  f1: 0.645535
  mean_distance_px: 2.216319
repro:
  commit: a577e137013287a546a4f3ce793afc6f21a73136
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.ball_detection.scripts.train
    model=conv_next_unet data.heatmap_size='[72,128]' data.batch_size=6 data.num_workers=2
    metrics.nms_kernel=3 training.learning_rate=5e-5 training.trainer.max_epochs=24
    training.checkpoint.monitor=val/f1 training.checkpoint.mode=max training.early_stopping.monitor=val/f1
    training.early_stopping.mode=max training.early_stopping.patience=8 training.early_stopping.min_delta=0.0005
    run.init_weights=ckpt/ball_detection/last.ckpt run.output_dir=outputs/ball_detection/convnext_v2_ft
artifacts:
  run_dir: knowledge/runs/run-i618-convnext-v2-ft
  predictions: knowledge/runs/run-i618-convnext-v2-ft/pred_test.npz
  log: .training_queue/logs/1783405858437554277_1226752_i618_convnext_v2_ft.log
  output_dir: outputs/ball_detection/convnext_v2_ft/logs/version_0
  curves: knowledge/runs/run-i618-convnext-v2-ft/curves.png
  tb_logdir: outputs/ball_detection/convnext_v2_ft/logs/version_0
parents: []
relations:
- to: run-i551-dinov3-rope-tracknet-t3-train
  rel: compares
tags:
- ball_detection
- conv_next_unet
- subpixel
- fine-tune
---

## 考察 / Findings

### 要約

既存 convnext ckpt（Colab 学習, val/loss 早期停止, ep11）を起点に val/f1(max) 監視 + subpixel 精緻化メトリクスで fine-tune した結果、TrackNet test F1 が **0.5764 → 0.7218**（native プロトコル, best ckpt=ep13）へ改善。`ckpt/ball_detection/last.ckpt` として配備し、tennis_clip パイプラインで baseline を全面的に上回った（カバレッジ 79.9→92.0%、3D teleport 消滅）。

### アーキテクチャ詳細

モデルは既存とバイト同一の `conv_next_unet`（mdd 2ch, T=8, dims [64,128,256,512], 出力 72x128 = 入力 288x512 の 1/4 格子）。差分は学習レシピのみ:

- `run.init_weights=ckpt/ball_detection/last.ckpt`（weight-only, optimizer 新規）
- lr 5e-5（既存 1e-4 の半分）+ warmup 200 steps + cosine、bs6（bs8 は WSL2 VRAM 15.1GB で危険域のため）
- **checkpoint/early_stopping の monitor を val/loss(min) → val/f1(max)** に変更（patience 8）
- メトリクスを native 解像度（heatmap [72,128], nms_kernel 3）+ **subpixel_refine（log-parabolic peak 精緻化, #618 で新規実装）** で計測

### メトリクスの解釈

frontmatter の metrics は**学習終了時の last-epoch (ep21) での Lightning test 値**（f1 0.6455）。配備したのは val/f1 ベストの **ep13 ckpt** で、manifest 評価（outputs/ball_detection/evaluation_subpixel_ab/native）では **test F1 0.7218 / precision 0.7357 / recall 0.7084 / dist 2.18px**。last-epoch と best-ckpt の差（0.65 vs 0.72）が大きく、val/f1 は epoch 間で ±0.1 揺れる（curves.png 参照）。旧 ckpt の同プロトコル値は 0.5764、subpixel 無し legacy 公表値は 0.4111。

負例 FPR は test 0.0000 → 0.0877 と微増（recall 押し上げの代償）。実クリップでは stride4+max_score 構成でも >30px jump が 0 になったため、実害は観測されていない。

### アーキテクチャ⇄メトリクスの因果考察

- 旧 ckpt の F1=0.41 は検出能力ではなく **72x128 格子量子化に律速**されていた（原画像で ~10px 刻み、4px 閾値下では完璧な検出器でも F1≈0.5 が上限）。subpixel 精緻化だけで 0.41→0.58 に跳ねたことがこれを裏付ける。
- fine-tune の +0.14（0.58→0.72）は (1) val/f1 での ckpt 選択（val/loss は f1 と乖離することが i551/i579 でも既知）、(2) 22ep の追加学習、の複合。寄与分離はしていない（仮説: ckpt 選択の寄与が大きい。epoch 間の f1 揺れ幅 ±0.1 が根拠）。
- 学習初期 (ep0-3) は val/f1 0.33-0.45 と一時的に大きく退行しており、lr 5e-5 でも既存最適点から一度離れてから回復している。

### 既存実験との比較

- dinov3_rope 系（[[run-i551-dinov3-rope-tracknet-t3-train]] f1=0.094, i579 staged 最良 0.106）とは桁違い。frozen backbone 構成より、TrackNet に直接 fit した conv U-Net + 適切な評価プロトコルが圧倒的に優位。
- 原 convnext ckpt（knowledge ノード無し, Colab 産）が実質の parent。同一アーキ・同一データで、監視指標と評価プロトコルの改善のみでこの差が出た。

### 次に有効な実験

1. scratch 60ep 版（i618_convnext_v2_scratch, キュー実行中）が ft を超えるか
2. 負例 FPR 微増の実クリップ長期影響（occlusion 区間での誤検出）確認
3. 学習時から subpixel decode を意識した loss（DARK 式 distribution-aware 学習）や、heatmap 288x512 版（resize-logits 学習）との比較
4. Web/YouTube データ混合（broadcast ドメイン適応）— i579 の「マルチフレームでは Web 混合が逆効果」の再検証
