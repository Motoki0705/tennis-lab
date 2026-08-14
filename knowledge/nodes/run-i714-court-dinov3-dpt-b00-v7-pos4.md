---
id: run-i714-court-dinov3-dpt-b00-v7-pos4
type: run
title: DINOv3+DPT B00 KP7 v7（positive weight 4）
issue: 714
provider: codex
session: 019ff617-dfb3-7182-8c2a-1d0705cf3ff0
date: '2026-08-14'
status: failed
config:
  model: dinov3 + dpt + lora
  loss: focal BCE, kp positive_weight 4.0
  data: synthetic_court B00 KP7, default augmentation, rescaled crop, train short-side
    256, validation short-side 512
  max_epochs: 50
  seed: 714
metrics:
  best_val_kp_mean_dist_px: 159.72650146484375
  final_val_kp_mean_dist_px: 261.1883544921875
  held_out_test_kp_mean_dist_px: 158.88372113657954
  held_out_test_kp_mean_dist_normalized: 0.1522374183936796
  held_out_test_median_px: 85.97591742155629
  held_out_test_p90_px: 437.29696506866753
  held_out_test_fraction_gt_40px: 0.5843960990247562
repro:
  commit: 36d6bc303b33895261ae0bf741b34c31ba376a4a
  branch: feat/issue-714-court-data-composition-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.court_detection.scripts.train data/source=synthetic_court data/processing=kp
    model/encoder=dinov3 model/decoder=dpt training=lora loss.kp.positive_weight=4.0
    data.batch_size=4 data.num_workers=4 data.augmentation.train_scales='[256]' data.augmentation.val_short_side=512
    training.trainer.max_epochs=50 training.checkpoint.monitor=val/kp_mean_dist training.checkpoint.mode=min
    training.early_stopping.monitor=val/kp_mean_dist training.early_stopping.mode=min
    training.early_stopping.patience=10 training.qualitative_logging.enabled=false
    run.seed=714 run.output_dir=issue714/i714-court-dinov3-dpt-b00-v7-pos4 run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i714-court-dinov3-dpt-b00-v7-pos4
  predictions: knowledge/runs/run-i714-court-dinov3-dpt-b00-v7-pos4/pred_test.npz
  log: .training_queue/logs/1786661253604227440_1360691_i714-court-dinov3-dpt-b00-v7-pos4.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue714/i714-court-dinov3-dpt-b00-v7-pos4/logs/version_0
  diagnostic: .codex/tasks/issue-714/logs/v7-test-prediction-diagnostic.json
  curves: knowledge/runs/run-i714-court-dinov3-dpt-b00-v7-pos4/curves.png
  tb_logdir: outputs/issue714/i714-court-dinov3-dpt-b00-v7-pos4/logs/version_0
parents:
- run-i714-court-dinov3-dpt-b00-v6-rescaled-crop
relations: []
tags:
- court-detection
- synthetic-court
- b00
- dinov3
- dpt
- kp7
- rescaled-crop
- positive-weight-4
- negative
---

## 考察 / Findings

### 要約

v6と同じDINOv3+DPT、default augmentation、rescaled crop条件で、KP Gaussian正例の
`positive_weight`だけを1から4へ変更した。best validationは159.7265 px、held-out testは
158.8837 pxで32/40 pxの合格条件を満たさず、v6からも大幅に悪化したため不採用とする。

### アーキテクチャ詳細

DINOv3 ViT-B/16 + DPT + rank-8 LoRA、synthetic semantic KP7、batch size 4、train
short-side 256、validation short-side 512、seed 714を使用した。best checkpointはepoch 12で、
early stoppingによりepoch 22まで実行し、278 test samplesを評価した。

### メトリクスの解釈

best/final validationは159.7265/261.1884 px、held-out testは158.8837 px
（normalized 0.152237）だった。保存predictionからの再計算は158.6907 px、median
85.9759 px、p90 437.2970 px、40 px超58.44%である。train側のbest距離も57.1803 pxに
留まり、v6のような小さいmedianを再現できなかった。

### アーキテクチャ⇄メトリクスの因果考察

v6比でbest validationは57.8170 px、testは51.4172 px悪化した。全1,946
sample-channelで4 peakすべてがthresholdを超えた一方、semantic channel別meanは
126.23–176.56 px、court slot別meanは151.8005/163.8961 pxだった。特定channelや第二courtの
collapseではなく、正例強調によって低品質peakまで一様に受理され、位置局在そのものが崩れた
という観測に整合する。因果は一変数比較を根拠とするが、heatmap幅の直接観測ではないため
機序の断定はしない。

### 既存実験との比較

v6のbest validation 101.9095 px、test 107.4665 px、median 11.9733 px、40 px超
36.08%の全指標を下回った。`positive_weight=4`はbackground優勢を補うという仮説を支持せず、
positive weightをさらに上げる実験は停止する。

### 次に有効な実験

最良の現行baselineであるv6へ戻し、`positive_weight=1`を維持したままtrain short-sideだけを
256から512へ変更する。trainとvalidationの解像度契約を揃え、v6で残ったview/coverage依存の
long tailが縮小するかを同じvalidation/test閾値で検証する。
