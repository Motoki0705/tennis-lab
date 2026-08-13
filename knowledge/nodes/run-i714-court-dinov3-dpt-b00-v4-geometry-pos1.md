---
id: run-i714-court-dinov3-dpt-b00-v4-geometry-pos1
type: run
title: DINOv3+DPT B00 KP7 v4（aspect ratio修正・positive weight 1）
issue: 714
provider: codex
session: 019ff617-dfb3-7182-8c2a-1d0705cf3ff0
date: '2026-08-13'
status: failed
config:
  model: dinov3 + dpt + lora
  loss: focal BCE, kp positive_weight 1.0
  data: synthetic_court B00 KP7, default augmentation, train short-side 256, validation
    short-side 512
  max_epochs: 50
  seed: 714
metrics:
  best_val_kp_mean_dist_px: 118.25890350341797
  final_val_kp_mean_dist_px: 118.98297882080078
  held_out_test_kp_mean_dist_px: 116.70930938253137
  held_out_test_kp_mean_dist_normalized: 0.11182721464174947
  held_out_test_median_px: 17.560386657714844
  held_out_test_p90_px: 382.9532043457037
  held_out_test_fraction_gt_40px: 0.423105776444111
repro:
  commit: 18bac72eac5954a95db79f55d26cea250ddbad9c
  branch: feat/issue-714-court-data-composition-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.court_detection.scripts.train data/source=synthetic_court data/processing=kp
    model/encoder=dinov3 model/decoder=dpt training=lora loss.kp.positive_weight=1.0
    data.batch_size=4 data.num_workers=4 data.augmentation.train_scales='[256]' data.augmentation.val_short_side=512
    training.trainer.max_epochs=50 training.checkpoint.monitor=val/kp_mean_dist training.checkpoint.mode=min
    training.early_stopping.monitor=val/kp_mean_dist training.early_stopping.mode=min
    training.early_stopping.patience=10 training.qualitative_logging.enabled=false
    run.seed=714 run.output_dir=issue714/i714-court-dinov3-dpt-b00-v4-geometry-pos1
    run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i714-court-dinov3-dpt-b00-v4-geometry-pos1
  predictions: knowledge/runs/run-i714-court-dinov3-dpt-b00-v4-geometry-pos1/pred_test.npz
  log: .training_queue/logs/1786613628379200254_618811_i714-court-dinov3-dpt-b00-v4-geometry-pos1.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue714/i714-court-dinov3-dpt-b00-v4-geometry-pos1/logs/version_0
  diagnostic: .codex/tasks/issue-714/logs/v4-test-prediction-diagnostic.json
  curves: knowledge/runs/run-i714-court-dinov3-dpt-b00-v4-geometry-pos1/curves.png
  tb_logdir: outputs/issue714/i714-court-dinov3-dpt-b00-v4-geometry-pos1/logs/version_0
parents:
- run-i714-court-dinov3-dpt-b00-v3-pos32
relations: []
tags:
- court-detection
- synthetic-court
- b00
- dinov3
- dpt
- kp7
- aspect-ratio
- negative
---

## 考察 / Findings

### 要約

v3のaspect-ratio保存geometry、DINOv3 + DPT + LoRA、full B00、seed 714を固定し、
KP focal BCEの`positive_weight`だけを32から1へ戻した。50 epochを完走してheld-out test
predictionを保存できたが、best validation 118.2589 px、test 116.7093 pxで、事前定義した
32/40 pxの合格条件を満たさなかったため不採用とする。

### アーキテクチャ詳細

DINOv3 ViT-B/16の4層（blocks 2/5/8/11）をDPT decoderへ入力し、rank 8、alpha 16の
LoRAで学習した。synthetic semantic KP7、batch size 4、default augmentation、train
short-side 256、validation short-side 512、最大50 epoch、seed 714である。best checkpointは
epoch 43で、修正済みtrusted loaderによりそのcheckpointから278 test samplesを評価した。

### メトリクスの解釈

best/final validation距離は118.2589/118.9830 px、held-out testは116.7093 px
（normalized 0.111827）だった。保存値をproduction metricと同じnearest-peak規則で再計算した
値は116.7586 pxで、保存されたnormalized peak座標の有限精度による0.0493 pxの小差に収まる。testのmedianは17.5604 px
だがp90は382.9532 px、40 px超は42.31%であり、少数の大誤差が平均を強く押し上げている。

### アーキテクチャ⇄メトリクスの因果考察

全7 semantic channelと両court slotでpeak欠落は0で、court slot別meanも117.6634/
116.0751 pxだった。unique assignment診断でも両slotは129.6794/128.6715 pxと同程度である。
したがって単一classや第二courtの系統的collapseではない。`complex_center`視点のmean
150.3588 px、40 px超51.63%に対し、`far_baseline`は71.3193 px、29.71%であり、failureは
view-conditioned long tailである。train loss/距離が約0.001/24.20 pxまで下がる一方、
validationは118.98 pxなので、default augmentationの強いcrop・affine・perspectiveと
validation camera分布のずれによる汎化不良が有力な仮説である。

### 既存実験との比較

positive weight 32のv3（best 186.4902 px）から68.2313 px改善し、32倍weightが過補償だった
という仮説を支持した。一方、旧square geometry・256 px validationのv2（42.7476 px）より
悪い。v2とはgeometryと評価解像度が異なるため単純比較はできないが、aspect-ratio保存だけで
long tailを解消できず、v4のdefault augmentationが次の分離対象になった。

### 次に有効な実験

v4のdata、model、loss、seed、train/validation解像度を固定し、augmentation presetだけを
`default`から`light`へ変更するv5を実行する。crop scale 0.2–1.0、25度回転、18%移動、
0.65–1.5 affine scale、18度shear、強いperspective/photometric変換を緩和し、視点依存の
long tailが縮むかを同じvalidation/test契約で判定する。
