---
id: run-court-scene-diversity-colab-scene-3-s42
type: run
task: court_detection
sequence: 53
recorded_at: '2026-10-28'
title: 固定5500枚・3 sceneのDINOv3-S384学習と別domain評価
provider: codex
session: court-scene-diversity-l4-20261027-r2
date: '2026-10-27'
status: done
config:
  scenes: 3
  backbone: frozen DINOv3-S
  transformer_dim: 384
  seed: 42
  max_steps: 33160
  train_images: 5500
  synthetic_val_images: 500
  real_val_images: 2211
  batch_size: 8
  longside: 512
  resumed_from_step: null
metrics:
  synthetic/kp_mean_distance_px: 2.644688129425049
  synthetic/kp_median_distance_px: 1.6523398160934448
  synthetic/line_dice: 0.7056074142456055
  synthetic/seg_miou: 0.9663647413253784
  synthetic/semantic_line_miou: 0.5038962960243225
  real/kp_mean_distance_px: 43.695472717285156
  real/kp_median_distance_px: 3.6769473552703857
  real/line_dice: 0.455488383769989
  real/seg_miou: 0.3322363793849945
  real/semantic_line_miou: 0.23183582723140717
artifacts:
  run_dir: knowledge/runs/run-court-scene-diversity-colab-scene-3-s42
  config: knowledge/runs/run-court-scene-diversity-colab-scene-3-s42/resolved-config.yaml
  metrics: knowledge/runs/run-court-scene-diversity-colab-scene-3-s42/metrics.json
  curves: knowledge/runs/run-court-scene-diversity-colab-scene-3-s42/curves.png
  tb_logdir: outputs/court_detection/campaign/scene-diversity-curves-s42/scene-3
parents:
- run-court-scene-diversity-preparation-s42
- run-court-scene-diversity-6000-v1
relations: []
papers: []
tags:
- scene-diversity
- fixed-budget
- colab
---

## 結果と解釈

B00・B01: 各train1833 / val167、B02: train1834 / val166。合計train5500枚、合成val500枚で33160 optimizer更新を完了した。凍結DINOv3-S＋後段幅384、seed42、batch8、L4を全条件で固定した。共通構成・分割・比較の制約は[比較group](000055-group-court-scene-diversity-colab-s42.md)を参照する。

| 最終33160更新の評価 | KP平均 px ↓ | KP中央値 px ↓ | seg mIoU ↑ | line Dice ↑ | semantic-line mIoU ↑ |
|---|---:|---:|---:|---:|---:|
| 合成val 500枚 | 2.644688 | 1.652340 | 0.966365 | 0.705607 | 0.503896 |
| 実画像val 2211枚 | 43.695473 | 3.676947 | 0.332236 | 0.455488 | 0.231836 |

KPのpxは**長辺512設定の前処理後モデル座標**であり、原画像のpixel誤差ではない。実画像2211枚は全条件で同一ID・同一順序とし、pose教師がない実画像をpose誤差0として集計していない。合成valと実画像valを混ぜず最終checkpointから別々に評価した。任意のKP-pose整合性0は無効な診断項目であり完全精度を示さない。

実画像KP平均43.695 pxは1・2条件より低く、実画像seg mIoUも0.3322へ増えた。一方、合成valのKP平均は2.645 pxで1条件より高い。合成valの構成は条件間で異なる。

元のDINOv3-S事前学習重みとseed42から独立に開始した。前条件のcheckpointを初期値に使用していない。

## 保存・追試の根拠

[全評価JSON](../../runs/run-court-scene-diversity-colab-scene-3-s42/evaluation.json)にはcheckpoint SHA、実global_step、解決済み設定、各domainの評価IDと順序・実件数・有限指標を保持した。[結果要約](../../runs/run-court-scene-diversity-colab-scene-3-s42/result.json)、[実行来歴](../../runs/run-court-scene-diversity-colab-scene-3-s42/runtime.json)、[実payload照合](../../runs/run-court-scene-diversity-colab-scene-3-s42/final-checkpoint.json)を保存し、Drive版の入力・学習出力・評価出力をhash照合した。画像とcheckpoint本体はDriveへ保持しGitには複製しない。

[学習曲線](../../runs/run-court-scene-diversity-colab-scene-3-s42/curves.png)と`curves.json`は保存済みTensorBoardの正規履歴を使う。曲線のvalidationは学習時の**合成500＋実画像2211のaggregate**であり、上表のdomain別最終評価とは区別する。全10回の定期validationと学習lossを記録する。

共通の[可搬実行scriptとコマンド](../../runs/run-court-scene-diversity-colab-s42/README.md)を再利用する。元の準備configは書き換えていない。単一seed・固定scene追加順・条件ごとに異なる合成valであり、モデルの自動採用は行わない。
