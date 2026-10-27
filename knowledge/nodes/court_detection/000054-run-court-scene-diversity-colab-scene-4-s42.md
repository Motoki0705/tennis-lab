---
id: run-court-scene-diversity-colab-scene-4-s42
type: run
task: court_detection
sequence: 54
recorded_at: '2026-10-28'
title: 固定5500枚・4 sceneのDINOv3-S384学習と別domain評価
provider: codex
session: court-scene-diversity-l4-20261027-r2
date: '2026-10-27'
status: done
config:
  scenes: 4
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
  synthetic/kp_mean_distance_px: 2.9891910552978516
  synthetic/kp_median_distance_px: 1.8102253675460815
  synthetic/line_dice: 0.6769481301307678
  synthetic/seg_miou: 0.962997317314148
  synthetic/semantic_line_miou: 0.4525994062423706
  real/kp_mean_distance_px: 23.605810165405273
  real/kp_median_distance_px: 3.4234418869018555
  real/line_dice: 0.4775984585285187
  real/seg_miou: 0.525092363357544
  real/semantic_line_miou: 0.24964648485183716
artifacts:
  run_dir: knowledge/runs/run-court-scene-diversity-colab-scene-4-s42
  config: knowledge/runs/run-court-scene-diversity-colab-scene-4-s42/resolved-config.yaml
  metrics: knowledge/runs/run-court-scene-diversity-colab-scene-4-s42/metrics.json
  curves: knowledge/runs/run-court-scene-diversity-colab-scene-4-s42/curves.png
  tb_logdir: outputs/court_detection/campaign/scene-diversity-curves-s42/scene-4
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

B00〜B03: 各train1375 / val125。合計train5500枚、合成val500枚で33160 optimizer更新を完了した。凍結DINOv3-S＋後段幅384、seed42、batch8、L4を全条件で固定した。共通構成・分割・比較の制約は[比較group](000055-group-court-scene-diversity-colab-s42.md)を参照する。

| 最終33160更新の評価 | KP平均 px ↓ | KP中央値 px ↓ | seg mIoU ↑ | line Dice ↑ | semantic-line mIoU ↑ |
|---|---:|---:|---:|---:|---:|
| 合成val 500枚 | 2.989191 | 1.810225 | 0.962997 | 0.676948 | 0.452599 |
| 実画像val 2211枚 | 23.605810 | 3.423442 | 0.525092 | 0.477598 | 0.249646 |

KPのpxは**長辺512設定の前処理後モデル座標**であり、原画像のpixel誤差ではない。実画像2211枚は全条件で同一ID・同一順序とし、pose教師がない実画像をpose誤差0として集計していない。合成valと実画像valを混ぜず最終checkpointから別々に評価した。任意のKP-pose整合性0は無効な診断項目であり完全精度を示さない。

実画像KP平均23.606 px、seg mIoU0.5251で、この単一seedの4条件中、実画像の集計5指標が最も良かった。1条件比のKP平均低下56.2439%は記述的な比較であり、scene数の因果効果や未見撮影への性能保証ではない。

元のDINOv3-S事前学習重みとseed42から独立に開始した。前条件のcheckpointを初期値に使用していない。

## 保存・追試の根拠

[全評価JSON](../../runs/run-court-scene-diversity-colab-scene-4-s42/evaluation.json)にはcheckpoint SHA、実global_step、解決済み設定、各domainの評価IDと順序・実件数・有限指標を保持した。[結果要約](../../runs/run-court-scene-diversity-colab-scene-4-s42/result.json)、[実行来歴](../../runs/run-court-scene-diversity-colab-scene-4-s42/runtime.json)、[実payload照合](../../runs/run-court-scene-diversity-colab-scene-4-s42/final-checkpoint.json)を保存し、Drive版の入力・学習出力・評価出力をhash照合した。画像とcheckpoint本体はDriveへ保持しGitには複製しない。

[学習曲線](../../runs/run-court-scene-diversity-colab-scene-4-s42/curves.png)と`curves.json`は保存済みTensorBoardの正規履歴を使う。曲線のvalidationは学習時の**合成500＋実画像2211のaggregate**であり、上表のdomain別最終評価とは区別する。全10回の定期validationと学習lossを記録する。

共通の[可搬実行scriptとコマンド](../../runs/run-court-scene-diversity-colab-s42/README.md)を再利用する。元の準備configは書き換えていない。単一seed・固定scene追加順・条件ごとに異なる合成valであり、モデルの自動採用は行わない。
