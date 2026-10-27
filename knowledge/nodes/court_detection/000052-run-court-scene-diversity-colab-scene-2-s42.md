---
id: run-court-scene-diversity-colab-scene-2-s42
type: run
task: court_detection
sequence: 52
recorded_at: '2026-10-28'
title: 固定5500枚・2 sceneのDINOv3-S384学習と別domain評価
provider: codex
session: court-scene-diversity-l4-20261026-r1
date: '2026-10-26'
status: done
config:
  scenes: 2
  backbone: frozen DINOv3-S
  transformer_dim: 384
  seed: 42
  max_steps: 33160
  train_images: 5500
  synthetic_val_images: 500
  real_val_images: 2211
  batch_size: 8
  longside: 512
  resumed_from_step: 26528
metrics:
  synthetic/kp_mean_distance_px: 2.2009236812591553
  synthetic/kp_median_distance_px: 1.5342432260513306
  synthetic/line_dice: 0.7093492746353149
  synthetic/seg_miou: 0.9666748642921448
  synthetic/semantic_line_miou: 0.5156534910202026
  real/kp_mean_distance_px: 83.44595336914062
  real/kp_median_distance_px: 67.9181900024414
  real/line_dice: 0.36101362109184265
  real/seg_miou: 0.16365623474121094
  real/semantic_line_miou: 0.19059240818023682
artifacts:
  run_dir: knowledge/runs/run-court-scene-diversity-colab-scene-2-s42
  config: knowledge/runs/run-court-scene-diversity-colab-scene-2-s42/resolved-config.yaml
  metrics: knowledge/runs/run-court-scene-diversity-colab-scene-2-s42/metrics.json
  curves: knowledge/runs/run-court-scene-diversity-colab-scene-2-s42/curves.png
  tb_logdir: outputs/court_detection/campaign/scene-diversity-curves-s42/scene-2-resumed
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

B00・B01: 各train2750 / val250。合計train5500枚、合成val500枚で33160 optimizer更新を完了した。凍結DINOv3-S＋後段幅384、seed42、batch8、L4を全条件で固定した。共通構成・分割・比較の制約は[比較group](000055-group-court-scene-diversity-colab-s42.md)を参照する。

| 最終33160更新の評価 | KP平均 px ↓ | KP中央値 px ↓ | seg mIoU ↑ | line Dice ↑ | semantic-line mIoU ↑ |
|---|---:|---:|---:|---:|---:|
| 合成val 500枚 | 2.200924 | 1.534243 | 0.966675 | 0.709349 | 0.515653 |
| 実画像val 2211枚 | 83.445953 | 67.918190 | 0.163656 | 0.361014 | 0.190592 |

KPのpxは**長辺512設定の前処理後モデル座標**であり、原画像のpixel誤差ではない。実画像2211枚は全条件で同一ID・同一順序とし、pose教師がない実画像をpose誤差0として集計していない。合成valと実画像valを混ぜず最終checkpointから別々に評価した。任意のKP-pose整合性0は無効な診断項目であり完全精度を示さない。

実画像KP平均83.446 pxは1条件の53.949 pxより悪化した。B01追加による単調改善は観測されなかった。scene構成と再開の両方が異なるため、悪化の原因をscene数だけに帰属しない。

scene2のみ、元VMが失われたため保存済み26528更新checkpointから同じconditionを再開した。246個のoptimizer stateのstep、scheduler.last_epoch=26528、復元ログを照合し、最終payloadはversion_1の33160更新だった。許可された設定差は`run.resume`だけである。旧attemptの定期監視では27112更新（破棄下限584）まで観測した。その後に回収した最終TensorBoardは27150更新まであり、そのうち少なくとも622更新は最終モデルへ継承されていない。旧jobは終了code不明のまま保持し、成功に置き換えない。mid-epoch DataLoaderが再開可能でないという警告があり、worker RNGとsampler順の完全な継続は保証できない。VM消失原因は未確定。

## 保存・追試の根拠

[全評価JSON](../../runs/run-court-scene-diversity-colab-scene-2-s42/evaluation.json)にはcheckpoint SHA、実global_step、解決済み設定、各domainの評価IDと順序・実件数・有限指標を保持した。[結果要約](../../runs/run-court-scene-diversity-colab-scene-2-s42/result.json)、[実行来歴](../../runs/run-court-scene-diversity-colab-scene-2-s42/runtime.json)、[実payload照合](../../runs/run-court-scene-diversity-colab-scene-2-s42/final-checkpoint.json)を保存し、Drive版の入力・学習出力・評価出力をhash照合した。画像とcheckpoint本体はDriveへ保持しGitには複製しない。

[学習曲線](../../runs/run-court-scene-diversity-colab-scene-2-s42/curves.png)と`curves.json`は保存済みTensorBoardの正規履歴を使う。曲線のvalidationは学習時の**合成500＋実画像2211のaggregate**であり、上表のdomain別最終評価とは区別する。26528更新に中断線を示し、旧attemptの26528以後の破棄尾部を除外して再開attemptへ接続した。

共通の[可搬実行scriptとコマンド](../../runs/run-court-scene-diversity-colab-s42/README.md)を再利用する。元の準備configは書き換えていない。単一seed・固定scene追加順・条件ごとに異なる合成valであり、モデルの自動採用は行わない。
