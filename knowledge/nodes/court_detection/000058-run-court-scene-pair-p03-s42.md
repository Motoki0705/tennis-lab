---
id: run-court-scene-pair-p03-s42
type: run
task: court_detection
sequence: 58
recorded_at: '2026-11-10'
title: Court P03（B00+B03）最終33160更新と共通評価
issue:
- 1585
- 1586
provider: codex
status: done
summary_impact: none
config:
  run.seed: 42
  updates: 33160
  batch_size: 8
  training_scene_images:
    B00: 2750
    B03: 2750
  fit_synthetic_val_scene_images:
    B00: 250
    B03: 250
  real_val_images: 2211
  common_synthetic_eval_images: 2439
metrics:
  real/kp_mean_distance_px: 18.578781127929688
  real/kp_median_distance_px: 2.9732167720794678
  real/seg_miou: 0.5608432292938232
  real/line_dice: 0.47918930649757385
  real/semantic_line_miou: 0.2384512573480606
  synthetic_scene_macro/kp_mean_distance_px: 7.208341282150188
  synthetic_scene_macro/kp_median_distance_px: 3.123391807079315
  synthetic_scene_macro/seg_miou: 0.9056867916769852
  synthetic_scene_macro/line_dice: 0.5267444383777248
  synthetic_scene_macro/semantic_line_miou: 0.3314114766917585
repro:
  commit: 17ad301bf8d80a5d094f28bb48f3431c7d9b5d89
  command: bash knowledge/runs/run-court-scene-pair-colab-s42/recovery_bootstrap.sh
  note: 実行runのcaptured patchとVM control helpersを併用。以下はcampaign入口であり単一条件だけを再学習する命令ではない。
artifacts:
  run_dir: knowledge/runs/run-court-scene-pair-p03-s42
  metrics: knowledge/runs/run-court-scene-pair-p03-s42/metrics.json
  provenance: knowledge/runs/run-court-scene-pair-p03-s42/provenance.json
  drive_evaluation: outputs/court_detection/evaluate/scene-pair-5500-s384-colab/P03-s42/evaluation.json
  drive_checkpoint: outputs/court_detection/train/scene-pair-5500-s384-colab/P03-s42/logs/version_1/checkpoints/update-00033160.ckpt
  drive_outputs:
  - outputs/court_detection/train/scene-pair-5500-s384-colab/P03-s42
  - outputs/court_detection/evaluate/scene-pair-5500-s384-colab/P03-s42
  resolved_config: knowledge/runs/run-court-scene-pair-p03-s42/resolved-config.yaml
  tb_logdir: .cache/court-scene-pair-fit-curves-verified/P03
  curves: knowledge/runs/run-court-scene-pair-p03-s42/curves.png
parents:
- run-court-scene-pair-execution-s42
- run-court-scene-pair-manifests-v1
relations:
- to: run-court-scene-diversity-colab-scene-2-s42
  rel: compares
  note: scene-2と同じ学習設定・予算で、manifestと出力先・再開指定だけを変更
papers: []
tags:
- scene-pairs
- seed42
- synthetic-only
- common-evaluation
---

## 観測と条件

P03はB00 + B03を各2,750枚、合成train計5,500枚、fit用合成val計500枚で独立学習した。凍結DINOv3 ViT-S/16、Transformer384・8層・6heads・FFN1024、DPT512、dense head256/depth2、長辺512、BS8、seed42、33,160更新、warmup4,974、lr1e-4→1e-6のcosine、3,316更新ごとのval/saveを固定した。早期終了せず最終checkpointを共通評価した。解決済み設定は[config](../../runs/run-court-scene-pair-p03-s42/resolved-config.yaml)、値と来歴は[metrics](../../runs/run-court-scene-pair-p03-s42/metrics.json)・[provenance](../../runs/run-court-scene-pair-p03-s42/provenance.json)。

KPのpxは長辺512設定の前処理後モデル座標。IoU/Diceは0〜1。実画像valは2,211枚でpose教師なし、共通合成評価は2,439枚（B00=1,056、B01=643、B02=462、B03=278）でpose教師あり。4scene平均は各sceneの集計値の非加重平均であり、2,439枚のpooled値ではない。

実KP平均 18.578781 px、KP中央値 2.973217 px、SEG mIoU 0.560843、LINE Dice 0.479189、semantic-line mIoU 0.238451。共通合成4sceneのKP中央値平均は3.123392 px。単独の結果の順位から採用を決めない。

## 再開・証跡の範囲

保存更新19,896からoptimizer・scheduler・loops・RNG等をfull-state復元して最終33,160へ到達した。初回を含む元restart記録はDriveに保持し、このrunには小さい抽出を保存した。旧VM消失前の未保存末尾の実最終更新はnullのままで、不完全epochのsampler位置・worker RNGを厳密には再現しない。連続実行と同じbatch列・確率軌跡という意味での完全再開ではない。

実行時の原campaignがfull checkpoint payload・設定・入力/pose・評価ID/順序・Drive内容を検証してから完了印を公開した。今回のCPU記録作業はcheckpointを全download/再loadしていない。raw評価JSON・checkpoint・TensorBoard eventはDriveに保持し、Gitには小さい抽出と採用履歴の曲線だけを残す。fit曲線のvalidationは合成500＋実画像2211のaggregateで、最終domain別評価や共通2439評価とは別である。

summary_impact: none。summaryの現在判断は全6ペアを束ねるgroupで更新する。この単一run自体はcheckpointの採用を変えず、groupの根拠を提供する。
