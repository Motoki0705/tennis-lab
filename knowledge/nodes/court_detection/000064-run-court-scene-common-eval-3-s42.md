---
id: run-court-scene-common-eval-3-s42
type: run
task: court_detection
sequence: 64
recorded_at: '2026-11-12'
title: Court既存scene-3の4シーン共通再評価
issue:
- 1585
- 1586
provider: claude
status: done
config:
  run.seed: 42
  updates: 33160
  batch_size: 8
  training_scene_images:
    B00: 1833
    B01: 1833
    B02: 1834
  fit_synthetic_val_scene_images:
    B00: 167
    B01: 167
    B02: 166
  real_val_images: 2211
  common_synthetic_eval_images: 2439
metrics:
  real/kp_mean_distance_px: 43.695472717285156
  real/kp_median_distance_px: 3.6769473552703857
  real/seg_miou: 0.3322363793849945
  real/line_dice: 0.455488383769989
  real/semantic_line_miou: 0.23183582723140717
  synthetic_scene_macro/kp_mean_distance_px: 5.003663260727476
  synthetic_scene_macro/kp_median_distance_px: 2.949115216732025
  synthetic_scene_macro/seg_miou: 0.9317676335586209
  synthetic_scene_macro/line_dice: 0.6099038820050655
  synthetic_scene_macro/semantic_line_miou: 0.3971447249160151
artifacts:
  run_dir: knowledge/runs/run-court-scene-common-eval-3-s42
  metrics: knowledge/runs/run-court-scene-common-eval-3-s42/metrics.json
  provenance: knowledge/runs/run-court-scene-common-eval-3-s42/provenance.json
  drive_evaluation: outputs/court_detection/evaluate/scene-pair-5500-s384-colab/scene-3-s42/evaluation.json
  drive_checkpoint: outputs/court_detection/train/scene-diversity-6000-s384-colab/scene-3-s42/logs/version_0/checkpoints/update-00033160.ckpt
  drive_outputs:
  - outputs/court_detection/evaluate/scene-pair-5500-s384-colab/scene-3-s42
parents:
- run-court-scene-diversity-colab-scene-3-s42
- run-court-scene-pair-execution-s42
relations:
- to: run-court-scene-diversity-colab-scene-3-s42
  rel: evaluates
  note: 既存最終checkpointの同一実画像valと共通合成valによる再評価
papers: []
tags:
- scene-pairs
- seed42
- synthetic-only
- common-evaluation
summary_impact: none
repro:
  commit: 17ad301bf8d80a5d094f28bb48f3431c7d9b5d89
  command: .venv/bin/python -m src.tasks.court_detection.scripts.evaluate_ablation
    --checkpoint /content/tennis-lab/outputs/court_detection/train/scene-diversity-6000-s384-colab/scene-3-s42/logs/version_0/checkpoints/update-00033160.ckpt
    --expected-step 33160 --split val --series scene-pairs --synthetic-eval-manifest
    /content/tennis-lab/data/court_scene_diversity_6000_v1/eval-4scene.val.subset.json
    --output /content/tennis-lab/outputs/court_detection/evaluate/scene-pair-5500-s384-colab/scene-3-s42
    --data-root /content/tennis-lab/data --checkpoint-root /content/tennis-lab/ckpt
    --artifact-root /content/tennis-lab/outputs --external-asset-root /content/tennis-lab/third_party
---

## 既存checkpointの共通評価

既存scene-3の最終33,160更新checkpointを再利用した。学習・元のnode・元の500枚合成val評価は変更していない。新しい評価では4scene共通2,439枚のmanifestを明示し、実画像valは元と同じ2,211 IDと順序を使った。

KPのpxは長辺512設定の前処理後モデル座標。IoU/Diceは0〜1。実画像valは2,211枚でpose教師なし、共通合成評価は2,439枚（B00=1,056、B01=643、B02=462、B03=278）でpose教師あり。4scene平均は各sceneの集計値の非加重平均であり、2,439枚のpooled値ではない。

実画像5指標とsample ID順序は旧bundleのevaluation.jsonと全て完全一致し、許容誤差による一致ではない。来歴は[provenance](../../runs/run-court-scene-common-eval-3-s42/provenance.json)、小さい指標抽出は[metrics](../../runs/run-court-scene-common-eval-3-s42/metrics.json)。実KP中央値3.676947 px、共通合成4sceneのKP中央値平均2.949115 px。

旧合成val500枚は各条件の学習scene構成に依存するため、今回の共通評価値と同じ母集団として扱わない。旧fit曲線は旧training nodeに保持し、再評価nodeには複製しない。再評価のみなので新しいfit曲線はない。scene-2は旧学習で26,528更新から再開しており、そのsampler/RNG制限は旧nodeの記録を引き継ぐ。

summary_impact: none。全9条件を同じ合成評価集合で比較する根拠を追加するもので、既存学習の結果・採用判断を単独で変更しない。現在判断の更新はcomparison groupに集約する。
