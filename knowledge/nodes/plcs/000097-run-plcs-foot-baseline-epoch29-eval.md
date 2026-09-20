---
task: plcs
sequence: 97
recorded_at: 2026-09-14
date_source: experiment_date
papers: []
id: run-plcs-foot-baseline-epoch29-eval
type: run
title: PLCS axial split既存最良epoch29の同条件評価
provider: codex
date: '2026-09-14'
status: done
config:
  model: plcs_multiview_axial_split (hidden=512, pos/rot 6 layers each)
  loss: position=1, rotation=1, canonical/reprojection=0
  data: single_object_camera_view_v2/test.txt; seed=1234; float32 CUDA
metrics:
  position_error_m: 0.4985384492455994
  position_xy_error_m: 0.48332761394390755
  angular_error_deg: 10.60165023803711
  position_accuracy_0.5m: 0.5851155435871743
  real_root_reprojection_px: 89.83115381572857
artifacts:
  run_dir: knowledge/runs/run-plcs-foot-baseline-epoch29-eval
  predictions: knowledge/runs/run-plcs-foot-baseline-epoch29-eval/baseline_best_test.npz
  output_dir: outputs/plcs/foot_residual/comparison
parents: []
relations:
- to: run-plcs-foot-baseline-epoch19-eval
  rel: compares
tags:
- plcs
- baseline
- evaluation
session: 01a09b38-6993-7e23-9122-9482895fe0b4
repro:
  commit: 43291cdb
  branch: experiments/plcs-foot-residual
  command: .venv/bin/python scripts/plcs_foot_residual/evaluate.py --checkpoint /home/kamimura/projects/tennis-lab/outputs/plcs/plcs_multiview_axial_split/logs/version_0/checkpoints/plcs-epoch=29.ckpt
    --label baseline_best --baseline-config /home/kamimura/projects/tennis-lab/outputs/plcs/plcs_multiview_axial_split/config.yaml
    --dataset data/plcs/single_object_camera_view_v2 --clip data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --output outputs/plcs/foot_residual/comparison
---

## 考察 / Findings

### 要約
指定された既存configの学習成果物から、validation最良のepoch29を強い比較基準として再評価した。合成testの3D誤差0.498538m、実クリップのroot再投影誤差89.831154px。

### アーキテクチャ詳細
既存axial splitの位置直接回帰。比較configとcheckpoint内のmodel/data/loss/training設定の一致は `baseline_config_match.json` に記録。checkpoint SHA256は `370de2f2d920e500cd13964412e4bd861a343efa680f32e709d16b7cd3782d1a`。本ノードは評価runであり、既存モデルの再学習はしていない。

### メトリクスの解釈
998 testシーン・127744フレーム、seed1234、中心128フレーム、reference camera_0、float32。同じGPU上で残差モデルと評価した。実クリップは共通の2D検出・手動courtから本番PLCS predictorを実行。実3D正解はなく、元sceneのモデル推定値はGTとして扱わない。評価のみのため本runの学習曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
直接回帰は疑似位置の極端な誤差に拘束されない。疑似位置誤差2m以上の層では水平誤差0.637782mを達成しており、足元prior導入時にも残すべき比較基準となる。ただし、この評価だけでは耐性の原因を確定できない。

### 既存実験との比較
運用scene生成に使われたepoch19よりも強いcheckpointを主比較に採用。epoch19の保存済みtest評価はbf16だったため厳密なpaired比較は本epoch29と新モデルのfloat32結果に限定する。群全体の比較表は `group-plcs-foot-residual` を参照。

### 次に有効な実験
残差方式の極端なprior誤差層を改善する際は、本直接回帰モデルを比較対象として維持する。
