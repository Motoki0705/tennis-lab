---
id: run-court-b01-canonical-sfm
type: run
title: B01：手動確認済み配置からSfM制約付きCourt v3を正式生成
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: done
config:
  scene: B01
  sfm_boundary_expansion_percent: 5.0
metrics:
  courts: 3
  accepted_frames: 2052
  rejected_frames: 172
  minimum_expanded_hull_clearance_m: 0.9165731152093644
repro:
  commit: c6bc1c097a8ae421be1b6133a073ae93ae451d58
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. .venv/bin/python outputs/court-b00-b03-generation/run_requested.py
    outputs/court-b00-b03-generation/B01-requested.yaml
artifacts:
  run_dir: knowledge/runs/run-court-b01-canonical-sfm
  output_dir: /home/kamimura/projects/tennis-lab-worktrees/court-sfm-bounded/outputs/court-b00-b03-generation
  log: knowledge/runs/run-court-b01-canonical-sfm/run.log
parents:
- run-b01-court-expansion-5
relations: []
tags:
- court
- synthetic-data
- canonical-generation
---

## 考察 / Findings

### 要約
最新mainの手動alignment対応を取り込み、通常ScenePipelineRunnerでcourt_datasetとreportを生成した。手動の3面配置を維持し、2,224候補から2,052枚を採用した。

### アーキテクチャ詳細
円・楕円・長方形・スーパー楕円、採用コート中心の平均をcomplex中心に使用。SfM水平凸包を頂点平均から5%拡張、0.5m余白、半径倍率0.65〜0.95、注視点半径1mのjitter。通常のmanual validatorで再構成との結合と測定値を検証し、実験用のaccepted-status置換は使用していない。

### メトリクスの解釈
正式dataset/reportのstageはcompleted。全公開サンプルをHEADERS_ONLYモードで再検証し、全採用カメラが拡張境界内、最小余白0.9166m。代表画像ではネット・遠景のぼけが残る。下流精度は未評価。

### アーキテクチャ⇄メトリクスの因果考察
mainの手動alignment機能により、元の配置と測定根拠をそのまま正式パイプラインへ接続できた。配置を自動最適化し直す必要はなかった。

### 既存実験との比較
前回はgeometry adapterによるpublic renderer検証のみ。今回はcanonical dataset、labels、diagnostics、reportの公開と正式reader検証まで完了した。

### 次に有効な実験
B00/B02/B03の現行alignment再生成と同条件のCourt生成。

実行用一時wrapperの最後の結果表示で不存在のMutableRunManifest.to_dictを呼び、queue jobはexit1になった。これはdataset/reportの正常公開後のログ表示エラーであり、公開stage両方completedと独立validate_court_dataset/validate_alignment_outputsで成功を確認した。wrapperは後続run向けに修正した。保存済みrun.json/logのexit1を上書きしていない。
