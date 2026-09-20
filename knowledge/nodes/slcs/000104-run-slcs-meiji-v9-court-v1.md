---
task: slcs
sequence: 104
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-v9-court-v1
type: run
title: 2pass Courtを全56clipへ適用し比較実験との完全一致を確認
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-19'
status: done
config:
  stage: court
  dataset: slcs/meiji_rgb_v9
  observation_directory: tennis_scene/precompute/meiji_dino_vitpose/s42-005
  crop_refinement_padding_px: 20.0
  checkpoint_sha256: b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383
  fit_thresholds_unchanged: true
metrics:
  generated_clips: 56
  generation_failures: 0
  calibration_clips: 3
  exact_comparisons: 150
  stable_comparison_inputs: 110
repro:
  commit: 4af1c8d1d0e5ee3d4d02d2166cddff7a1cd78334
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m src.tennis_scene.scripts.build_slcs_dataset stage=court paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-court-v1
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v9/s42-001
  comparison: knowledge/runs/run-slcs-meiji-v9-court-v1/comparison.json
  initial_comparison_failure: knowledge/runs/run-slcs-meiji-v9-court-v1/failedcomparison.json
parents:
- run-slcs-meiji-court-crop-comparison-v1
relations: []
tags:
- slcs
- meiji
- court
- calibration
- reproducibility
---

## 考察 / Findings

### 要約
同じCourt重み・fit閾値で初期推定のCourt範囲を含む2pass cropを通常生成へ導入し、Meiji対象56clipのCourtとoutsource ballの取り込みが失敗0で完了した。校正3clipのproductionと先行probe A/C、cam2旧結果の150比較が完全一致した。これはCourt観測の採用根拠であり、全体3D教師とSLCSの完成を意味しない。

### アーキテクチャ詳細
初回ball cropでHを推定し、Hが投影するCourt14点と初期ROIのunionへ20pxを加え、同じ9frameを再推論する。cam0/1は2pass、full画像のcam2は1pass。raw点・score・frame index・ROI・Hを両passで保存し、失敗時のfallbackは無い。3校正clipのHを同一収録・同一camera layoutへ配布する。校正や人物選択が異なる過去の教師はコピーしない。

### メトリクスの解釈
queue stage=courtは56clipで終了code0、failuresは空。CPU比較は3校正clipの150項目でdtype/shape/value完全一致、110入力のpre/post dual SHAも一致した。cam0/1はprobe A/Cのraw/score/indices/H/ROIと公開samples/Hを照合。cam2は旧raw/H/keypoints/diagnosticsと照合した。cam2 rawには先行probeの保存SHAが無いため、同じ旧Court由来rootと今回の前後SHAによる確認であり、過去probeによるpinとは区別する。学習曲線は無い。

### アーキテクチャ⇄メトリクスの因果考察
同一入力・重み・設定で通常生成が比較実験と完全一致したため、先行probeで観測した局所白線ずれの改善を同じ校正clipで再現できる。全画面や絶対3D正解の精度を示すものではなく、教師の三角測量・再投影・速度・coverage確認は後続で行う。

### 既存実験との比較
親runのC候補を実装し、初回A・最終Cの一致を確認した。初回CPU比較は旧raw samplesをs42-004直下に置くと仮定したdriverのパス誤りで停止した。旧rawの実在root s42-001を必須引数で明示し、probe記録と旧Court NPZ/JSONのSHAを照合して修正した。元scriptとfailedcomparison.jsonを保持した。この失敗は新たなhash内容不一致ではなく、GPU生成の再実行もしていない。

### 次に有効な実験
新Hで人物選択を再計算し、全入力配列が変わらないcameraだけ観測を再利用する。RGB特徴は入力media/manifest/spec/producerが同一のcacheだけ再利用する。その後Meiji全体の3D教師を生成し、固定閾値のstrict品質レポートと画像確認を行う。
