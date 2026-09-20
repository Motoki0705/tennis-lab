---
task: slcs
sequence: 85
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-input-sensitivity-v1
type: run
title: 'SLCS既存重み: 固定train窓の入力時間対応への感度'
provider: codex
date: '2026-09-19'
status: done
config:
  checkpoint: slcs/real-rgb-pilot-augmented-e60-v2.ckpt; selected epoch55
  data: slcs/real_rgb_pilot_v2; exact archived train batches from gradient probe
  mode: eval
  seed: 42
  precision: cpu float32
  augmentation: false
  training: false
metrics:
  train_windows: 2
  optimizer_steps: 0
  broadcast_ball_reverse_rms_m: 0.012049290432079834
  meiji_ball_reverse_rms_m: 0.008267279819744292
  broadcast_dino_reverse_rms_m: 0.0004643324902272577
  meiji_dino_reverse_rms_m: 0.00018879421530948597
  broadcast_full_teacher_error_m: 5.202080428442751
  meiji_full_teacher_error_m: 6.787822631010033
  prior_eval_max_abs_difference: 0.0
artifacts:
  run_dir: knowledge/runs/run-slcs-input-sensitivity-v1
  diagnostics: knowledge/runs/run-slcs-input-sensitivity-v1/result/results.json
  predictions: knowledge/runs/run-slcs-input-sensitivity-v1/result/meiji.npz
repro:
  commit: 57726f2c
  branch: codex/slcs-input-sensitivity
  command: env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -B knowledge/runs/run-slcs-input-sensitivity-v1/probe.py --output-dir knowledge/runs/run-slcs-input-sensitivity-v1/REPRO_NEW_RUN_ID
  checkpoint_sha256: b925fc2a8cc2eb3dc80aacb7e2e6f8389d40ab48f8afb3c74408b97cbd17d30c
parents:
- run-slcs-ball-gradient-probe-v1
relations: []
tags:
- slcs
- real-rgb
- cpu-diagnosis
- input-sensitivity
---

## 考察 / Findings

### 要約
固定train窓2つで、ボール観測の時間反転に対する3D出力差はRMS 8–12mm、DINO時刻対応の反転では0.2–0.5mmだった。既存重みのこの局所介入への感度は小さい。分布外介入であり、入力を完全に無視していること、頑健性、教師精度や汎化品質を証明しない。

### アーキテクチャ詳細
勾配診断で保存したbroadcast_shanghai/clip_004 cam0 start0とvideo_000/clip_000 cam0 start0の120frame入力を読み、全tensor hashを元記録に照合した。入力DINOを再計算せず、checkpoint・設定・保存入力ファイルは独立2実装のSHA-256で検査した。元データ由来のclip/source frame情報は結果JSONに引き継ぎ、現時点の生動画やdatasetを再読込・再hashしたという主張はしない。

validation選定済みepoch55のSHAを固定し、既存adapter経由のCPU FP32・eval・seed42でfull、ball_reverse、dino_reverseを各1回実行。ball_reverseでは非paddingフレームのUVと入力confidence/validityであるball_visを対として反転する。入力の欠測位置も一緒に移動するので、純粋なUVだけの効果とは分離できない。dino_reverseでは非padding sampleのtoken配列だけを反転し、絶対時刻dino_frame_idxとpadding maskを固定する。いずれも教師、教師mask/weight、通常frame_idxとtimestamp、player/court入力は変更しない。

### メトリクスの解釈
差は全120実フレームに対し、正本COURT_COORD_SCALE_XYZで一度だけmへ変換したXYZ差のEuclidean normから計算する。RMSはsqrt(mean(dx²+dy²+dz²))。教師誤差はproductionのtarget_ball_valid & ~padding_mask上の非重み付き平均距離であり、confidence-weighted学習lossではない。有効数はbroadcast96、Meiji118。軸別RMS・平均・最大、全出力・教師・mask・入力mappingは成果物に保存した。

| domain / 介入 | 出力差RMS (m) | 平均 (m) | 最大 (m) | 教師誤差 (m) |
|---|---:|---:|---:|---:|
| broadcast / full | 0 | 0 | 0 | 5.202080 |
| broadcast / ball_reverse | 0.012049 | 0.009276 | 0.021483 | 5.200468 |
| broadcast / dino_reverse | 0.000464 | 0.000421 | 0.000768 | 5.201944 |
| Meiji / full | 0 | 0 | 0 | 6.787823 |
| Meiji / ball_reverse | 0.008267 | 0.006618 | 0.019336 | 6.790601 |
| Meiji / dino_reverse | 0.000189 | 0.000179 | 0.000313 | 6.787897 |

追加学習もTensorBoard曲線もない。教師は旧pilot版で、新Meiji教師の品質評価ではない。

### アーキテクチャ⇄メトリクスの因果考察
入力時間対応を変えてもメートル規模の教師誤差はほぼ変化しない。時間対応への弱い依存はほぼ定数出力という既存観測と整合する。ただしDINO反転はsample集合・各画像の空間情報を維持しており、静的scene/context依存まで検査しない。ボール反転も値集合を保存する。入力無視、学習原因、lossの因果を断定しない。2窓・1seed・eval modeの結果であり、本学習のdropoutやBF16を代表しない。

### 既存実験との比較
親の勾配診断eval出力と今回fullは両domainで全要素完全一致し、モデルstateも不変だった。親の勾配計算を複製せず、入力を固定する保存済みbatchと既存model_ioを再利用した。test/val推論・test選定・学習queue投入はない。反転mapping、教師とpadding不変、逆操作、m単位と教師maskを独立した小さな決定的テスト3件で確認した。

### 次に有効な実験
Meiji全体教師/QCの完了後、既定のball smooth重みだけを0にする60epoch比較を優先する。本診断はその追加学習ではない。時間依存を改善できたかを見る際は教師誤差・予測分散とともに同じ固定窓への入力感度も比較するとよい。静的RGB依存は別のno_rgb等の介入が必要であり、この時間反転から代替推定しない。
