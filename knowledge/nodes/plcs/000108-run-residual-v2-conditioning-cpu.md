---
id: run-residual-v2-conditioning-cpu
type: run
task: plcs
sequence: 108
recorded_at: '2026-09-21'
title: 残差入力のBF16埋め込み感度をCPU診断
provider: codex
status: done
config:
  device: cpu
  samples_per_task: 8
  indices:
  - 1
  - 3
  - 4
  - 12
  - 16
  - 22
  - 25
  - 48
  dataset: full v2 validation, seed42, 3 views
  transform: asinh(residual_uv / 0.01)
  parameter_updates: 0
  checkpoint: smoke epoch0; see script
metrics:
  plcs:
    raw_lost_fraction: 0.464
    asinh_lost_fraction: 0.0217
    raw_cosine_sample_median_mean: 0.449
    asinh_cosine_sample_median_mean: 0.9985
  blcs:
    raw_lost_fraction: 0.57
    asinh_lost_fraction: 0.0644
    raw_cosine_sample_median_mean: 0.269
    asinh_cosine_sample_median_mean: 0.9909
  input_bf16_underflow_fraction: 0.0
artifacts:
  run_dir: knowledge/runs/run-residual-v2-conditioning-cpu
parents:
- run-plcs-residual-v2-gpu-smoke-r2
- run-blcs-residual-v2-gpu-smoke-r2
relations: []
papers: []
tags:
- triangulation-residual-v2
- cpu-diagnostic
- input-conditioning
date: '2026-09-21'
session: 01a0bf91-5494-74c0-8793-3424866618ed
repro:
  commit: 4671f9bad0873a23054a3311009b9ecf4a6bbc88
  command: PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    .venv/bin/python knowledge/runs/run-residual-v2-conditioning-cpu/conditioning_probe.py
---

## 考察 / Findings

### 要約
再投影差分は通常特徴より小さく、MLP埋め込み内でその影響がBF16の丸めに埋もれる場合が多い。固定asinh変換はFP32との差分方向をよく保つ。学習停滞の原因を確定した診断ではなく、入力conditioningを独立に比較する根拠とする。

### 条件と指標
完了済smoke checkpointのmodel.embedだけをCPUで計算し、差分blockあり/zeroの出力差を比較した。各taskでnormal calibration 2、normal observation 2、hard combined 4の8例をcorruption family/severityだけで選び、GT値は選択・正規化に使っていない。reported metricsはscoutの測定値を丸めて保存したもの。lost fractionはFP32の埋め込み差が非zeroの要素中、BF16の差がexact zeroだった率。各sampleで計算後8例を単純平均する。cosineも各sample内中央値の8例平均である。再現scriptは返却された計測式を保存した。

入力residualそのもののBF16 underflowは0%。O(1)のcamera/maskを含むlinear出力上の小差の丸めが問題候補である。rawのFP32差/full embedノルムはPLCS約0.211%、BLCS約0.154%、asinh後は約14.6%、7.92%だった。

### 判断と限界
次はfeatures.residual_encoding/asinhとscale0.01をconfig・checkpoint契約へ明示し、同じデータ・モデル・目的関数でrawと比較する。既存checkpointの入力意味は書き換えない。CPU BF16はGPU kernelと完全に同一ではなく、full Transformer・head・loss gradientやGT補正の予測可能性は未検証。scale変更はカメラ校正に欠ける情報を生成しない。optimizerを実行しないため学習曲線は存在しない。
