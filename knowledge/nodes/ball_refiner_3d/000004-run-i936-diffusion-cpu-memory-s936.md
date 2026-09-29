---
id: run-i936-diffusion-cpu-memory-s936
type: run
task: ball_refiner_3d
sequence: 4
recorded_at: '2026-09-29'
title: 絶対x0 flow matchingの解析的入力CPU 100-update診断
issue: 936
provider: codex
date: '2026-09-29'
status: done
config:
  input_kind: analytic_memory_fixture_v1
  seed: 936
  updates: 100
  batch_size: 2
  frames: 128
  learning_rate: 0.0001
  maximum_seconds: 780
  allocator_limit_gib: 4.0
  model:
    width: 128
    layers: 4
    heads: 4
    feedforward_multiplier: 4
    time_frequencies: 8
    dropout: 0.0
  loss:
    x0: 1.0
    reprojection: 0.01
    physics: 0.0001
    event: 0.1
metrics:
  elapsed_seconds: 12.519750058010686
  parameters: 817285
  updates: 100
  checkpoint_bytes: 3291460
  first_update:
    update: 1
    loss: 0.6418809294700623
    gradient_norm: 8.656871795654297
    x0: 0.23158837854862213
    reprojection: 20.95867156982422
    physics: 792.521484375
    event: 1.2145369052886963
  last_update:
    update: 100
    loss: 0.1448688507080078
    gradient_norm: 5.567133903503418
    x0: 0.004739877302199602
    reprojection: 11.34908676147461
    physics: 49.5773811340332
    event: 0.21680372953414917
artifacts:
  run_dir: knowledge/runs/run-i936-diffusion-cpu-memory-s936
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/i936-memory-smoke/cpu-r2-20260929
parents:
- run-i936-synthetic-smoke-v2-s936
relations: []
papers: []
tags: []
repro:
  commit: 08bfc644d8bb96ad44b278394cdd3b676a456d35
  branch: campaign930/i936-2-synthetic-diffusion
  command: bash knowledge/runs/run-i936-diffusion-cpu-memory-s936/repro.sh /absolute/new/output
session: 01a0ed26-b4eb-7c30-baea-c8a3cb41cf9e
---


## 結論

CPUで解析的入力の100 updatesが12.52秒で完了し、全更新のloss/gradientが有限だった。
モデルは817,285 parameters、B=2/T=128/M=64、幅128・4層・4heads、fp32/eager。
保存したdiagnostic checkpointは3,291,460 bytes。
生成器の12-rally失敗とは独立した計算graphの検査であり、
synthetic dataset学習・実データの精度・汎化・較正を示すものではない。

## モデルと目的関数

全3D成分の平均・共分散・camera subsetを非線形MLPで符号化し、
混合重みで集約して時間Transformerに入力する。混合の平均だけには置換しない。
出力は絶対位置x0とhit/bounce logits。court座標の共有正規化を使い、残差出力とGANはない。
直線経路x_t=(1−t)noise+t*x0を使い、Euler法の速度は(predicted_x0−x_t)/(1−t)。
一様tのx0 MSEはvelocity MSE換算で(1−t)²の重みになる。
1-step回帰は同一backboneのstate/time入力を0に固定する。

4損失はx0、全2D成分を使うStudent-t再投影、弱い重力残差、イベントBCE。
重力項はBLCSのdrag/Magnus/windを完全再現せず、イベント±5と差分stencilを除外する
弱いpriorの土台に限定する。これを完全なphysicsモデルとして採用する判断はしていない。

## 入力と検証範囲

`analytic_memory_fixture_v1` はcamera-only JSON、解析的なbounce軌道、
全64成分・相関2D共分散・64frameの分散拡大を持つtensor fixture。
失敗datasetから成功ラリーを抽出して学習していない。val/testの教師は使わない。
checkpointはdiagnostic_only=trueで、本学習初期値として再利用しない。

100-update実行はfrontmatterのcommit（08bfc644）。
[全更新JSONL](../../runs/run-i936-diffusion-cpu-memory-s936/updates.jsonl)を保存したため、
TensorBoard曲線は別途作らない。値は最適化診断で、test精度や採用指標ではない。
入力hash、config、パラメータ数、時刻、checkpoint SHAは
[manifest](../../runs/run-i936-diffusion-cpu-memory-s936/manifest.json)を参照。

後続65c83463でpadding/全不在の2D covarianceを評価対象から外す修正と、
GPUのdriver memory測定とbudget guardを追加した。100-update CPUログを
その後続版の実行ログとは称さない。最終の通常検証は以下。

- model CPU forward/backwardは全4損失と全parameter gradientを検査。
- head固定で絶対x0出力を検査し、入力軌道への残差加算がないことを確認。
- 全成分順序の不変性、64番目の成分の影響、paddingと実gapの区別を検査。
- 複数sampleのempirical covariance、seed再現、同backbone回帰を検査。
- 物理の実秒とimpact mask、外れ値再投影の有限勾配、padding/全不在の無効2D行列を検査。
- [関連34 tests](../../runs/run-i936-diffusion-cpu-memory-s936/final-tests.log)成功（13.66秒）、
  ruff/mypy成功、[設定監査99境界](../../runs/run-i936-diffusion-cpu-memory-s936/scaffold-audit.log)成功。

## 次の一手

許可された1件のGPU memory診断を同じ解析的入力でqueueへ投入し、
allocator peakとdriver total-minus-freeを報告する。GPU結果はこのCPUノードに混ぜない。
本学習・回帰/平滑化対照の品質評価・LOCO・pipeline接続の前には、
生成器の正depth問題と#935由来の劣化較正を解決する必要がある。
