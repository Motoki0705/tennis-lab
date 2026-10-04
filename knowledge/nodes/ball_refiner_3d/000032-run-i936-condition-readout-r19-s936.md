---
id: run-i936-condition-readout-r19-s936
type: run
task: ball_refiner_3d
sequence: 32
recorded_at: '2026-10-01'
date: '2026-10-01'
title: 現(c)両encoderの平均read-outは5.53/3.64mm：位置はpool tokenに残る
issue: 936
provider: codex
status: done
config:
  checkpoint_updates: 20000
  fit_train_rallies: 512
  fit_train_frames: 218320
  validation_rallies: 16
  validation_frames: 6383
  components: 125
  target: full input mixture mean; no GT
  solver: gelsd float64 SVD, intercept, rcond=1e-12, no ridge
  rule_val_rmse_m_max: 0.1
metrics:
  flow:
    val:
      frames: 6383
      rmse_m: 0.005530089954245954
      p50_m: 0.000790360153969957
      p95_m: 0.008015746904189266
      maximum_m: 0.247883828833169
    rank: 129
    passed: true
    resources:
      elapsed_seconds: 108.93086565297563
      peak_process_rss_bytes: 1612603392
      minimum_available_host_bytes: 20413915136
      gpu_jobs: 0
  regression:
    val:
      frames: 6383
      rmse_m: 0.0036431996759799406
      p50_m: 0.0004905344255297651
      p95_m: 0.005971631173069098
      maximum_m: 0.14021986964373515
    rank: 129
    passed: true
    resources:
      elapsed_seconds: 81.66345340607222
      peak_process_rss_bytes: 1628622848
      minimum_available_host_bytes: 23175061504
      gpu_jobs: 0
repro:
  commit: 1a506efbefd15ef329b399d5d834828dedb412e9
  branch: campaign930/i936-2-synthetic-diffusion
  command: plan.json commands.flow and commands.regression, each executed once in
    the recorded cwd
artifacts:
  run_dir: knowledge/runs/run-i936-condition-readout-r19-s936
  predictions: knowledge/runs/run-i936-condition-readout-r19-s936
  log: knowledge/runs/run-i936-condition-readout-r19-s936/audit.log
parents:
- run-i936-offset-source-cpu-r18-s936
- run-i936-combined512-physics10-r16-s936
relations:
- to: run-i936-condition-readout-r11-s936
  rel: compares
papers: []
tags:
- cpu
- frozen-encoder
- readout
- negative-bottleneck-evidence
---

[事前登録](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5922845132)とrun19 directiveに従い、
現(c)20kの両armを凍結して各1回のCPU read-outを実施した。
**flow 0.005530090m、regression 0.003643200mで両方ともval RMSE≤0.10mを達成した。**
この値は入力混合平均の復元誤差であり、合成GTへの3D推定精度ではない。

## 方法と固定条件

全512train・218,320frameを一度ずつfitに使い、同16val・6,383frameを固定評価した。
教師は保存された125成分の重み×平均の総和のみ。float64で計算し、成分削除・再正規化・GT教師なし。
実モデルのpool直後128次元tokenに切片を足し、float64 SVD最小二乗（gelsd、rcond=1e-12、ridgeなし）
で387係数をfitした。encoder/state/time/Transformer/headは更新しない。
未使用48valと64testの配列を開かず、seed/checkpoint/valによる係数や閾値の選択なし。
rank不足の別solver・ridgeへのfallbackなし。H・現在のanchored bank・#959 control・(c)は維持。

[実行前plan](../../runs/run-i936-condition-readout-r19-s936/plan.json)が条件・12source/checkpoint hash・
正確なコマンドの正本。現checkpoint SHAはflow c1f61ba8… / regression cfdc70fd…。
旧bank/64trainのnode000017を現encoderの証拠として代用せず、新たに測定した。

## 全表と線形layer

[train/val×全体/可視camera 0/1/2/3の全20行](../../runs/run-i936-condition-readout-r19-s936/tables.md)に
件数、RMSE、p50、p95、最大誤差、事前規則の判定を保存した。
[flow layer](../../runs/run-i936-condition-readout-r19-s936/flow/head.npz)と
[regression layer](../../runs/run-i936-condition-readout-r19-s936/regression/head.npz)はfloat64の全係数・
全129特異値・rankを含む。特異値のJSON表現も[audit.json](../../runs/run-i936-condition-readout-r19-s936/audit.json)に保存。

| arm | val RMSE m | p50 m | p95 m | maximum m | SVD rank | singular max / min | condition number | 規則 |
|---|---:|---:|---:|---:|---|---|---:|---|
| flow | 0.005530090 | 0.000790360 | 0.008015747 | 0.247883829 | 129/129 | 791.585208 / 0.005673315 | 139527.8 | 合格 |
| regression | 0.003643200 | 0.000490534 | 0.005971631 | 0.140219870 | 129/129 | 664.508652 / 0.003048150 | 218003.9 | 合格 |

train RMSEは3.09/2.30mm、3camera可視valは1.58/0.94mm。
1/2camera層の復元誤差は高く、最大誤差が0.10mを超えるframeもある。
規則は全valのRMSEであり、「全frame≤0.10m」や全混合分布の完全保存は主張しない。
正規化・raw特徴からの復元誤差は各manifestに残した。

## 観測と解釈

poolされた現encoder tokenから入力平均をmm単位で復元できる。
したがって、run18で見た0.475m級のwell-observed出力誤差を
「平均位置がencoderで大きく失われたため」と説明する仮説は支持されない。
ただし共分散・全分布情報の保持、Transformer以降での利用可能性、学習最適化は証明していない。
train最小特異値は小さく条件数も大きいので、未知分布への安定性保証にもならない。
SVD layerを推論へ挿入したり、混合平均への残差出力へ切り替えたりしない。
(c)の性能・正式15軸パレート判定は変わらず、repro3不採用も維持する。

次は**pooled tokenを絶対位置headの入力へ直接連結する経路**を単一要因候補とする。
位置が残る局所tokenを時間Transformer/最終LayerNormだけを経由させる現在の経路が、
観測忠実度の学習を妨げているかを調べる。tokenの直接利用でjitterが増える反証も評価する。
これはTransformerとLayerNorm個別の原因を証明する実験ではない。
設定・初期値の公平性・資源・判定規則を別の実行前planで固定できた場合のみ1件をqueueへ入れる。

## 検証・資源・限界

通常検証は10tests（9.32秒、-n4）・ruff・mypy4files・pre-commit成功。
両arm×全val/subset、除外valへのhash/read禁止、全成分平均のfit教師、checkpoint update不一致、
rank不足の明示SVD解、p50を検査した。
[audit.py](../../runs/run-i936-condition-readout-r19-s936/audit.py)はencoder再実行・refitをせず、
528保存NPZの全成分混合平均とmaskをNumPyで再計算し全教師値を照合した。
全20行の50分位点/RMSE等、layerのrank・129特異値・artifact hash、前後12source hash、
全528rally hash、元train/val IDが一致した。GT位置配列はこの監査では読まない。

flow 108.931秒/RSS1.613GB/最低空き20.414GB、
regression 81.663秒/RSS1.629GB/最低空き23.175GB、CPU各1process/native1、GPU0件。
両arm成果物は合計21,140,247 bytes、100MB上限内。新datasetなし。
学習curve/TensorBoardは凍結診断のため該当しない。test/追加val・実Meiji・pipeline未実施。
現在のbankが由来するJPEGと2026-10-01決定のmp4本番経路の差は未解消である。
