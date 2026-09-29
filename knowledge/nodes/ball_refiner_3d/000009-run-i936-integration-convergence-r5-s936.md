---
id: run-i936-integration-convergence-r5-s936
type: run
task: ball_refiner_3d
sequence: 9
recorded_at: '2026-09-30'
title: 固定12ラリーの積分収束を全成分・全frameで監査
provider: codex
status: done
config:
  source_rallies: 12
  source_frames: 4809
  source_components: 64
  seed: 936
  prior_sigmas: 4.0
  initial_cells:
  - 16
  - 24
  - 32
  levels:
  - 7
  - 8
  - 9
  refine_cells:
  - 512
  - 1024
  - 2048
  nll_tolerance_nat: 0.05
  log_evidence_tolerance_nat: 0.05
  mean_tolerance_m: 0.02
  covariance_relative_tolerance: 0.05
metrics:
  frames: 4809
  nonconverged_frames: 4809
  nonconverged_rate: 1.0
  nll_only_nonconverged_frames: 4388
  nonconverged_components: 52587
  nonconverged_component_mass_above_half_frames: 279
  worker_seconds: 7123.238005265826
  failed_frames: 0
artifacts:
  run_dir: knowledge/runs/run-i936-integration-convergence-r5-s936
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/diagnostics/i936-convergence-r5
parents:
- run-i936-synthetic-smoke-r3-s936
relations: []
papers: []
tags: []
issue: 936
date: '2026-09-30'
repro:
  commit: 4d89fe6e8faec34ab6c5fa6932453e7124d18330
  command: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD"
    .venv/bin/python knowledge/runs/run-i936-integration-convergence-r5-s936/audit_smoke.py
    --source /home/kamimura/projects/tennis-lab/data/ball_refiner/synthetic-3d-i936-smoke-r3
    --plan "$PWD/src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml" --output /absolute/new/audit
    --batch 0
---

旧12-rally smokeの4,809frameを、固定済みの2D GMM/camera/priorで全数再積分する監査。
**4,809/4,809frameが処理成功し、厳格な全成分基準では全4,809frameがcapで未収束（100%）**だった。
GPU・再シミュレーション・seed選別は行っていない。
新K=4の較正prefixとは別cohortで、入力は旧K=3の全64成分。実Meiji球の精度評価ではない。

## 判定する量

各frameでHを16/7/512→24/8/1024→32/9/2048（初期格子/levels/refine cells）と増量する。
隣接段階の全componentのlog evidence差≤0.05nat、平均L2差≤0.02m、共分散相対Frobenius差≤0.05を要求する。
共分散の分母は前後のnormの大きい方。float64で重み0にunderflowした成分も判定対象。

混合NLLは最初/直前/現在の全成分平均と各world軸±1周辺標準偏差の点で最大絶対差≤0.05natを要求する。
このprobe集合にはGTを使わない。synthetic GTでのNLLは停止後の診断として別保存する。
全条件が満たされるか3段階capへ到達したら最後の分布を返す。NLLの良い格子の選択はしない。
未収束frame/成分は保持し、flagと実際の差分を保存する。

これは有限格子間の安定性チェックであり、連続積分の誤差上界ではない。
有限box外tail、Laplace誤差、各productをGaussian momentsへ近似する誤差を保証しない。
NLL probeは全空間を網羅せず、低確率領域も含む。strict判定と確率質量で見た影響を混同しない。

## 入力の固定と再現

[source provenance](../../runs/run-i936-integration-convergence-r5-s936/provenance.json)に
入力dataset、plan SHA、数値core全fileとadapterのSHAを保存した。
積分のcoreは実装commit a642c5c7から変更していない。実行期間のpath境界/knowledge追加commitは
このcohortの分布計算を変更していないことを、全core fileとadapterのbytes一致で確認した。
旧ラリーの全NPZを元manifestのSHAと照合してから読む。

再現は`audit_smoke.py`の`--batch 0..4`（既定1,000frame/batch、4 workers）を
新しい同一output directoryへ順番に実行し、`summarize.py`で全12ラリーの全frame identityを照合する。
本実行はprefix検証と合計4 CPU以内で並行できるよう、batch0=1,000frame、続きは500frameへ分割した。
時間は各frameのworker秒とbatch wall秒を区別し、並行batchのwall合計を全体実時間と呼ばない。
TensorBoardは使用しない。

## 全数結果

[summary.json](../../runs/run-i936-integration-convergence-r5-s936/summary.json)を集計の正本とする。
[frame_diagnostics.npz](../../runs/run-i936-integration-convergence-r5-s936/frame_diagnostics.npz)には
全frame identity、全成分の達成差分・flag・重み、各段階の差分を保存した。
原12ラリー・全frameの欠落/重複なし、全NPZ SHA、全64成分を検査した。
[float32 export検査](../../runs/run-i936-integration-convergence-r5-s936/verification.json)でも
全4,809frameのSPD・active cameraの正depth・camera subset質量・flagと差分の一致を確認。
subset質量の最大丸め差は3.04e-7、float32化後のactive depth最小は0.13096mだった。

| Rally | 全frame=未収束frame | NLL probe差 p95 (nat) |
|---|---:|---:|
| train-00000 | 254 | 0.636122 |
| train-00001 | 497 | 0.606919 |
| train-00002 | 512 | 0.738076 |
| train-00003 | 512 | 0.403531 |
| val-00000 | 178 | 0.933607 |
| val-00001 | 512 | 0.677968 |
| val-00002 | 512 | 0.636208 |
| val-00003 | 199 | 7.847845 |
| test-00000 | 343 | 3.028250 |
| test-00001 | 266 | 0.433322 |
| test-00002 | 512 | 0.383006 |
| test-00003 | 512 | 0.622194 |

混合NLL probeだけの未達は4,388/4,809（91.2%）。全成分の未収束は52,587/307,776成分。
未収束成分の確率質量は中央値0.000953、平均0.05981、p95=0.94764で、279frameでは0.5を超えた。
したがって、全frame未収束を「常に微小重みだけの問題」として無視できない。
NLL probeは低確率領域も含み、GT誤差・HDR coverageとは異なる指標である。

| 最後の隣接段階の差分 | p50 | p95 | max | 許容値 |
|---|---:|---:|---:|---:|
| 最大NLL probe差 (nat) | 0.17654 | 0.86566 | 8.70359 | 0.05 |
| 最大log evidence差 (nat) | 0.08360 | 31.24665 | 93.65706 | 0.05 |
| 最大component平均L2差 (m) | 0.55682 | 1.85233 | 4.09093 | 0.02 |
| 最大component共分散相対差 | 0.06918 | 0.87594 | 1.08043 | 0.05 |

指定例`val-00003/frame130`はcap3段階で未収束。24→32の差はNLL probe0.64948nat、
evidence0.26759nat、平均0.50797m、共分散相対0.38016、未収束成分19/64。
synthetic GTの停止後NLLは1.20717nat（密度m⁻³）。GTは停止判定や格子選択に使っていない。

累積worker時間7,123.238秒、batch wallの合計2,646.893秒。
一部batchは合計4 worker以内で並行しており、後者を全実行期間のwall時間とは呼ばない。
CPU/native thread1のみで、空きRAMは観測時23GB以上だった。
通常検証は関連194 tests、境界15 tests、path修正後8 tests、ruff/mypy、設定監査99境界。
生成元4d89fe6eの[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/36631859820)は
6,542 passed / 133 skippedで成功し、knowledge/webui/labelも成功した。

## 採用と次の判断

生成器はこのcap/flag契約を使用する。未収束は例外による処理失敗とは別であり、
flagを保存したことを積分精度の達成やfull生成の承認へ読み替えない。
96ラリーの開発生成は暫定較正とともに起動する。全数率を受けて閾値を緩めたり成分/seedを選別しない。
640ラリーは#935の最終出力を待ち、この未収束の多さと計算費用も踏まえて積分方式/予算を再判断する。
本結果は積分の収束達成ではなく、未達を可視化して保存する契約の検証である。
[較正prefix](000008-run-i936-provisional-degradation-r5-s936.md)との率の差は、
母数・K・gap配置・劣化が異なるため、較正改善の因果効果とは扱わない。
