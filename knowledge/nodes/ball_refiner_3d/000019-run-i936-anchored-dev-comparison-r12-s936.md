---
id: run-i936-anchored-dev-comparison-r12-s936
type: run
task: ball_refiner_3d
sequence: 19
recorded_at: '2026-09-30'
title: Anchored bankの96ラリー監査と同一軌道の3D条件比較
provider: codex
status: done
issue: 936
date: '2026-09-30'
config:
  seed: 936
  method: fixed_hybrid
  components: 125
  counts:
    train: 64
    val: 16
    test: 16
  quality_splits:
  - train
  - val
  hdr_threshold_samples: 512
  hdr_independent_volume_samples: 512
  bank_sha256: 0697fe921daf79c7960d616ed0437efecd3b7195f3f8a7dd6188048dd8ecc858
  report_sha256: 2c9cd7ba9a1d63addeaecb808441fbae0447ad21e4a3a279ed1694fcbf67ae01
metrics:
  generated_rallies: 96
  generated_frames: 40774
  generation_failures: 0
  generation_seconds: 4322.4420120749855
  dataset_bytes: 213418311
  control_nll_nat: 3.6776749997561584
  anchored_nll_nat: -2.979325308256937
  control_hdr_coverage:
  - 0.42725858069820016
  - 0.7890425438210165
  - 0.850436008103585
  anchored_hdr_coverage:
  - 0.6768548694911771
  - 0.8953286943245544
  - 0.9242197363398807
  control_hdr_volume_mean_m3:
  - 19.168230896397056
  - 118.8454383773954
  - 192.99035231867592
  anchored_hdr_volume_mean_m3:
  - 3.260055637410204
  - 20.96394497767134
  - 33.62856239019538
  control_mean_rmse_m: 7.107854752954343
  anchored_mean_rmse_m: 4.321989447354296
  paired_quality_rallies: 80
  test_arrays_read: 0
artifacts:
  run_dir: knowledge/runs/run-i936-anchored-dev-comparison-r12-s936
  output_dir: /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/outputs/c936-r12-anchored
repro:
  commit: 66d11166
  branch: campaign930/i936-2-synthetic-diffusion
  command: timeout --signal=TERM --kill-after=15s 1140s env CUDA_VISIBLE_DEVICES=''
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/.venv/bin/python
    -u -m src.tasks.ball_refiner.scripts.audit_conditions_3d --dataset /home/kamimura/projects/tennis-lab/data/ball_refiner/i936-dev-h-anchored-r11-s936
    --output /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/outputs/c936-r12-anchored
    --samples 512
parents:
- run-i936-h-dev-r9-s936
- run-i936-context-t128-r12-s936
relations: []
papers: []
tags: []
---

run11で開始したanchored bankによる96ラリー生成を回収し、**96/96成功、失敗0**を確認した。
旧#959 bankのdev原本を保持し、全80train+valのpaired比較を完了した。GT NLLとHDR90/95被覆・体積は改善したが、HDR50の過大被覆と1camera層の大誤差が残る。
[事前の母数・MC予算](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5909800124)、
[生成回収コメント](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5910035752)を参照。

## 生成と独立した保存監査

[最終生成manifest](../../runs/run-i936-anchored-dev-comparison-r12-s936/generation-manifest.json)のSHAは
`ba1759196f3472939c529dc0d7cd5a866cefca064b0fc693ec037a3f239c7366`。
[起動記録](../../runs/run-i936-anchored-dev-comparison-r12-s936/c936-r11-anchored-generation-launch.json)に
run11のsource commit `2ff58ce7`、PID2360417、正確なコマンドを保持した。生成中にsource36入力/config/bankを変更していない。
全96件のNPZ SHA/bytes、JSONとmanifest、rally seed、camera source、展開plan、bank/reportを
[監査台帳](../../runs/run-i936-anchored-dev-comparison-r12-s936/anchored-generation-audit.json)で照合した。
全96件のphysics/proposal seeds/events/gap/split/geometryは旧controlと一致する。
さらに全80train+valのnative/再標本化軌道、timestamp、全camera行列、event/free-flight/gap/out-of-frame maskの内容hashが厳密一致した。plan差はbank/report/path/hash/statusだけ。

64train/16val/16test、27,676/6,383/6,715frame、全5,096,750成分。240Hzから60000/1001Hz、Hと全K4/125成分を維持。
226物理提案→96採用は旧controlと同じで、数値/solver失敗から別seedを選び直していない。
**testの16件はmetadata/file hashだけで、今回NPZ配列を開かず品質も採点していない。**
これは「同じ96件で比較」と「test禁止」を[明示的に整合](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5909494002)した範囲である。
生成器自身は全96件の保存直前に有限性/SPDを確認しているが、今回testを独立reader検証したとは言わない。

生成4,322.442秒（72.041分）、sum rally elapsed16,980.688秒、最大単一worker RSS882,940KiB。
総出力213,418,311 bytes（NPZ211,371,020）。CPU user+system時間は未計測。
旧controlの52分との差には共有負荷も含むため、bankだけの計算費用差とは解釈しない。

## 固定した比較の意味

品質の母数は全80train+val / 34,059frame、baselineは同じ16val / 6,383frame。
NLLはGTの全3D混合密度（nat、m⁻³）。HDR50/90/95は全混合の密度閾値から求める。
各frameの閾値512標本と、独立した体積512標本、seedは`SeedSequence([rally_seed, frame, 93612])`で両bank共通。
体積はR³上の`E_q[1(log q(X)>=threshold)/q(X)]`、楕円体近似や成分体積の単純和ではない。
MC標準誤差はframeごとの推定閾値に条件付きで、閾値推定の不確実性を含まない。
frame相関やrally母集団を無視した有意性は主張しない。0重み成分も配列から除外しない。

可視camera数はocclusion/out_of_frameのどちらでもない台数で、amodal presenceとは区別する。
全体はframe加重で、ラリー平均の平均ではない。trainとvalを分けた全表もmanifestへ保存する。
混合平均とRTSは既定定数・入力由来イベント・全frameを用い、GTイベントで調整しない。
旧controlの全baseline指標/RTS診断はrun10保存値と厳密一致した。

新bankは[#935 handoff](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5908787736)の全14,652行/K4を保持した
[受領snapshot](../../runs/run-i936-anchored-dev-comparison-r12-s936/calibration/calibration.json)。
6clip fit frameに配布倍率1.8125を適用したempirical材料で、独立/OOF性能ではない。
旧bankとの差にはhead/平均/重み/presence/共分散を含み、共分散倍率だけの因果比較ではない。
camera間独立bootstrap、16frame block内だけの相関、32/64gapへの外挿、負例presence、境界clipの限界を引き継ぐ。

## 旧bank → 新bankのpaired結果

[全表（train・val・合算、HDR体積中央値/MC SE、baseline層別）](../../runs/run-i936-anchored-dev-comparison-r12-s936/comparison.md)、
[比較JSON](../../runs/run-i936-anchored-dev-comparison-r12-s936/comparison.json)と全frameのNPZを保存した。

| 可視camera | frames | GT NLL nat（旧→新） | HDR50/90/95 %（旧） | HDR50/90/95 %（新） | 混合平均RMSE m（旧→新） |
|---|---:|---:|---:|---:|---:|
| all | 34059 | 3.678 → -2.979 | 42.73 / 78.90 / 85.04 | 67.69 / 89.53 / 92.42 | 7.108 → 4.322 |
| 0 | 2406 | 5.591 → 3.166 | 36.49 / 76.56 / 84.75 | 58.23 / 86.62 / 89.94 | 5.670 → 3.718 |
| 1 | 1213 | 9.738 → 7.270 | 10.55 / 43.20 / 56.06 | 29.43 / 55.48 / 63.73 | 19.821 → 16.300 |
| 2 | 4306 | 6.489 → -0.670 | 31.00 / 62.15 / 71.64 | 61.73 / 84.95 / 88.46 | 13.072 → 7.117 |
| 3 | 26134 | 2.757 → -4.401 | 46.72 / 83.54 / 88.62 | 71.31 / 92.14 / 94.64 | 4.061 → 1.548 |

| 可視camera | 平均HDR50体積 m³（旧→新） | 平均HDR90体積 m³（旧→新） | 平均HDR95体積 m³（旧→新） |
|---|---:|---:|---:|
| all | 19.168 → 3.260 | 118.845 → 20.964 | 192.990 → 33.629 |
| 0 | 40.869 → 12.439 | 220.543 → 67.910 | 336.672 → 104.438 |
| 1 | 164.007 → 58.146 | 1064.852 → 376.448 | 1786.410 → 607.220 |
| 2 | 46.932 → 1.945 | 287.115 → 17.781 | 458.130 → 29.785 |
| 3 | 5.873 → 0.084 | 37.849 → 0.667 | 62.118 → 1.120 |

NLLは全camera層で改善し、分布を広げただけの被覆改善ではない。
ただしHDR50は全体42.73→67.69%で過大被覆、特に3cameraで71.31%。1cameraのHDR95は63.73%に留まり、混合平均RMSE16.30mも大きい。
HDR95体積の全体中央値は31.674→0.026363m³で、平均192.990→33.629m³と大きく異なる。少数の広い分布が残り、代表値を混同しない。
Hの既定や全125成分を変更する根拠にはせず、今回もGT密度・被覆・費用を併記する。

同じ16valのbaselineは以下。数値は無調整で全frameを評価した。

| bank / method | RMSE m | gap m | event±5 m | 加速度p95 m/s² | 再投影p95 px | behind / 17,528 |
|---|---:|---:|---:|---:|---:|---:|
| control / mixture_mean | 7.363 | 4.931 | 6.895 | 25943.2 | 239.59 | 0 |
| control / mixture_mean_rts | 6.972 | 4.565 | 6.150 | 2020.9 | 202.86 | 0 |
| anchored / mixture_mean | 4.994 | 3.480 | 4.439 | 22384.5 | 82.08 | 0 |
| anchored / mixture_mean_rts | 4.408 | 3.020 | 3.318 | 669.4 | 74.81 | 1 |

新RTSはRMSE/加速度/正depth再投影p95を改善するが、behindが0→1件となった。再投影は条件付きで、全軸での優位や実データの受入を主張しない。
T128診断は旧bankでの固定checkpointを測った別実験で、新bank上のflow/regression精度は未測定。両表を直接同条件の比較として混ぜない。

80件のreader/float32 3D SPD/全125成分・presence質量を確認。新bankの保存0重み1,459,878成分も配列に残し、削除していない。全34,059frameはHの収束未評価として保持した。重み和最大誤差4.38e-8。
採点はcontrol510.39秒/RSS0.952GB、anchored495.31秒/RSS0.951GB。最低MemAvailableは11.72/15.19GB。
[回収台帳](../../runs/run-i936-anchored-dev-comparison-r12-s936/collection.json)で全192保存NPZ（各80条件＋16baseline）とbank/reportのSHAを照合した。test NPZ配列読込0、GPU0。

次は#935最終回収を待って640件の生成日程を決める。モデル比較は学習とvalidationの文脈を揃えて新bankで固定し、その後に更新数/physics等の因子を分離する。

## 640ラリー生成計画（未起動）

正確な展開設定・入力SHA・新出力path・コマンドは[計画JSON](../../runs/run-i936-anchored-dev-comparison-r12-s936/pilot-plan.json)に固定した。
正本の`dataset_plan.yaml`（SHA `f76b4642ccab03cd0837cd5403b2a46b615100675d24808b65b48cb39461dcc4`）を`--mode pilot`で使い、
512train/64val/64test、seed936、4workers/native1、同じbank/report、H、最大512frameを保つ。
新directory `data/ball_refiner/i936-pilot-h-anchored-s936` は作成していない。

split加重外挿はserial worker elapsed **32.318時間**、理想4worker wall **8.079時間**。
共有負荷/IOを含むworker経過時間であり、純CPU実測時間とは区別する。
予約案は**4workersで8.1〜10時間（専有換算32.3〜40core-hours）、RAM4〜6GB、disk2.0GB**。
NPZ予測1,420,031,172 bytes＋metadata/progress約13,648,607 bytes。
seedごとのsolver費用や有限proposal枯渇は線形外挿で保証しない。
#935最終seed/execute-load結果をorchestratorが回収して予定するまで起動しない。別bank採用ならこの計画を改版する。

## 検証と実施しなかった範囲

新11tests（単峰/多峰NLL/HDR体積・0重み・unpaired拒否）、sampler/学習接続を含む92tests、
configuration audit、ruff/mypy/pre-commit成功。生成中の1CPU制限を優先しpytestは-n0で実行した。
[code 66d11166のCI](../../runs/run-i936-anchored-dev-comparison-r12-s936/code-ci.json)は全成功。
同runの[T128規則達成](000018-run-i936-context-t128-r12-s936.md)によりphysics比較のGPU jobは不要、0件。
640件生成、test品質評価、実Meiji評価、pipeline統合は行わない。学習ではないためTensorBoard曲線はない。
