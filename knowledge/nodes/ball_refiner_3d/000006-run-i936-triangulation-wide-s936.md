---
id: run-i936-triangulation-wide-s936
type: run
task: ball_refiner_3d
sequence: 6
recorded_at: '2026-09-30'
title: 広いpriorでの正depth制約と成分別A/B併用の検証
provider: codex
status: done
config:
  prior_mean_m:
  - 0
  - 0
  - 2
  prior_variance_m2:
  - 36
  - 144
  - 9
  components: 64
  laplace:
    max_components: 64
    max_nfev: 100
  voxel:
    initial_cells: 16
    levels: 7
    refine_cells: 512
    prior_sigmas: 4.0
  trials: 108
  diagnostic_frames: 10
metrics:
  hybrid_failures: 0
  main_frames: 108
  main_nll_m3: -0.6143428411397144
  main_coverage95: 0.9814814814814815
  regression_frames: 10
  regression_failures: 0
artifacts:
  run_dir: knowledge/runs/run-i936-triangulation-wide-s936
parents:
- run-i936-synthetic-smoke-v2-s936
- run-i936-triangulation-abc-s936
relations: []
papers: []
tags: []
issue: 936
date: '2026-09-30'
repro:
  commit: d0dacaf5
  command: bash knowledge/runs/run-i936-triangulation-wide-s936/repro.sh /absolute/new/output
---

広いpriorでは旧Aをそのまま使えない。正depthを保つAと、非正則成分を明示的に
体積積分するHを実装した。Hは全118入力で64成分を保持して成功し、
同じ設定の12-rally smokeへ進む。**境界近くの1入力でvoxel予算への感度が残り、
積分が数値的に収束したとは主張しない**。本学習・本番方式の確定とは区別する。

## 根本原因と修正

run 2の失敗10frameの全成分を旧solverで再評価した
[raw fits](../../runs/run-i936-triangulation-wide-s936/legacy-diagnosis.json)では、
背後への収束14成分と未収束1成分を記録した。
未収束もval-00002/frame0/cameras(0,2)/components(1,0)で、100評価後のdepthは
34.7977 / **−1.46466m**、cost60.5245、optimality0.00565。
透視投影は負depthでも計算でき、無制約最小二乗がcamera planeを越えた。
境界に向かう組合せも、正規な内点MAPとして扱っていた。

修正Aは正depth半空間内のline searchで全試行点を保持する。
1e-4mの数値近接面、1e-3mの境界検出、depth方向3σの局所tail検査を明示。
通常Laplaceが不適切な成分は理由付きerrorにする。Hはその理由を受けた場合だけ、
指定したBの積分evidence・Gaussian momentsを同じ組合せに割り当てる。
prior置換・成分削除・seed選別・jitterはない。frame×componentの方式codeを保存する。
数値定義・tail/box近似の正本はgeometry README。

## 比較条件

固定seed936の12ラリーを全て使った。各ラリーの等間隔8frameと共有gap中央を
前もって指定した108frameが主比較。前run失敗10frameは別cohortで、成功率へ混ぜない。
入力そのものをcases.npz、計画/seed/物理棄却履歴/校正SHAをcases.jsonへ保存。
3D・2Dは合成で、Meijiから読むのはcamera校正のみ。K=3、prior=(0,0,2)m、
variance=(36,144,9)m²、mean error scale=.25をrun 2から変更していない。
物理シミュレータの既知の提案棄却は全履歴を保存し、三角測量の成否で再標本化しない。

B/Hのvoxelはinitial16、7levels、各level最大512細分化cell、prior±4σ。
HDRは1024 samples、Cは128 particles。Cは正depthの内点solverを各粒子へ使う
限定したrandomize-then-optimize実装で、1粒子の失敗でframeを失敗にする。
Cの結果をsampling一般の否定に使わない。旧A0はc800d9f5のコードをbundleに凍結。
CPU/native thread=1で、他laneの負荷があり厳密な専有benchmarkではない。

NLLは自然対数（密度単位m⁻³）。NLL欄は成功例のみであり、失敗率が違う行の
優劣をその平均だけで比べない。coverageは全108件を分母とし失敗を未被覆と数える。
早期失敗の時間は成功処理の速度ではない。

| 方法 | 成功/108 | 3D NLL（成功分） | 95% HDR（全入力） | median / p95 ms |
|---|---:|---:|---:|---:|
| A0 | 89/108 | -1.244 | 82.4% | 95.59 / 147.39 |
| A | 0/108 | 算出不可 | 0.0% | 0.98 / 1.61 |
| B | 108/108 | -0.389 | 97.2% | 331.12 / 421.46 |
| C | 23/108 | 0.295 | 21.3% | 49.05 / 181.78 |
| H | 108/108 | -0.614 | 98.1% | 359.24 / 1093.60 |

前run失敗cohortではA0/A=0/10、B=10/10（NLL0.005、95% coverage90%）、
C=1/10、H=10/10（NLL−0.978、coverage100%）。
Hの主比較NLLはBより0.226低いが、相関する少数ラリーでの差であり有意差とは扱わない。
教師誤差.25σに対して広い予測分布なので、高coverageを較正の成功とは呼ばない。

## 予算感度と却下した試行

12ラリーの先頭frame・共有gap中央・前run失敗10frame、計34frameで、
initial24/8levels/1024cellsへ増量した。BのNLL差の平均−0.000514、絶対差p95 0.0964。
Hの絶対差p95は0.00345だが、val-00003/frame130（64frame gap）だけ0.9683変化した。
同例のH NLLはinitial16で1.4751、24で2.4434、32/9levels/2048cellsで1.2072。
これを収束確認や「小差」とは呼べない。非正則・細い光線状分布に対する
軸固定cell積分と単一組合せのGaussian moments近似の限界を残す。

比較用にcameraのray座標と正depth区間のtruncated-normal quantileで積分する試行も保存した。
12 angular×48 depthでは主108件のうち7件がrank不足の共分散で失敗。
同じframe130のNLLは5.058→24×96で−0.950と不安定だったため採用しない。
experimental_ray_volume.py/experimental_hybrid.pyとraw結果は再現用の研究bundleだけにあり、
generatorのfallbackや本APIには接続しない。
truncated-normalの境界標準化は[Scipy公式仕様](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html)に従った。
単眼境界の1D積分照合でも96点の共分散誤差0.00270が許容0.002を超えた負例を保持する。

## 暫定判断と検証範囲

generatorは全64成分を保持できるH(initial16/7levels/512cells)を明示してsmokeを行う。
これはsmokeを成立させる工学上の採用であり、境界posteriorの精度保証ではない。
全量生成・本学習の前には、この積分感度と#935からの劣化較正をorchestratorが判断する。
正確な全体表とmethod countsはsummary.json、全入力の成否はfinal.jsonが正本。
源codeはd0dacaf5に固定。通常geometry/refiner 146 tests、ruff/mypy、
strict configuration/path audit 99境界が成功した。単眼/共役極限、全subset質量、
背後固定fixture、境界積分の有限予算比較、空の正depth領域の明示失敗を検査した。
GPU、実Meiji球、拡散学習は実行していない。TensorBoardはない。
