---
id: run-i936-ray-convergence-r6-s936
type: run
task: ball_refiner_3d
sequence: 10
recorded_at: '2026-09-30'
title: 光線座標の積分で固定12ラリーの収束を再監査
provider: codex
status: done
config:
  source_rallies: 12
  source_frames: 4809
  components: 64
  orders:
  - 12
  - 20
  - 32
  - 48
  - 64
  nll_tolerance_nat: 0.05
  log_evidence_tolerance_nat: 0.05
  mean_tolerance_m: 0.02
  covariance_relative_tolerance: 0.05
metrics:
  frames: 4809
  converged_frames: 4225
  converged_rate: 0.8785610313994594
  nonconverged_frames: 584
artifacts:
  run_dir: knowledge/runs/run-i936-ray-convergence-r6-s936
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/diagnostics/i936-ray-r6-v2
parents:
- run-i936-integration-convergence-r5-s936
relations: []
papers: []
tags: []
issue: 936
date: '2026-09-30'
repro:
  commit: fcda13104cd9d40402b71375091341358e3888fd
  command: See runs/run-i936-ray-convergence-r6-s936/repro.sh; use a new output directory
    for each replay.
---

固定12ラリーの全4,809frameを再積分し、**4,225frame（87.86%）が収束、584frameがcapで未収束、処理失敗0**。
旧監査の収束0/4,809から改善したが、全camera欠損は162/360（45.0%）、別K=4標本は2/9であり、積分問題全体の解決とは扱わない。
閾値・native軌道・全2D GMM・camera・mask・seedを維持し、全64成分と未収束flagを保存した。
これは数値開発用smokeであり、実Meiji球の精度やモデルの独立holdout評価ではない。

## 診断

[格子診断](../../runs/run-i936-ray-convergence-r6-s936/grid-diagnosis.json)の4例では、prior箱は48×96×24m。
最終予算の初期cellは1.5×3×0.75m、最深cellは5.86×11.72×2.93mmだが、
最深levelに達した推定質量は0.024〜1.78%に過ぎなかった。大部分が中間levelに残るため、最小cell幅だけでは精度を判断できない。
代表的な単眼成分の主軸σは0.12/0.16/9.46m、別例は0.20/0.81/8.51m。
cell内RMSが最小主軸σを上回る質量は小さく、単に全質量が巨大cell内にあるという説明でもない。
有限のrefine枠で中間cellに残る細長い分布のmoment誤差が重要だと考える。

単眼のdepth解析積分により、角度次数12→20→32→48→64の差は代表例でほぼ丸め誤差になり、
旧voxelの成分平均から0.22〜0.57m動いた。複数視点の指定例val-00003/frame130も閾値に到達した。
実装契約と近似の限界は[geometry README](../../../src/utils/geometry/probabilistic_triangulation/README.md)を参照。
mode探索にGTを使わず、全active cameraから決定論的に初期化する。

初回8a9a726fの500frame batchは468件保存、409収束/59未収束、32件chunk失敗だった。
test-00000/frame252で、積分cameraを変えた後に別のmodeを再利用していた。
fcda1310で同じ物理位置を新しい座標へ移し、回帰fixtureを追加した。
[初回失敗](../../runs/run-i936-ray-convergence-r6-s936/initial-batch-summary.json)と版・source hashを残し、同じ全入力を新規directoryへ再監査した。

## 全数の結果と費用

数値の正本は[summary.json](../../runs/run-i936-ray-convergence-r6-s936/summary.json)。
全frame identity・全成分の差分/重み/flagはframe_diagnostics.npz、全段階の履歴は出力auditの各chunk JSONに保存する。
NLL/evidence/平均/共分散の隣接差は経験的な誤差推定で、共通して見落とすmode、正則AのLaplace誤差、Gaussian要約誤差の上界ではない。

| 対象 | 収束 / 全frame |
|---|---:|
| 全体 | 4,225 / 4,809（87.86%） |
| 全camera欠損 | 162 / 360（45.0%） |
| イベント前後±5 | 771 / 853（90.39%） |

未収束成分の質量は全frame平均0.000216、p95=0.000821、0.5超のframeは0件。
微小質量でも成分を捨てず、これを厳格な収束の代わりに使わない。
未収束例では最大の平均差8.63mが残る。全体率だけで欠損区間の数値品質を保証しない。

val-00003/frame130は3段階（次数32）で収束。
最終差はNLL0.003185nat、evidence0.000309nat、平均0.001793m、共分散相対0.000777。
GTは停止後にだけ評価し、NLL=1.1260natだった。

平均0.6897秒/frame、p50=0.1328秒、p95=3.1804秒、最大13.8515秒。
累積worker経過時間3,316.86秒、4 workerのbatch wall合計1,145.51秒。
診断やCI修正の中断を除くbatch合計で、run全体の実時間ではない。並行負荷を制御した専用speed benchmarkでもない。
CPU最大4/native thread1、GPU0、available RAM観測最小23GB。

再保存先は`/home/kamimura/projects/tennis-lab/data/ball_refiner/synthetic-3d-i936-smoke-ray-r6`。
全12件を標準readerで再読込し、float32 SPD・正depth・全subset質量・flag整合性を検証した。
出力23,319,731 bytes。[入力不変の照合](../../runs/run-i936-ray-convergence-r6-s936/verification.json)では、
変更対象以外の全配列が一致。旧run5最終条件との混合平均差はp50=0.000178m、p95=0.0980m、最大0.4649mで、276frameが2cmを超えた。

## K=4への適用限界と停止判断

[固定K=4費用標本](../../runs/run-i936-ray-convergence-r6-s936/cost-k4.json)は、旧較正prefix3件の各frame0/36/71を全125成分で計算した。
9件中2件収束、7件未収束、処理失敗0、平均7.228秒/frame。全K=4の率を推定する標本とは扱わない。
test-00003/frame0の[成分診断](../../runs/run-i936-ray-convergence-r6-s936/k4-component-diagnosis.json)では、未収束質量0.001160、
最大平均差6.777mの3視点成分は重み3.55e-26。単眼次数4倍の対照でも未収束で、2〜3視点productに残る。
この1frameの質量を他frameへ一般化しない。

旧96件jobのPGID3812111をSIGTERMで停止し、親・resource tracker・4 workersの残存0を確認した。
停止時9件、32,438,753 bytes。元dataは保持しmanifestをstoppedへ変更、変更前manifest/SHAと停止記録をbundleへ保存。大きい元JSONはlossless gzipとし、archives.jsonで展開後SHAを固定する。
frozen worktreeはlockedのまま。再開はしていない。

同標本による96件の外挿は、平均400.75frameなら理想4 CPUで約19.3時間、全件512frameなら約24.7時間。
余裕込み約30時間・CPU4・GPU0・出力0.8GBを提案上限とするが、K=4未収束のため再開を推奨しない。
[判断コメント](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5901703184)が正本。
次は広い複数視点productに対する適応積分または複数modeのimportance積分を、同じ入力・全成分で比較する。
640件は最終#935出力と収束法を待つ。GPU、実Meiji球評価、pipeline変更なし。TensorBoardは使わない。

## 通常検証

geometry/refiner 169 tests、設定/path関連378 tests、ruff/mypy成功。
新CLIの境界登録漏れによるCI 2件をe87dd49fで修正し、[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/36650051598)は
6,551 passed / 133 skipped、knowledge/webui/labelも成功した。研究記録の最終headのCIとは区別する。
