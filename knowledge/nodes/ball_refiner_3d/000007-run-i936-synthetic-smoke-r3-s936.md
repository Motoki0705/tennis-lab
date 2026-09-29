---
id: run-i936-synthetic-smoke-r3-s936
type: run
task: ball_refiner_3d
sequence: 7
recorded_at: '2026-09-30'
title: 正depth A/B併用で59.94fpsの固定12ラリーsmokeを完走
provider: codex
status: done
config:
  seed: 936
  workers: 4
  method: explicit_component_laplace_voxel_hybrid
  fps_numerator: 60000
  fps_denominator: 1001
  components: 64
  requested_rallies: 12
metrics:
  completed_rallies: 12
  failed_rallies: 0
  total_frames: 4809
  elapsed_seconds: 813.5280487479758
  npz_bytes: 13006215
  output_total_bytes: 13122533
  matched_comparison_frames: 118
  pilot_4_worker_hours: 11.034312375449808
  pilot_npz_bytes: 742866288
artifacts:
  run_dir: knowledge/runs/run-i936-synthetic-smoke-r3-s936
  output_dir: /home/kamimura/projects/tennis-lab/data/ball_refiner/synthetic-3d-i936-smoke-r3
parents:
- run-i936-triangulation-wide-s936
- run-i936-synthetic-smoke-v2-s936
relations: []
papers: []
tags: []
issue: 936
date: '2026-09-30'
repro:
  commit: d0dacaf5895c4e4887e60b82031fcb1b9dac2699
  command: bash knowledge/runs/run-i936-synthetic-smoke-r3-s936/repro.sh /absolute/new/output
---

固定seed936の12ラリー（train/val/test各4件）が全件成功した。
4,809frame、全frameの64成分、64frame共有gapを含む。seedや失敗成分を選別せず、
run 2と同じ物理提案seed/棄却履歴を維持した。追加GPU・diffusion学習・実Meiji球評価はない。

設定・camera/コードSHA・全イベント・各ラリーの時刻/サイズの正本は
[manifest](../../runs/run-i936-synthetic-smoke-r3-s936/manifest.json)。
実行開始時のHEADはd0dacaf5。実行中のknowledge-only commitでも、
生成器が読んだ全source/config/calibration hashが終了時まで一致することを確認した。

## 全ラリー

NPZのサイズは圧縮後のbytes、時間はsimulation・triangulation・保存を含むworkerのwall time。
主実行は4 worker並列で813.528秒。各worker時間の和2,670.203秒と区別する。

| Rally | seed | frames | 共有gap | 全体秒 | 三角測量秒 | NPZ bytes |
|---|---:|---:|---:|---:|---:|---:|
| train-00000 | 3349379694 | 254 | 8 | 195.07 | 193.42 | 681,159 |
| train-00001 | 3716130156 | 497 | 16 | 281.02 | 280.15 | 1,329,645 |
| train-00002 | 669701129 | 512 | 32 | 279.29 | 276.02 | 1,376,919 |
| train-00003 | 1189098374 | 512 | 64 | 281.86 | 278.64 | 1,386,981 |
| val-00000 | 724086207 | 178 | 8 | 108.54 | 108.34 | 475,785 |
| val-00001 | 1583466478 | 512 | 16 | 205.93 | 205.02 | 1,395,879 |
| val-00002 | 1338722038 | 512 | 32 | 234.17 | 232.27 | 1,408,473 |
| val-00003 | 535296226 | 199 | 64 | 159.30 | 159.04 | 529,447 |
| test-00000 | 3741586814 | 343 | 8 | 305.21 | 304.67 | 896,833 |
| test-00001 | 1606478505 | 266 | 16 | 128.51 | 126.31 | 712,218 |
| test-00002 | 883072688 | 512 | 32 | 205.12 | 202.16 | 1,407,336 |
| test-00003 | 2488589600 | 512 | 64 | 286.19 | 284.60 | 1,405,540 |

## 再読込と入力の同一性

[verification.json](../../runs/run-i936-synthetic-smoke-r3-s936/verification.json)と
verify_smoke.pyを保存した。全NPZのSHA/bytes、240Hz原系列からの正確な60000/1001Hz時刻、
nativeイベント秒と最近傍frame、±5frameのphysics mask、full covarianceのfloat32 SPD、
全camera subsetのBernoulli質量、amodal presenceと遮蔽の分離を確認した。
全frame×64成分のmethod code/countとactive cameraでの正depth平均も検査した。

事前に固定した比較の118frameについて、seed・物理提案履歴、合成GT、camera K/R/t、
#935 BallGMM2Dを再復号した全mean/covariance/weights/presenceがbit単位で一致した。
成功した別ラリーへ置換していない。64frame gapは各splitのindex3で検証した。

全307,776成分の内訳はprior=4,809、通常Laplace=241,718、
volume:boundary_laplace_tail=54,825、volume:camera_boundary=6,247、
volume:iteration_budget=177。積分方式を診断に明示し、全ての配列行を保持した。
極小evidenceのfloat32 underflowで0になる重みの件数もrally metadataへ残す。
閾値pruning・成分削除をしたものではない。

## 費用と次の判断

NPZ合計13,006,215 bytes、JSONを含むdata directoryは13,122,533 bytes。
3split別の平均からtrain512/val64/test64へ外挿すると、
累積worker時間158,894.10秒（44.137時間相当）、**理想4 workerで39,723.52秒＝11.034時間**、
NPZ約742,866,288 bytes（0.743GB）。
これは4並列の共有CPU環境で測ったworker latencyからの単純外挿。
逐次実行を実測した値ではなく、並列overhead・他laneの負荷変動・未見seedの物理棄却や
数値失敗は予測しない。最大4 processでの全量生成はorchestratorが別途scheduleする。
**このrunでは640ラリーを生成していない**。

観測した空きRAMは最小17GBで6GB条件を維持。各workerのpeak RSSはrally JSONへ保存。
新規dataとknowledge bundleの総量は10GBを大きく下回る。通常検証は関連146 tests、
型修正後の対象11 tests、ruff/mypy、strict configuration/path audit 99境界が成功した。
TensorBoardは使っていない。

方式比較nodeで記録した、境界1frameのvoxel予算に対するNLL感度は残る。
このsmokeの全件成功を積分精度の収束・#935劣化較正・本学習の承認へ読み替えない。
#936 acceptanceの合成生成はsmokeまで達成で、全量/較正、品質対照、実LOCO、
pipelineとimports撤廃は後続である。
