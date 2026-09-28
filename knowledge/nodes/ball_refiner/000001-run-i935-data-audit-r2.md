---
id: run-i935-data-audit-r2
type: run
task: ball_refiner
sequence: 1
recorded_at: '2026-09-28'
title: 2D refiner用のamodal教師と既存文脈coverageの監査
issue: 935
provider: codex
date: '2026-09-28'
status: done
config:
  dataset: ball-mix-v1
  pose_threshold: 0.5
  supervision: single_observed_or_explicit_out_of_frame
metrics:
  clips: 429
  frames: 196262
  train_observed_frames: 67462
  train_absent_frames: 1063
  meiji_camera_clips: 171
  meiji_complete_context_clips: 35
  saturated_pose_slots: 323
  total_pose_slots: 655468
repro:
  commit: d53db5e995b10ba12d3870ca295f59663c207b8d
  branch: campaign930/i935-2-data-training
  command: .venv/bin/python -m src.tasks.ball_refiner.scripts.audit_data --store /home/kamimura/projects/tennis-lab/data/ball_detection/ball-mix-v1 --meiji-context-root /home/kamimura/projects/tennis-lab/outputs/player_association/evaluate/meiji_clips/i933-observe-v1-20260927/stores --output /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-2-data-training/outputs/ball_refiner/analyze/data_audit/i935-run2-v2
artifacts:
  run_dir: knowledge/runs/run-i935-data-audit-r2
parents: [run-i934-meiji-holdout-e0-r7]
relations: []
papers: []
tags: [data-audit, amodal-supervision, context-coverage]
---

## 結論

`ball-mix-v1`の全429 clip・196,262 frameを監査し、元splitを保持して単一球のamodal教師を
組み立てられた。一方、既存Meijiのpose/courtは一部しか生成されていない。
初期pilotは文脈なしを基準にし、fullの比較前に文脈生成を完了させる。
学習・精度評価は実施しておらず、TensorBoard曲線・checkpoint・GPU jobはない。

## 教師と母数

| source/split | 全frame | observed位置教師 | 明示的out_of_frame | 可視球なし・amodal不明 |
|---|---:|---:|---:|---:|
| TrackNet train | 16,118 | 15,405 | 0 | 637 |
| Meiji train | 25,905 | 21,871 | 0 | 0 |
| chat train | 63,600 | 30,186 | 1,063 | 23,711 |
| Meiji val | 26,961 | 23,007 | 0 | 0 |
| Meiji test | 36,006 | 28,806 | 0 | 0 |

完全な全source/split・理由別内訳とclipのhashは[監査JSON](../../runs/run-i935-data-audit-r2/audit.json)に保存。
表の残りはunreviewed・unresolved・interpolated・occlusion_estimatedで、主教師にはしない。
実storeの複数instance frameは0件だが、将来の入力を想定してテストで除外を検証した。
trainの存在既知frameは正例67,462、負例1,063で偏りが強い。Meijiには確定負例がないため、
Meijiの存在BCEだけから不存在の較正を主張できない。
空frameをamodal負例に昇格させるには、元sourceで明示的な不存在を確認する別工程が必要。

座標はstoreの縮小率を戻し、sourceの`(W-1,H-1)`で正規化した。
時刻は元の整数PTS差とtime_baseから作り、nominal FPSで置き換えない。
Meijiの注釈はChatGPT支援の視覚レビューで、独立した人手GTではない。
本監査ではtestのラベルの意味・母数を確認しただけで、refiner予測や精度を見て設定を選んでいない。

## 文脈のcoverageと特徴契約

| Meiji split | camera-clip数 | poseあり | courtあり | 両方あり |
|---|---:|---:|---:|---:|
| train (video_002) | 72 | 12 | 12 | 12 |
| val (video_000) | 36 | 12 | 12 | 12 |
| test (video_001) | 63 | 11 | 12 | 11 |

読込対象は#933の`i933-observe-v1-20260927/stores`にある採用済み成果物のみ。
media hash・source解像度・frame数/FPS・descriptor/配列checksum・現行依存関係を確認した。
poseには独立したPTS列がないので、同一mediaの全frame indexをstoreのPTSへ束縛する。
元動画やJPEG全体の再hash・再decodeはしていない。未生成はnot_generatedとし、
実行済み検出なしのmaskへ変換しない。35件は偏ったsubsetであり、全splitとの公平なfull比較には使えない。
TrackNet/chatのpose/courtは未監査で、存在しないとは断定しない。

既存courtは`camera_view_v2` KP14・frame 0。20点への推測変換は不要で、#955のYAML既定値を14に合わせた。
ViTPoseの肘/手首655,468 slot中323件はscore>1で、最大1.04232097だった。
上流実装はheatmap peakを保存しており、確率ではない。refinerの有界featureへ`min(score,1)`で
写すことを明示し、各clipに変換名・変換件数・元maxを保存した。確率較正とは呼ばない。
元成果物は変更していない。仮の20点や架空のposeを生成して補完する経路もない。

## 検証と次の実験

新規CPU14テストは、空/unknown/複数球/推定ラベル、縮小座標・不等間隔PTS、
source不一致・配列破損・frame 0以外のcourtの拒否、pose joint順序・検出mask、
raw score変換の記録、CLIの全frame母数と上書き拒否、groupのsplit漏洩を確認した。
精度・学習収束・GPU・pipelineのGMM保存/load-onlyは未検証。

次はft-e13を凍結し、全frameのtop-K・局所patchをhash付きcacheへ保存する。
文脈なしbaselineのwindow/runnerとCPU optimizer接続を確認した後、共有queueでpilotを行う。
full/poseなし/文脈のみの比較に進む前に、同一母数で文脈生成を完了させる必要がある。
学習戦略のユーザー合意は未取得で、[暫定判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5867545149)に従う。
