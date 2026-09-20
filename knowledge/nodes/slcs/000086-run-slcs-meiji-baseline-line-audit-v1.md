---
task: slcs
sequence: 86
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-baseline-line-audit-v1
type: run
title: 'Meiji baseline画素診断: LSD配列形状で停止'
provider: codex
date: '2026-09-19'
status: failed
config:
  device: cpu
  data: video_002/clip_017 cam0
  frames:
  - 0
  - 453
  - 907
metrics:
  exit_status: 1
  saved_images: 0
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-baseline-line-audit-v1
  output_dir: outputs/tennis_scene/analyze/meiji_baseline_line_audit/s42-001
parents:
- run-slcs-meiji-court-visual-qc-v1
relations: []
tags:
- slcs
- meiji
- court
- diagnostic-failure
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 10a8a73edf38348eb6322b901d05329b6b2c7f5d
  branch: codex/slcs-real-rgb
  command: bash knowledge/runs/run-slcs-meiji-baseline-line-audit-v1/repro.sh /new/unique/output
---

## 考察 / Findings

### 要約
白線画素診断の初回は、OpenCV LSDの返却配列を扱うコードのshape想定が誤り、画像保存前にexit 1で停止した。校正精度についての結果は得ていない。

### アーキテクチャ詳細
固定RGB ROIからLSD線分と白帯候補を抽出するCPU診断。元Court・観測・教師へ書き込まない。失敗時のexact source、patch、command、SHA、logを保存し、canonical probeもその実行時bytesを保持した。

### メトリクスの解釈
保存画像0、確定定量結果なし。失敗により入力前後hashのreceiptは保存されなかったため、完了した入力整合検査を主張しない。学習曲線は無い。

### アーキテクチャ⇄メトリクスの因果考察
線分arrayからscalarを取り出してしまい、4要素へのreshapeが失敗した。これは診断実装の不具合であり、画像やCourtモデルの品質不良を示さない。

### 既存実験との比較
親のCourt画像確認は完了済み。本runはその局所的なずれを画素で測る追加処理で、追加の精度結果を得られなかった。

### 次に有効な実験
返却arrayをreshape(-1,4)等で正規化した別run v2で、同じ候補条件・別出力へ実行する。失敗runを成功runで上書きしない。
