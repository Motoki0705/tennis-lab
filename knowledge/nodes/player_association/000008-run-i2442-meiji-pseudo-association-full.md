---
id: run-i2442-meiji-pseudo-association-full
type: run
task: player_association
sequence: 8
recorded_at: '2026-11-24'
title: Meiji 4cam の人物対応疑似正解の実行と棄却
issue: 2442
provider: codex
date: '2026-11-24'
status: done
summary_impact: none
config:
  model: gpt-6.1-terra
  effort: xhigh
  seed: 2442
  show_prior_association: false
  max_parallel: 4
metrics:
  published_clips: 69
  published_frames: 1205
  format_rejected_frames: 0
  annotated_boxes: 34664
  ambiguous_boxes: 328
  elapsed_seconds: 1701.183
  cli_seconds_per_frame: 4.940497
  input_tokens: 3713178
  output_tokens: 300823
  human_review_frames_prepared: 170
  human_review_frames_checked: 0
artifacts:
  run_dir: knowledge/runs/run-i2442-meiji-pseudo-association-full
  output_dir: outputs/player_association/evaluate/meiji_4cam/i2442-pseudo-association/full
parents:
  - run-i2442-meiji-pseudo-association-pilot
---

旧IDを伏せた同期4視点の画像から、69 clip・1,205 frame・34,664 boxの疑似正解を固定datasetへ保存した。69件すべてが初回の形式検査を通過した。バッチは28分21秒、CLIの稼働時間は4.94秒/frameだった。

2026-11-24、生成結果を見たユーザーが品質不足として棄却した。人による170 frameの抜き取り確認は未実施で、層別誤り率は未測定。ラベルと暫定採点値を現行方式の評価・採用の根拠から外した。 [棄却判断・実測・元の実行履歴](../../runs/run-i2442-meiji-pseudo-association-full/README.md)
