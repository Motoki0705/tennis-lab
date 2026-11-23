---
id: run-i2442-meiji-pseudo-association-pilot
type: run
task: player_association
sequence: 7
recorded_at: '2026-11-24'
title: Meiji 疑似正解に向けた Terra 注釈の試行
issue: 2442
provider: codex
date: '2026-11-24'
status: done
summary_impact: none
config:
  model: gpt-6.1-terra
  effort: xhigh
  seed: 2442
metrics:
  singles_known_boxes: 322
  singles_correct_boxes: 320
  singles_player_units: 281
  singles_correct_player_units: 279
  doubles_blind_frames: 16
  doubles_blind_seconds_per_frame: 9.2285625
  proposed_full_sample_frames: 1205
artifacts:
  run_dir: knowledge/runs/run-i2442-meiji-pseudo-association-pilot
  output_dir: outputs/player_association/evaluate/meiji_4cam/i2442-pseudo-association
relations:
  - to: run-i2363-trt-player-video
    rel: related
    note: 使用した既存経路の69 clipと段階別の保存成果物
---

Meiji 3cam の人手確定4 clip・48 frame を、役割とIDを伏せて注釈した。box単位で320/322、人物ID単位で279/281が一致した。meiji_4cam は停止・1視点欠測・確信層の3 clip・16 frame を注釈し、旧IDを示す比較も1 clipで実行した。旧IDを伏せた試行のCLIは平均9.23秒/frame。69 clip の抽出結果は、層ごとの抽出枠を適用して重複を除くと1,205 frameだった。

[試行記録](../../runs/run-i2442-meiji-pseudo-association-pilot/README.md) に、固定入力・選択規則・再現手順・予算見積もりと人の抜き取り案を記録した。ダブルスの参照の誤り率と新旧の追跡・対応方式の性能は今後の評価対象なので、summary の現行判断は維持する。
