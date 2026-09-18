---
id: run-slcs-meiji-v8-root-replay-v1
type: run
title: 'Meiji v8本番補正照合: P0不変の想定で停止'
provider: codex
date: '2026-09-18'
status: failed
config:
  diagnosis: CPU production refine_scene; hips control vs hips_and_shoulders candidate
  source_data: four completed meiji_rgb_v7 raw scenes
  threshold_changes: false
metrics:
  clips_with_saved_comparison: 2
  control_arrays_exact: true
  ball_and_yaw_unchanged: true
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v8-root-replay-v1
  output_dir: outputs/tennis_scene/analyze/meiji_v8_root_replay/s42-001
  diagnostics: knowledge/runs/run-slcs-meiji-v8-root-replay-v1/results.json
parents:
- run-slcs-meiji-root-support-probe-v1
relations: []
tags:
- slcs
- meiji
- cpu-diagnosis
- root-support
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: b494d168721f20688dc47fc6f105ec67828d3b62
  branch: codex/slcs-real-rgb
  command: env CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -B knowledge/runs/run-slcs-meiji-v8-root-replay-v1/probe.py --output-dir
    outputs/tennis_scene/analyze/meiji_v8_root_replay/s42-001
---

## 考察 / Findings

### 要約
本番refine_sceneを用いた照合で「candidateでもP0位置は全clip不変」とした診断scriptの想定が誤っており、3本目で停止した。実装の例外ではない。保存JSONは中断時のrunningを保持するが、processは終了コード1で終了した。

### アーキテクチャ詳細
旧raw sceneへhipsとhips_and_shouldersをそれぞれ適用。旧保存refinedとの位置・全関節・ball・yaw・label_quality一致を検査した後、候補P0不変をassertしていた。

### メトリクスの解釈
2本分の比較を保存した。001/000のP0は保存済み独立CPU候補にも最大約0.32mの変化があり、1m超が0frameという結果を「全く変化しない」と誤解した。原データ・teacherは変更していない。

### アーキテクチャ⇄メトリクスの因果考察
このassertはモデル契約ではなく、探索結果の誤った要約だった。削除して変化量を両選手に対して報告し、旧条件・ball/yaw一致と固定coverage判定は維持する。

### 既存実験との比較
親runのJSONが既にP0最大0.3202mを記録しており、実装後の新しい現象ではなかった。

### 次に有効な実験
同じ4clipで診断を修正して別runへ再実行する。失敗した本runのsnapshotとsource commitは残す。
