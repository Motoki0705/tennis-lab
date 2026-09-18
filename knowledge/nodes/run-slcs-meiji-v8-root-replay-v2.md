---
id: run-slcs-meiji-v8-root-replay-v2
type: run
title: 'Meiji v8本番補正照合: 旧条件を4clip完全再現'
provider: codex
date: '2026-09-18'
status: done
config:
  diagnosis: CPU production refine_scene; hips control vs hips_and_shoulders candidate
  source_data: four completed meiji_rgb_v7 raw scenes
  threshold_changes: false
metrics:
  clips_with_saved_comparison: 4
  control_arrays_exact: true
  ball_and_yaw_unchanged: true
  candidate_min_player_label_fraction: 0.9756838905775076
  candidate_max_player0_change_m: 0.32019585371017456
  candidate_max_player1_change_m: 1.3133991956710815
  candidate_player1_change_over_1m_frames: 44
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v8-root-replay-v2
  output_dir: outputs/tennis_scene/analyze/meiji_v8_root_replay/s42-002
  diagnostics: knowledge/runs/run-slcs-meiji-v8-root-replay-v2/results.json
parents:
- run-slcs-meiji-v8-root-replay-v1
- run-slcs-meiji-root-support-boundaries-v1
relations: []
tags:
- slcs
- meiji
- cpu-diagnosis
- root-support
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: fdb6aa99580a0b5e92c68bd4254a91457ce3f950
  branch: codex/slcs-real-rgb
  command: env CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -B knowledge/runs/run-slcs-meiji-v8-root-replay-v1/probe.py --output-dir
    outputs/tennis_scene/analyze/meiji_v8_root_replay/s42-002
---

## 考察 / Findings

### 要約
本番refine_sceneのhips条件は保存済み4clipの位置・全関節・ball・yaw・label_qualityを完全再現した。hips_and_shouldersでもball/yawは完全不変で、全選手の教師率は97.57%以上。固定閾値のままMeiji v8で全体品質確認へ進める。

### アーキテクチャ詳細
同じraw sceneを2回独立ロードし、設定差をplayer_root_view_supportだけに限定。source archive/metadataとcourt.npzを二実装SHAで記録。候補の配列・weight・sourceをNPZへ保存した。GPUやモデル推論は使用せず、正規datasetは書き換えていない。

### メトリクスの解釈
最大変位はP0=0.320195854m、P1=1.313399196m。1m超はP1で44frame、P0で0frame。P0は001/000で変化し、他3clipは不変。001/000 P0教師率0.99696→0.99544、P1は0.97416→0.97568。他P1は0.98926→0.98006、0.98656→0.97715、0.98154→0.97692。ball/yawの正確性を示す値ではなく、今回の変更対象外であることの検証。学習曲線なし。

### アーキテクチャ⇄メトリクスの因果考察
旧条件が完全一致するため、新実装の一般的な計算変更による差ではない。独立CPU探索はfloat64へ変換していたが本番はfloat32で、一部境界の棄却数が1frame異なる。これは閾値を緩めず本番計算を採用した結果であり、固定精度・source・configで再現する必要がある。

### 既存実験との比較
v1の誤ったP0不変assertを、両選手の変化を記録する診断へ修正した。旧条件の再現・候補ball/yaw不変・固定coverage合格という必要な検査はそのまま維持した。独立実測3D精度の改善や52未生成clipへの一般化はまだ立証していない。

### 次に有効な実験
新Meiji v8へ全clipを生成して品質レポートと映像で確認する。旧v7と混在させず、支持不足の区間は除外し固定閾値を維持する。
