---
task: slcs
sequence: 112
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-v9-observation-reuse-v4
type: run
title: DINO/ViTPoseの実測SHA差を明示警告したMeiji v9観測再利用
provider: codex
date: '2026-09-19'
status: done
config:
  device: cpu
  cuda_visible_devices: ''
  omp_mkl_threads: 4
  checkpoint_warning_roles:
  - dino
  - vitpose
  source_observations: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004
  target_observations: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-005
  automatic_retry: false
metrics:
  elapsed_seconds: 134.96623973200622
  attempts: 1
  clips: 56
  raw_cameras: 168
  reused_people_cameras: 142
  recompute_cameras: 26
  recompute_clips: 20
  audited_inputs: 2064
  inputs_before_prepublication_after_equal: true
  checkpoint_warnings: 0
  published_file_counts:
    '*_detections.npz': 168
    '*_detections.metadata.json': 168
    '*_people.npz': 142
    '*_people.metadata.json': 142
    '*_people.reuse.json': 142
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-observation-reuse-v4
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_observation_reuse/s42-004
  log: knowledge/runs/run-slcs-meiji-v9-observation-reuse-v4/stderr.log
  summary: knowledge/runs/run-slcs-meiji-v9-observation-reuse-v4/summary.json
  recompute: knowledge/runs/run-slcs-meiji-v9-observation-reuse-v4/recompute.json
parents:
- run-slcs-meiji-v9-observation-reuse-v3
- run-slcs-vitpose-redownload-v1
relations: []
tags:
- slcs
- meiji
- integrity
- cpu
- explicit-warning
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 0e2c252a86fd2068849ac9a8d59636d39db1b530
  branch: codex/slcs-real-rgb
  command: env PYTHONPATH=. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python knowledge/runs/run-slcs-meiji-v9-observation-reuse-v4/reuse.py
    --source /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004
    --target /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-005
    --output /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_observation_reuse/s42-004
---

## 考察 / Findings

### 要約
明示的なDINO/ViTPose checksum警告policyを適用したv4 CPU driverを1回実行し、v8観測をv9 Courtへ再利用した。56 clipのraw検出168 cameraと人物観測142 cameraを公開した。26 camera（20 clip）は人物選択が変わるため再計算が必要で、peopleを公開しなかった。今回はcheckpoint警告0件で完了した。

### アーキテクチャ詳細
v1の選択・court・schema・copy helperを読み取り専用importし、historical moduleのglobalsを変更せずv4の実行処理から使用した。媒体・旧producer receipt・court・コピー先bytesを厳密検証し、6選択配列のdtype/shape/value完全一致の場合にのみpeopleを再利用した。許可された2checkpointのlive pinおよび入力前後差のみ明示警告へ分離し、実測SHAをauditへ保持する。今回は例外を使う必要がなかった。新people metadataは旧producer pinsを保持し、別reuse receiptへdeclared model identityと警告証跡pathを追加した。

### メトリクスの解釈
CPU実行のmonotonic elapsedは134.966秒、exit code 0、試行1回。raw NPZとreceiptは各168、people NPZ・metadata・reuse receiptは各142を再帰的に実在確認した。2064入力のbefore/prepublication/afterは完全一致し、DINO/ViTPoseも3段階で固定SHAに一致した。空のcheckpoint_warnings.jsonlも証跡として保存した。再計算cameraとclipの正確な一覧はrecompute.jsonを参照。GPU実行・学習はなく、収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察
Court v9で人物選択が変わらなかった142 cameraは、元の人物観測bytesをそのまま使用できる。26 cameraは選択変更または選択拒否を検知したため、古いpeopleを流用せず再計算へ分離した。今回のSHA一致はこのrunの観測であり、過去の読取差異の原因が解決したことまでは示さない。

### 既存実験との比較
v3はDINO初回固定SHAで停止した。本runはViTPose再取得済みの状態に明示警告policyを導入して実行したが、警告は発生せず、全入力監査を通過して再利用を公開できた。過去のfailed runとreceiptは書き換えていない。

### 次に有効な実験
recompute.jsonに列挙した26 cameraの人物観測をGPU queue経由で再計算し、品質監査と3D教師生成を進める。このrunは再計算を実施していない。
