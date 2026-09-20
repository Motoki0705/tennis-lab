---
task: slcs
sequence: 101
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-v8-observe-repair-v1
type: run
title: Meijiの3cameraを再生成し全168cameraのproducerと旧新配列を監査
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-19'
status: done
config:
  stage: observe
  clips:
  - video_002/clip_005
  - video_002/clip_006
  - video_002/clip_012
  regenerated_camera: cam2
  observation_directory: tennis_scene/precompute/meiji_dino_vitpose/s42-004
  checkpoint_pins: build_slcs_dataset.yaml
metrics:
  regenerated_cameras: 3
  regenerated_camera_frames: 811
  quarantined_files: 12
  audited_cameras: 168
  untouched_cameras_byte_identical: 165
  compared_npz_archives: 6
  array_differences: 0
  changed_metadata_fields: 3
  stable_input_hashes: 691
  audit_errors: 0
repro:
  commit: 41020d13080c5c1765c218924bbbb4d2aee2ef84
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m src.tennis_scene.scripts.build_slcs_dataset stage=observe paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
    'clip_ids=[video_002/clip_005,video_002/clip_006,video_002/clip_012]' output_dir=tennis_scene/generate/meiji_rgb_v8/s42-002
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v8-observe-repair-v1
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v8/s42-002
  audit: knowledge/runs/run-slcs-meiji-v8-observe-repair-v1/audit.json
  quarantine: knowledge/runs/run-slcs-meiji-v8-observe-repair-v1/quarantine.json
parents:
- run-slcs-meiji-v8-observe-v1
relations: []
tags:
- slcs
- meiji
- observation
- provenance-repair
---

## 考察 / Findings

### 要約
不一致のあった3cameraを元の検出設定で再生成し、全168cameraの保存producer SHAが固定pinsに一致した。対象3cameraのraw/people全配列は旧結果と完全一致し、対象外165cameraのpeople配列とreceiptはbyte一致した。観測の来歴監査は通過したが、間欠的な不一致の根本原因は未確定である。

### アーキテクチャ詳細
実行commitは41020d13080cで、DINO/ViTPoseのcameraごとの固定pin照合・推論前後照合・兄弟receipt照合を含む。元12fileはpreservation.jsonと独立コピーのdual SHAを再確認してquarantine.pyで隔離した。metadataを直接書き換えず、対象3clipのcam0/1は検証再利用、cam2のDINOとViTPoseを同じseed・設定で実行した。Court/ball/PLCS/BLCSの条件は変更していない。

### メトリクスの解釈
再生成した238+255+318=811camera-frameの全6NPZでkey・shape・dtype・値が完全一致した。metadata差は005/006のdetector_sha256、012のpose_sha256の計3fieldだけで、raw receiptに差は無い。全56clip168cameraのschema・有限値・mask・観測支持・source detection ID・raw/people SHAと設定を監査した。旧inventoryで対象外165cameraのpeople NPZ/receipt計330fileは不変。監査入力691fileの前後dual SHAも一致した。最終教師coverageやモデル精度は測っておらず、学習曲線は無い。

### アーキテクチャ⇄メトリクスの因果考察
この再実行では同じ配列が得られ、保存producer記録の不一致を解消した。配列一致は当該3cameraの再現結果であり、過去のhashが変わった理由、実行環境全体の正常性、モデルのメモリ内容を証明しない。期待SHA照合と不一致時の停止を維持する。

### 既存実験との比較
親runは全56clip生成後の3receipt不一致により採用監査がfailedだった。本runではその12fileを保存・隔離し、生成をやり直して全体を再監査した。旧runの失敗記録は保持する。教師生成を止めた理由のうち観測producerの不一致は解消したが、Courtの局所白線ずれは別途確認中である。

### 次に有効な実験
同じCourt checkpointを使うcrop比較の画像・数値を確認し、校正recipeを決定する。その後にMeiji全体の3D教師とstrict品質確認へ戻す。監査CLIのCPU6テストとruff/mypyは成功した。再実行には本bundleのaudit.pyへcacheと新規output JSONを指定する。audit.jsonは実行時の入力hashと差分を保存し、quarantine.jsonは元ファイルの移動先を示す。
