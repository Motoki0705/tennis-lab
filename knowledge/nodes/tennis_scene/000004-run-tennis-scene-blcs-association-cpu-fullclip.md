---
id: run-tennis-scene-blcs-association-cpu-fullclip
type: run
task: tennis_scene
sequence: 4
recorded_at: '2026-09-23'
title: BLCS associationの全clip統合CPU検証・本番品質未達
provider: codex
date: '2026-09-23'
status: done
config:
  model:
    dropout: 0.1
    ffn_dim: 1408
    ffn_type: swiglu
    hidden_dim: 512
    max_identities: 10
    name: blcs_view_association
    num_heads: 8
    num_slots: 4
    num_stages: 12
    rope_dim: 64
  checkpoint_epoch: 14
  device: cpu
  OMP_NUM_THREADS: 2
  scenes:
  - scene_000686
  - scene_000270
  views:
  - 3
  - 5
  min_probability: 0.5
  min_assignment_gap: 0.6931471805599453
metrics:
  finite_cases: 6
  side_accuracy: 0.5
  identity_accuracy_including_abstentions: 0.007926455566905004
  accepted_fraction: 0.013646578140960163
artifacts:
  run_dir: knowledge/runs/run-tennis-scene-blcs-association-cpu-fullclip
  metrics: knowledge/runs/run-tennis-scene-blcs-association-cpu-fullclip/metrics.json
  log: knowledge/runs/run-tennis-scene-blcs-association-cpu-fullclip/stdout.log
  checkpoint: /home/kamimura/projects/tennis-lab/outputs/blcs/train/blcs_view_association/global_mha_mhc_d512_s12_e60_20260922/logs/version_1/checkpoints/blcs-epoch=14.ckpt
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/automatic_association/20260923-integration
parents:
- run-blcs-association-refactor-512-smoke
relations: []
papers: []
tags:
- association
- cpu
- full-clip
- negative-result
- integration
session: 01a0c384-7833-7021-868a-22a9433bfc02
repro:
  commit: 5d26aa0de6b2592fddbb2433df331ba70768b32b
  branch: codex/tennis-scene-auto-association
  command: env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 /home/kamimura/projects/tennis-lab/.venv/bin/python
    tests/benchmarks/association_integration.py --task blcs --checkpoint /home/kamimura/projects/tennis-lab/outputs/blcs/train/blcs_view_association/global_mha_mhc_d512_s12_e60_20260922/logs/version_1/checkpoints/blcs-epoch=14.ckpt
    --scene-root /home/kamimura/projects/tennis-lab/data/blcs/multi_object_camera_view_v2
    --scenes scene_000686 scene_000270 --views 3 5 --device cpu --boundary-padding
    --output /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/automatic_association/20260923-integration/cpu-blcs-5d26aa0d.json
---

## 結果

新しいraw観測APIを、再生成済みCourtV2のtest scene全体へ接続した。BLCSのepoch 14 checkpointをCPU・2 threadで実行し、6ケースすべてが有限値で完走した。カメラは保存順の先頭3/5 view、referenceはcamera ID辞書順先頭。GT sideとphysical identityは採点だけに使い、モデル入力へ渡していない。

side正解率は0.5000、棄却も不正解に数えたID正解率は0.007926で、採用基準に達しない。IDは全view/time共通のHungarian対応付けで採点した。モデルの一対一割当だけでなく確率・次善割当差による棄却を適用しているため、従来の学習時raw identity accuracyと直接比較しない。追加paddingケースは数値健全性の確認に使い、集約精度の分母からは除外した。FPを加えていないsource観測なのでFP precision/recallは未定義であり、未測定をFP崩壊の根拠にしない。

## 条件と限界

2 sceneを長さ境界の確認用に選んだ小規模診断であり、test全体の精度、GPU性能、実動画E2E、3D精度を示さない。Global MHA+mHC、幅512・12 stage・8 headを変更していない。短いclipは512へpadding、長いclipは全時間軸のまま約1024まで処理した。入力ファイルとcheckpointのSHAはrepro.json、各caseのframe数・logits・時間はmetrics.jsonに保持した。学習やTensorBoard曲線はこのrunでは生成しない。

## 判断

以前の512-frame GPUスモークは実行経路の確認だった。本runも配線と長い入力の処理は確認したが、モデル品質の合格ではない。本番のassociation配布重みへは昇格しない。既存のPLCS/BLCS 3D baselineの位置誤差とも比較できない。

次は、60epoch学習のCUDA illegal memory accessを別途解消し、validationでside両クラスrecall、IDの正解率と採用率を確認する。閾値の調整にこのtest subsetを使わず、固定した条件でtest全体・実動画の校正と再構成有効率を評価する。
