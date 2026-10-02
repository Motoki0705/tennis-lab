---
id: run-i935-context-shard-00059-tracknet-r12-20260928
type: run
task: ball_refiner
sequence: 8
recorded_at: '2026-09-29'
title: TrackNet最長901frameの文脈生成が成功、累計31trackと周辺人物を保持
issue: 935
provider: codex
session: 01a0e875-821a-7933-ab59-3a7e497a165c
date: '2026-09-28'
status: done
config:
  source: tracknet
  split: train
  clip_id: tracknet/game7/Clip4
  shard_index: 59
  max_tracks: 1024
  person_region_policy: full_frame
  pose_threshold: 0.15
  court_frames:
  - 0
  timeout_seconds: 1080
  plan_sha256: 1730b845df7348a9f800575d8fbf0442b04324fcfa27d10ee87e005081af3486
metrics:
  completed_clips: 1
  completed_frames: 901
  cumulative_tracks: 31
  max_simultaneous_observed_tracks: 15
  observed_pose_crops: 11137
  valid_arm_slots: 43964
  outside_image_valid_arm_slots: 24
  saturated_arm_slots: 0
  total_dense_arm_slots: 111724
  court_valid_points: 14
  stage_seconds: 927.2471149750054
  queue_wall_seconds: 967
repro:
  commit: db28d14d3a965fb0b6d7971cf364419ef980c137
  branch: campaign930/i935-9-context-shards
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: '''timeout'' ''--signal=TERM'' ''--kill-after=15s'' ''18m'' ''bash'' ''/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-9-context-shards/tests/benchmarks/ball_refiner_context_shard.sh''
    ''/home/kamimura/projects/tennis-lab'' ''/home/kamimura/projects/tennis-lab/data/ball_refiner/detector-ft-e13-trainval-r3-20260928''
    ''/home/kamimura/projects/tennis-lab/data/ball_refiner/context-fullframe-shards-r12-20260928/plan/plan.json''
    ''/home/kamimura/projects/tennis-lab/outputs/ball_refiner/generate/context/i935-context-shards-r12-20260928/scene-context.yaml''
    ''/home/kamimura/projects/tennis-lab/outputs/ball_refiner/generate/context/i935-context-shards-r12-20260928/dino_extension''
    ''59'' ''/home/kamimura/projects/tennis-lab/data/ball_refiner/context-fullframe-shards-r12-20260928/shard-00059-a1''
    ''/home/kamimura/projects/tennis-lab/outputs/ball_refiner/generate/context/i935-context-shards-r12-20260928/shard-00059-a1'''
artifacts:
  run_dir: knowledge/runs/run-i935-context-shard-00059-tracknet-r12-20260928
  log: knowledge/runs/run-i935-context-shard-00059-tracknet-r12-20260928/queue.log
  captured_repro: knowledge/runs/run-i935-context-shard-00059-tracknet-r12-20260928/captured-repro.sh
parents:
- run-i935-context-fullframe-pilot-r10-20260928
relations: []
papers: []
tags:
- context
- shards
- long-clip
- quality-audit
---

## 結果と証拠

queue job `1790607540292509890_751990_i935-context-shard-00059-tracknet-r12-20260928` はdone。全901frameの人物検出・追跡、
実観測boxの11,137cropへのViTPose、frame 0 court探索と別Pythonプロセスの保存後検証が完了した。
[検証結果](../../runs/run-i935-context-shard-00059-tracknet-r12-20260928/context-verification.json)と
[全frameのCPU監査](../../runs/run-i935-context-shard-00059-tracknet-r12-20260928/audit.json)は、NPZ checksum・frame/PTS・実JPEG hash・stored/source座標を照合した。
manifest SHA-256は `74a84b793fcf313f4dd3663fb1e41c0f3ee9990bdb9551089067ebcbb5b09729`。
run13で再読込しても同じmanifestとJPEGを保持していた。

累計trackは31、最大同時観測は15。
score >= 0.15の肘/手首が少なくとも一つあるframeは901/901、
有効slotは43,964。これは誰かのposeがあるという意味で、プレー中の人物のrecallではない。
1超の生peakは0 / 111,724 dense肘手首slot、
最大0.998341。dense分母はframe×累計track×4で未観測slotを含む。
観測crop×4の分母は44,548。飽和変換を確率較正とは呼ばない。

[5frameの画像](../../runs/run-i935-context-shard-00059-tracknet-r12-20260928/clip-00071-context.jpg)では主コートの両プレーヤーの腕がある一方、周辺スタッフ・観客を多数含む。frame 0 courtは表示線に概ね沿うが、GTとの照合はしていない。全clipの有効肘/手首で24 slotが画像外にあり、有限な画像外座標を切り詰めず保持する既存契約を確認した。
標本はラベルを参照しない先頭・1/4・中央・3/4・末尾の5frame。
pose/courtの独立GT精度・文脈による改善・ball refinerのholdout指標は測定していない。

検出/追跡 280.11秒、pose 608.87秒、
court 38.26秒、stage合計 927.25秒。
queueのrunningからdoneまでは967秒で18分制限内。起動・identity hash・保存後検証も含む壁時計で、
純GPU時間ではない。source最長clipの1例であり代表的throughputや全clipの上限ではない。

## 再現性と次の実験

生成commit・clean status・元commandを内容不変で保存した。
[元queue再現スクリプト](../../runs/run-i935-context-shard-00059-tracknet-r12-20260928/captured-repro.sh)は履歴資料で、元出力への上書きを拒否する。
全NPZを本bundleのclips/へ保存し、[manifest](../../runs/run-i935-context-shard-00059-tracknet-r12-20260928/context-manifest.json)にmodel/code/JPEG hashを保持した。
共通の[plan](../../runs/run-i935-context-shard-00059-tracknet-r12-20260928/plan.json)・
[scene設定](../../runs/run-i935-context-shard-00059-tracknet-r12-20260928/scene-context.yaml)・
[DINO build設定](../../runs/run-i935-context-shard-00059-tracknet-r12-20260928/build.json)は最初のprobe bundleを正本とする。
DINO binary・重み・raw JPEGは複製せず、元の絶対pathに保持している。
そのため、このbundle単体を任意の環境で動くGPU replay一式とは扱わない。
再生成は同じidentityの資産と新しいcache/report pathが必要で、binary再build時は新しいplanと別runになる。
CPU監査は[既存のaudit_context.py](../../runs/run-i935-context-fullframe-pilot-r10-20260928/audit_context.py)を
生成commit上で実行した。学習でないためTensorBoard曲線はない。

3sourceの比較・全生成の暫定判断・次の検証は
[probe群](000011-group-i935-context-shards-r12-probe.md)に集約する。
