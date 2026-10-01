---
id: run-i934-evidence-meiji-clip000
type: run
task: ball_detection
sequence: 19
recorded_at: '2026-09-28'
title: Meiji実clipのheatmap・top-K保存とload-only再開
issue: 934
provider: codex
session: 01a0e57b-9d42-7a60-a1e4-8ab2461e46ea
date: '2026-09-28'
status: done
config:
  checkpoint: ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt
  clip: video_001/clip_000
  artifact_schema: ball_detections.v2
  candidates: {max_candidates: 8, nms_kernel: 5, patch_size: 5}
metrics:
  camera_count: 3
  source_frames_per_camera: 1316
  total_source_frames: 3948
  total_candidates: 31584
  unobserved_frames_with_raw_candidates: 1748
  load_only_camera_count: 3
repro:
  commit: 87de832cb049d5da36cd3df7a1549fa240199db8
  branch: campaign930/i934-3-evidence-contract
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i934-3-evidence-contract
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i934-3-evidence-contract/.venv/bin/python
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i934-3-evidence-contract/tests/benchmarks/ball_detection_evidence.py
    --repo /home/kamimura/projects/tennis-lab --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_001/clips/clip_000
    --report /home/kamimura/projects/tennis-lab/outputs/ball_detection/evaluate/i934-evidence-meiji-clip000-r4-20260928
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-i934-evidence-meiji-clip000
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790557852275223391_1542131_i934-evidence-meiji-clip000-r4-20260928.log
parents: [run-ball-checkpoint-normalization-meiji-20260923]
relations: []
papers: []
tags: [meiji, pipeline, ball-evidence, contract-validation]
---

## 結果と解釈

#934 / #930 の出力契約を、Meiji video_001/clip_000（1920×1080、全1316 frame ×
3 camera）で検証した。共有queueのjob
`1790557852275223391_1542131_i934-evidence-meiji-clip000-r4-20260928` は成功。
本番のball nodeがschema v2を保存し、別のdisk storeからのload-only再開が全cameraで成功した。
保存配列のchecksum・dtype・shape・patch実値と境界maskの検証も読み出し時に通過した。
詳細の正本は[qualification.json](../../runs/run-i934-evidence-meiji-clip000/qualification.json)。

| camera | gate後observed frame | 非観測だが候補を保持したframe | node実行秒 |
|---|---:|---:|---:|
| cam0 | 634 | 682 | 19.146 |
| cam1 | 916 | 400 | 14.804 |
| cam2 | 650 | 666 | 14.481 |

各cameraのnative heatmapは `(1316,72,128)` float32、候補座標は
`(1316,8,2)`、patchは `(1316,8,5,5)`。全frameに8候補が残った。
native heatmapだけでcamera当たり48,513,024 bytesを占める。
score閾値0で抽出するため、単一点の閾値・trajectory gateにより非観測となった
1748 frameの証拠も後段へ渡せる。これは「8候補が正しい」という意味ではない。
元のheatmapと候補の選択窓を併存させるため、候補外の情報を再利用できる。

## 比較と限界

ft-e13の既存checkpointを用いた契約検証であり、学習・GTとの照合・recall・p95評価は
実施していない。観測frame数をrecallと解釈しない。先行の
`run-ball-checkpoint-normalization-meiji-20260923` による入力正規化の修正を前提とするが、
clipや検証目的が異なるため精度の前後比較ではない。全pipelineの3D再構成も実行していない。
学習がないためTensorBoard曲線は対象外。

実行時のconfig、scene manifest、queue再現bundleを同run directoryへ収録した。
大きな配列本体は
`/home/kamimura/projects/tennis-lab/outputs/ball_detection/evaluate/i934-evidence-meiji-clip000-r4-20260928/store/`
に置き、各artifactのpath・SHA-256はqualificationとscene manifestで固定する。
タイミングはnode全体の値で、GPU kernel単体のbenchmarkではない。
dense map保存による容量増加と、他clip・他architectureでの挙動は残課題。

## 次の実験

3 source混合でft-e13から学習し、Meiji video_001 holdoutを同じ入力前処理・復号条件で比較する。
元動画pixelでrecall/p95とcamera・visibility・手首距離による層別値を報告する。
本runを根拠にdeploy checkpointや候補数・patch幅の既定を変更しない。
