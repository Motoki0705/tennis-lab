---
task: slcs
sequence: 91
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-observation-review-v3
type: run
title: 'Meiji低支持観測の画像確認: 3clip・81画像'
provider: codex
date: '2026-09-19'
status: done
config:
  observations: tennis_scene/precompute/meiji_dino_vitpose/s42-004
  clips:
  - video_000/clip_008
  - video_001/clip_004
  - video_001/clip_008
  device: cpu
  diagnostic_only: true
metrics:
  clips: 3
  camera_streams: 9
  reviewed_frame_camera_images: 81
  min_detection_sample_coverage: 0.7777777777777778
  min_pose_supported_fraction: 0.7741935483870968
  min_torso_confident_fraction: 0.7056451612903226
  max_adjacent_box_center_previous_diagonal: 0.062462229281663895
  video_001_clip_008_raw_index1_two_view_torso_fraction: 0.9144385026737968
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-observation-review-v3
  output_dir: outputs/tennis_scene/analyze/meiji_observation_review/s42-003
  diagnostics: knowledge/runs/run-slcs-meiji-observation-review-v3/results.json
  support_checks: knowledge/runs/run-slcs-meiji-observation-review-v3/support_checks.json
parents:
- run-slcs-meiji-observation-review-v2
relations: []
tags:
- slcs
- meiji
- observation
- cpu-diagnosis
- visual-review
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: ba3a9dbf3ce041b7b9de7008693c04a7293fffaa
  branch: codex/slcs-real-rgb
  command: env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    .venv/bin/python -B knowledge/runs/run-slcs-meiji-observation-review-v1/probe.py
    --project-root /home/kamimura/projects/tennis-lab --observation-root /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004
    --clip video_000/clip_008 --clip video_001/clip_004 --clip video_001/clip_008
    --output-dir outputs/tennis_scene/analyze/meiji_observation_review/REPRO_NEW_RUN_ID
---

## 考察 / Findings

### 要約
完了済み23clipの検出率・腰肩支持率を参照して低支持の3clipを選び、9枚のcontact sheetに含まれる81画像を親が目視確認した。選定画像では対象選手を追跡して見え、明白な隣接コートへの切替えは見られなかった。一方、近側選手の画面外移動と、遠方選手の関節confidence低下を確認した。全frameの人物同一性や3D精度を保証する診断ではない。

### アーキテクチャ詳細
既存CPU probeを変更せず使い、各cameraの均等5frameと各playerの最大相対box移動区間の両端を選定した。動画・人物NPZ・metadataの前後dual SHA照合と記録済み動画SHAとの照合を実施済み。追加の読取確認では同じ9つの人物NPZを元reportのhashと前後照合し、関節5・6・11・12が全てconfidence ≥ 0.3のカメラ数をframeごとに数えた。結果と入力hashを `support_checks.json` に保存した。GPU推論、教師生成、閾値変更は行っていない。

### メトリクスの解釈
最小の検出sample率77.78%、pose_supported率77.42%、腰肩支持率70.56%はいずれもvideo_001/clip_004のcam2・視点内近側local P0。frame61/64/65ではfoot/lower-legだけが見え、pose_supportedはtrueでも肩・腰4関節のconfidenceはいずれも0.3未満で、v8のカメラ参加条件を満たさない。保存した `support_checks.json` は未整列のraw indexを視点間で集計したもので、共通人物軸の支持数としては扱えない。そこにあるvideo_000/clip_008のindex0=479/479、index1=476/479などの数値は当初の診断記録として保持する。

後続の[production人物対応のCPU照合](000088-run-slcs-meiji-canonical-association-check-v1.md)で、cam2のraw indexを反転して共通人物軸へ揃えることを実データで確認した。video_000/clip_008のcanonical P1は447/479frame（93.32%）となり、当初の476/479を訂正する。他の当run対象ではvideo_001/clip_004は両選手248/248、video_001/clip_008はP0=187/187・P1=171/187で数値は変わらなかった。いずれもconfidenceだけの計数で、再投影・速度・幾何条件を含む最終教師利用率ではない。

### アーキテクチャ⇄メトリクスの因果考察
video_000/clip_008とvideo_001/clip_004のcam2ではlocal P0（教師のcanonical P1）が画像左側へ部分的に出ている。footだけのframeの除外は確認できたが、部分的に映る全ての姿勢誤りを排除できる証拠ではない。video_001/clip_004のframe84/85は見切れが残りながら腰肩の閾値を通る。video_001/clip_008の遠方P1ではcam0/cam1の腰肩支持率が78.61%/80.21%で、canonical軸でも2視点支持が不足する16frameが残る。関節confidenceは正しさの確率ではなく、教師生成後の軌道と再投影の確認が必要である。

### 既存実験との比較
v1/v2の5clip・134画像と重複せず、画像確認は計8clip・215画像・2収録となった。低支持の例を選んだため、v2より低い支持率をデータ全体の品質悪化と解釈しない。旧v7教師やv8教師の生成数にも加算しない。学習runではないため収束曲線は対象外。

### 次に有効な実験
56clipのv8生成と厳密な品質集計を継続し、今回の低支持区間が最終的にどのsource code・weight・maskになるか確認する。未確認のvideo_002も画像と対応させる。2視点confidenceだけを根拠にclipを採用したり、採用閾値を緩めたりしない。
