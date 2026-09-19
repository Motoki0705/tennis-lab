---
id: run-slcs-meiji-one-clip-repro-takeover-v1
type: run
title: 'Meiji 1コマンド生成: cold/warm/cold完走、bit-exact比較は微小人物差で不成立'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: failed
config:
  recipe: scripts/datasets/build_real_rgb.sh --execute meiji stage=all
  clip: video_000/clip_000
  cold_comparison: exact dtype/shape/values, equal NaNs, zero tolerance
  warm_comparison: exact dataset/observation file bytes
metrics:
  successful_generation_invocations: 3
  frames: 1010
  cameras: 3
  warm_changed_dataset_files: 0
  warm_changed_observation_files: 0
  cold_dataset_files_compared: 13
  cold_observation_files_compared: 27
  cold_generation_files_compared: 14
  cold_ball_position_max_abs_difference_m: 0.0
  cold_refined_player_position_max_abs_difference_m: 0.00012677907943725586
  cold_raw_player_position_max_abs_difference_m: 0.00019931793212890625
  cold_player_yaw_max_abs_difference_rad: 0.000029802322387695312
  cold_pose_reprojection_max_abs_difference_px: 0.31922563764010903
  cold_player_speed_max_abs_difference_mps: 0.006157875061035156
  comparison_exit_code: 1
repro:
  commit: 57726f2c8f38fe50ab3870a7a030fad78ab14b4b
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python knowledge/runs/run-slcs-meiji-one-clip-repro-v1/probe.py
    --output-dir /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_one_clip_repro/s42-takeover-001
    --run-id takeover001
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-one-clip-repro-takeover-v1
  output_dir: outputs/tennis_scene/analyze/meiji_one_clip_repro/s42-takeover-001
  log: knowledge/runs/run-slcs-meiji-one-clip-repro-takeover-v1/queue.log
parents: [run-slcs-meiji-v9-full-qc-v2, run-slcs-meiji-v9-repair-start-failure-v1]
tags: [slcs, meiji, real-rgb, reproducibility, cold-warm, limitation]
---

## 考察 / Findings

### 要約

共有GPU queueで初回生成A、同じ出力へのwarm A、独立出力へのcold Bが全て終了コード0で完走した。
両coldは必須成果物・production readerによる1010frame/3camera教師・DINO特徴の検証を通過した。
warm後のdataset/observationは全byte不変。一方、cold間は人物系の微小な数値差と実行警告履歴差があり、
事前宣言した許容誤差0の比較は失敗した。このnodeのfailedは生成失敗ではなく厳密比較の失敗を表す。

### アーキテクチャ詳細

同じ1コマンドwrapperで観測、PLCS/BLCS推論、幾何補正、RGB特徴を順に実行した。
PLCS/BLCSは採用済み学習checkpointから復元した標準pathを使い、pinを変更していない。
モデル読込と公開前guardを通す通常recipeであり、手作業による中間配列差し替えは行っていない。
queue取得時commitとcommandはrepro bundleに保存した。実行中の追加commitはknowledgeだけで、
開始commitと記録時HEAD7621307bの`git diff --name-only ... -- src scripts`は空だった。
probeの最終source-hash guardにはcold比較失敗のため到達していないことも区別する。

### メトリクスの解釈

`cold_comparison.json`は全file・全NPZ配列のdtype/shape/有限性パターンと最大差を保存している。
ball 3D、outsource UV/visibility、Court、3cameraのDINOv3特徴、媒体は完全一致した。
公開refined player positionの最大成分差は0.000126779m（約0.127mm）、yawは約2.98e-5rad。
人物の差はcam2 detection boxes/scores、pose座標/confidenceから後続配列に認められ、
cam2 pose arrayの最大差は0.287964（座標pxとconfidence混在配列のため一律単位とは呼ばない）。
raw/refined sceneの全非有限値パターンは一致し、source detection IDs、track IDs、支持maskも一致した。
品質配列のplayer速度差は最大0.006158m/s、pose再投影差は最大0.319226px。
metadataの差は実際に異なるcam2_people.npzのdigestで、同じbyteと偽装していない。

generation Aだけに存在する`checkpoint_warnings.jsonl`はwarm時のlegacy receipt扱いを記録した6行。
今回の全行でobserved/expected digestは一致しており、checkpoint SHA不一致の新発生ではない。
実行履歴差を比較から後付けで除外せず、元の失敗判定・policyを保持した。学習曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

人物観測の微小差が後続player教師と品質値の差へ伝わることとは整合するが、根本原因は未確定。
単独clipの結果から全環境のbit-exact再現性を保証しない。球・特徴の一致とwarmのbyte安定性は
確認できた範囲の正の証拠であり、cold完全一致の失敗を成功へ読み替えるものではない。

### 既存実験との比較

全56clipの品質監査は構造・teacher/producer一致・有限性・品質の検証であり、独立生成間のbit-exact性とは別。
本runでもPLCS/BLCSのstrict pinを通過しており、9clip修復以前の記録不一致を再現したものではない。
ユーザーが軽微な再現性不安では止めず進めることを許可しているため、この規模の差を記録した上で
固定公開版real_rgb_v1の学習を継続する。許容閾値を後から作って本runを合格にしない。

### 次に有効な実験

予約解放後、予定していたball平滑化weight=0のpilot 60epochが自動で開始した。
validation選定重み・定数baseline・motionを比較し、全体版60epochとheld-out入力条件評価へ進む。
この微小差を理由とする再生成の反復やハードウェア原因の調査は行わない。
