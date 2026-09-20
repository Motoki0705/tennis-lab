---
task: slcs
sequence: 122
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-v9-test-teacher-review-v1
type: run
title: 'Meiji第3収録の教師QC: RGB重畳とunsupported区間の確認'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/meiji_rgb_v9
  clips: [video_002/clip_000, video_002/clip_003]
  device: cpu
metrics: {reviewed_clips: 2, selected_frames: 20, rgb_panels: 60, positive_ball_frames: 908, positive_player_frames: 2066}
repro:
  commit: 48243023
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m scripts.analysis.render_reconstruction_review
    --dataset-root /home/kamimura/projects/tennis-lab/data/slcs/meiji_rgb_v9
    --run-root /home/kamimura/projects/tennis-lab/outputs/tennis_scene/generate/meiji_rgb_v9/s42-002
    --output-root /home/kamimura/projects/tennis-lab/outputs
    --output tennis_scene/visualize/meiji_teacher_review/s42-test-recording-001
    --clip video_002/clip_000 --clip video_002/clip_003
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-test-teacher-review-v1
  output_dir: outputs/tennis_scene/visualize/meiji_teacher_review/s42-test-recording-001
parents: [run-slcs-meiji-v9-teacher-review-v1]
relations: []
tags: [slcs, meiji, quality, visualization, cpu]
---

## 考察 / Findings

### 要約

第3収録の2clipでraw/refinedのRGB重畳と軌跡図を確認した。これで3収録すべてに確認例がある。
選択frameの人物rootと腰観測の位置関係は概ね一致するが、unsupported端点を含む速度spikeは残る。
これは教師生成のQCであり、SLCSモデルのtest成績やモデル選定ではない。

### アーキテクチャ詳細

前回と同じframe選択・3視点表示・共通weight・実FPS微分を使用。raw/refinedの由来と2D配列を
表示前に照合した。モデル推論や学習、教師の閾値変更は行っていない。

### メトリクスの解釈

2clip計1,058frame中ball正weightは908、player-slotは2,066/2,116。
20時点60画像の確認であり、未表示frameや第3収録全clipの品質保証ではない。
clip000の終盤には支持の弱い遠方playerの飛びがあり、0 weightが明示されている。
速度図は全端点を表示するため、学習対象の連続支持点だけの最大速度とは区別する。
新規学習がなく収束曲線は対象外。独立実測3D精度は測っていない。

### アーキテクチャ⇄メトリクスの因果考察

補正後rootはrawのずれを軽減している一方、補正可能区間と低支持区間の境界で大きな微分が生じうる。
0 weightを隠さない表示と、同じmaskを使った数値監査を両方維持する必要がある。

### 既存実験との比較

前回のtrain/val収録3clipに、未見収録の教師2clipを追加した。選択frameで明らかなslot入替えは
見られないが、yaw・絶対奥行き・全体教師採用の判断を目視確認だけで行わない。

### 次に有効な実験

生成完了後の全体監査で支持率・欠測長・同mask速度・由来を検証し、その後に固定splitでSLCSを学習する。
