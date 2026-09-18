---
id: run-slcs-meiji-court-visual-qc-v1
type: run
title: Meiji第2・第3収録のCourt校正を実画像で確認
provider: codex
date: '2026-09-19'
status: done
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
config:
  data:
  - video_001/clip_003
  - video_002/clip_017
  device: cpu
  observation_root: tennis_scene/precompute/meiji_dino_vitpose/s42-004
  explicit_sample_root: tennis_scene/precompute/meiji_dino_vitpose/s42-001
  diagnostic_only: true
  calibration_refits: 0
metrics:
  clips: 2
  views: 6
  reviewed_frame_camera_images: 18
  contact_sheets: 6
  unchanged_input_files: 30
  max_recorded_median_error_delta_px: 0.0
  max_recorded_p95_error_delta_px: 0.0
  video_002_clip_017_cam0_near_baseline_inlier_points: 0
repro:
  commit: 4d24a2144680cace5af825659b8909aecd19e15e
  branch: codex/slcs-real-rgb
  command: bash knowledge/runs/run-slcs-meiji-court-visual-qc-v1/repro.sh /new/unique/output/path
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-court-visual-qc-v1
  output_dir: outputs/tennis_scene/analyze/meiji_court_visual_qc/s42-001
  diagnostics: knowledge/runs/run-slcs-meiji-court-visual-qc-v1/results.json
  visual_review: knowledge/runs/run-slcs-meiji-court-visual-qc-v1/visual_review.json
parents:
- run-slcs-real-court-probe-v3
- run-slcs-meiji-observation-review-v5
relations: []
tags:
- slcs
- meiji
- court
- visual-qc
---

## 考察 / Findings

### 要約
第2・第3収録の校正用2clip・全3cameraについて、保存Courtを18画像・6contact sheetに重ねた。数値再計算は保存記録と一致したが、第3収録cam0の近側baselineに画像上のずれが見える。診断の実行完了であり、Court精度や最終教師の品質合格ではない。

### アーキテクチャ詳細
新しいmodel load・forward・校正fitは行わず、保存homographyによる14点とCourt線、9frameの検出中央値、支持点とRANSAC採用/除外点を表示した。観測root s42-004に生Court sampleが無いため、s42-001を必須の明示入力にした。2clipのcourt.json/court.npz計4組のbyte一致を確認し、両rootと元動画を入力hashに含めた。自動fallbackは無い。表示用cropは推論・校正に使っていない。

### メトリクスの解釈
30入力の前後dual SHAは一致し、6動画は校正receiptのSHAと一致した。各cameraのfit中央値/p95を再計算し、保存値との差は全6viewで0px。6contact sheetを親が確認した所見はvisual_review.jsonに分離し、実行時点でhuman review pendingだったresults.json/run.jsonは変更しない。学習を伴わず収束曲線は無い。

第3収録cam0の支持/inlierは11/10、中央値4.34px・p95 63.62px。近側baselineの点2/3/7は支持不足、点5は除外され、近側baseline上にinlierが無い。p95は除外点を含む検出中央値との残差であり、実白線に対する独立GT誤差ではない。cam1は12/10・3.93/65.12px、cam2は14/14・3.06/6.35px。

### アーキテクチャ⇄メトリクスの因果考察
画像確認では、第3収録cam0の推定baselineが右側ほど実白線よりカメラ側へずれて見えた。近側に採用点が無いため外挿誤差が生じた可能性があるが、これは仮説で、歪みや検出誤差との寄与分離は未実施。中央値が小さいだけでは点が無い領域の正確さを示せない。一方、p95の大きさだけを推定Court全体の不良とも断定しない。cam1にも近側baselineの小さなずれが見える箇所があり、両clipのcam2は選定画像で比較的よく対応した。全6viewとも明白な隣接コートの取り違えは見えなかった。

### 既存実験との比較
Court probe v3は第1収録だけの開発比較だった。本runは第2・第3収録の保存校正を確認する診断で、checkpointや推論recipeを比較していない。人物観測review v5の支持率・人物対応確認を補うが、2D支持率や校正記録の一致を最終3D教師のcoverage・精度へ読み替えない。

### 次に有効な実験
Meiji全体の3D教師生成後、特に第3収録の再投影・速度・支持率とCourt線対応を確認する。近側baselineの独立した2D基準でずれを定量化し、必要なら校正条件を別recipe/観測版で比較する。未解決の線ずれを残したまま、数値ゲート通過だけを根拠にデータセットを完成扱いしない。現在のcacheや採用閾値は本診断で変更していない。
