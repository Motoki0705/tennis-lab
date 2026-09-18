---
id: run-slcs-meiji-observation-review-v5
type: run
title: Meiji第3収録の低支持3clipを画像と共通人物軸で確認
provider: codex
date: '2026-09-19'
status: done
config:
  observations: tennis_scene/precompute/meiji_dino_vitpose/s42-004
  clips:
  - video_002/clip_007
  - video_002/clip_008
  - video_002/clip_009
  device: cpu
  diagnostic_only: true
  reference_camera: cam0
  view_half_turns:
  - false
  - false
  - true
metrics:
  clips: 3
  camera_streams: 9
  reviewed_frame_camera_images: 81
  clip_frames: 1244
  min_detection_sample_coverage: 0.891566265060241
  min_pose_supported_fraction: 0.89209726443769
  min_torso_confident_fraction: 0.7942386831275721
  min_canonical_two_view_torso_fraction: 0.9970845481049563
  unsupported_pose_frames: 71
  pre_post_hash_files: 48
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-observation-review-v5
  output_dir: outputs/tennis_scene/analyze/meiji_observation_review/s42-005
  diagnostics: knowledge/runs/run-slcs-meiji-observation-review-v5/results.json
  canonical_support: knowledge/runs/run-slcs-meiji-observation-review-v5/support_checks.json
  visual_review: knowledge/runs/run-slcs-meiji-observation-review-v5/visual_review.json
  provenance: knowledge/runs/run-slcs-meiji-observation-review-v5/analysis_provenance.json
parents:
- run-slcs-meiji-observation-review-v4
- run-slcs-meiji-canonical-association-check-v1
relations: []
tags:
- slcs
- meiji
- observation
- cross-view
- cpu-diagnosis
- visual-review
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: f94f61dd0b16a32867b02de92781d3dfaf7fac3d
  branch: codex/slcs-real-rgb
  command: bash knowledge/runs/run-slcs-meiji-observation-review-v5/repro.sh REPRO_NEW_RUN_ID
---

## 考察 / Findings

### 要約
第3収録の初期完了10clipから、相対box移動が大きいclip_007、遠方の腰肩confidenceが低いclip_008、pose支持に欠損があるclip_009を追加確認した。親が9枚のcontact sheet・81画像を確認し、選定画像では明白な隣接コートへの人物切替えは見えなかった。canonical人物軸で2視点の両肩・両腰がconfidence ≥ 0.3となる割合は、全6個のclip/player系列で99.708–100%。これは最終3D教師の採用率ではない。

### アーキテクチャ詳細
実行commit f94f61ddの既存CPU画像probeを使用し、均等5frameと各人物の最大相対box移動の両端を重複除去して各視点9frameを選んだ。表示はview-local P0/P1である。support_probeは保存されたcourt/peopleへproductionのassociate_peopleを適用し、unsupported confidenceを0にしたraw配列の並べ替えと完全一致を確認した。cam0/1は[0,1]、cam2は[1,0]、全viewのraw track IDsは[0,1]。設定・manifest・動画・観測・画像・source等48ファイルの実行前後dual SHAが一致した。モデル推論・GPU・学習・教師採用閾値の変更は行っていない。

### メトリクスの解釈
最小検出sample率89.1566%、pose支持率89.2097%、腰肩confidence支持率79.4239%。canonical 2視点支持はclip_007のP0=343/343・P1=342/343、clip_008は両者243/243、clip_009は両者658/658。唯一のpose_supported=False区間はclip_009 cam2 local P1 / track1 → canonical P0のframe425–495（両端を含む71frame）で、残る17個のview/player系列にはこのmaskの欠損がない。高confidenceの誤推定、幾何・再投影・速度不適合はこの支持率から除外できない。学習runではないため収束曲線なし。

### アーキテクチャ⇄メトリクスの因果考察
clip_009のcam2 frame492はsupport=0表示で、同時刻のcam0/1には対応選手が見える。数値上もこのclipの共通人物軸では全frameで少なくとも2視点の腰肩支持があるため、単眼欠損だけで全view教師が失われる状態ではない。ただし三角測量・再投影・速度条件を通ることは未検証。cam0の遠方選手は小さく、画像の個々の関節の正確さは確定できない。目視所見と検出confidenceを実測3D精度に読み替えない。

### 既存実験との比較
v1–v4の10clip・267画像に対し、重複しない3clip・81画像を追加し、合計13clip・348画像・3収録となった。先行のcanonical照合10clip・6601frameと合わせ、13clip・7845frameでproductionのmask/並べ替えとの配列一致を確認した。本runは観測のリスク箇所を追加診断したもので、モデル間の改善比較や全56clipの品質保証ではない。

### 次に有効な実験
Meiji全体の観測生成を完了後、同じ固定設定で3D教師・幾何補正・品質判定を実行する。この71frame区間と低confidence区間について、canonical教師の支持・速度・再投影・最終weightを確認する。全体品質確認と1クリップ生成の再実行比較が完了してから、SLCSの単一損失変更を60epochで評価する。
