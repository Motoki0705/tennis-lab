---
id: run-i618-b00-fit-court-detections-v1
type: run
title: B00 fit-only court detection v1
issue: 618
provider: codex
date: '2026-07-25'
status: failed
config:
  model: run-i621-court-kp512-resume-r4-epoch21
  checkpoint_sha256: 501e651599389a8a6eda05c4bb6d2e265c23d276a76ced90692bb2f5c8e4d77f
  loss: inference-only
  data: b00-default-v1 fit groups
  image_size:
  - 959
  - 539
  detector_short_side: 512
  min_peak_score: 0.3
  min_confident_keypoints: 12
  min_homography_inliers: 12
  holdout_group_ids:
  - 2
  - 6
  - 10
  - 14
metrics:
  fit_camera_count: 363
  accepted_count: 0
  acceptance_rate: 0.0
  insufficient_detector_confidence_count: 335
  insufficient_homography_inliers_count: 363
  max_confident_keypoints: 13
  max_homography_inliers: 11
repro:
  commit: ac9e640903a6dfaecb65fc980f5dcf408bbcd589
  branch: main
  command: .venv/bin/python -m src.tennis_scene.scripts.infer_fit_view_courts
  provider_bundle_fingerprint: 5438ced3d08307cc0357b1509b8ae58c6fa834e023042bb5f7e7b0bf6d76066b
  artifact_fingerprint: deb24bf8db964f7b1b177c3afd9f1a26df897037f9f877daffdca94e66fb03c7
artifacts:
  output_dir: data/tennis/3dgs_alignment/b00-default-v1
  detections: data/tennis/3dgs_alignment/b00-default-v1/b00-fit-court-detections-v1-deb24bf8db964f7b.json
  report: .codex-loop/C03_FIT_COURT_DETECTIONS.md
parents:
- run-i621-court-kp512-resume-r4
relations: []
tags:
- court_detection
- 3dgs-blcs
- alignment
- fit-only
---

## 考察 / Findings

### 要約

B00 provider の fit group 363 view だけへ配備済み KP14 detector を実行したが、
凍結 gate の受理は **0/363** だった。holdout group `{2,6,10,14}` の画像は
推論せず、inventory だけを artifact に記録した。

### アーキテクチャ詳細

親 run の checkpoint を byte hash で固定し、provider の exact 959x539 sRGB
PNG を Pillow RGB decode、short-side 512 / 8-pixel align（実入力
904x512）、ImageNet normalization、subpixel peak refine で処理した。出力は
元 provider pixel へ戻し、confidence、normalized RANSAC homography、
court occupancy/visibility、line-edge evidence の固定 gate を適用した。
renderer/gsplat module は import していない。

### メトリクスの解釈

全 363 view が 12-homography-inlier gate を失敗し、最大でも 11 inlier。
335 view は score 0.30 以上が 12 keypoint 未満でもあり、confident count の
quartile は 6/8/10、最大 13 だった。raw score、keypoint、全 rejection 理由は
immutable JSON に残した。

### アーキテクチャ⇄メトリクスの因果考察

fit-only の事後診断では min inlier を 10 へ下げても他の frozen geometry/line
gate を通るのは frame 230 の 1 view だけだった。このため単一閾値が僅かに厳しい
だけではない。B00 は低い handheld camera から court が部分表示される view が
多く、per-channel global argmax が画面端や隣接 court の line intersection を混ぜ、
full-court single-view homography を形成しない、という仮説が観測と整合する。

### 既存実験との比較

親 run は同一 detector が validation 再評価で mean distance 1.708886 px を示した
一方、B00 では全 view が 12-inlier 未達だった。これは checkpoint/resize drift
ではなく、broadcast-like annotated validation と B00 partial/multi-court view の
domain/visibility 差を示す。holdout を見て選んだ結論ではない。

### 次に有効な実験

raw heatmap の複数局所 peak と confidence を保持し、fit view の部分的 landmark
集合から camera-fixed reprojectionで per-view court/symmetry hypotheses を生成する。
group を跨ぐ support で court instance を cluster し、単一-view full-court gate を
alignment の前提にしない。policy と acceptance threshold を fit data だけで固定後、
初めて quarantined holdout を一回評価する。
