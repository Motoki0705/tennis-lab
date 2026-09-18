---
id: run-slcs-meiji-v8-feature-reuse-v1
type: run
title: Meiji v7の検証済みRGB特徴をv8へ53clip再利用
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-19'
status: done
config:
  source: slcs/meiji_rgb_v7
  target: slcs/meiji_rgb_v8
  model: dinov3_vitb16
  device: cpu
  publication: immutable_hardlinks
metrics:
  selected_clips: 56
  linked_clips: 53
  linked_camera_features: 159
  skipped_missing_completion_clips: 3
repro:
  commit: 10a8a73edf38348eb6322b901d05329b6b2c7f5d
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: if [ ! -f /home/kamimura/projects/tennis-lab/.training_queue/done/1789739458621952459_956621_slcs-meiji-v8-observe-v1.job
    ]; then echo 'Required observation job did not complete' >&2; exit 1; fi; env
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -B
    knowledge/runs/run-slcs-meiji-v8-feature-reuse-v1/reuse_features.py --output-dir
    outputs/tennis_scene/analyze/meiji_v8_feature_reuse/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v8-feature-reuse-v1
  output_dir: outputs/tennis_scene/analyze/meiji_v8_feature_reuse/s42-001
  receipt: knowledge/runs/run-slcs-meiji-v8-feature-reuse-v1/migration_receipt.json
parents:
- run-slcs-meiji-rgb-features-v2
relations: []
tags:
- slcs
- meiji
- rgb-features
- cache-reuse
---

## 考察 / Findings

### 要約
元動画・manifest・DINOv3設定とSHA・特徴配列を照合し、既存53clip・159cameraのRGB特徴をv8へhardlink公開した。残る3clipは元の完了markerが無く、理由付きでskipした。

### アーキテクチャ詳細
共有queueでobserve終了後にCPU実行した。checkpoint固定SHAは73cec8be…と一致。各媒体とframe/camera identity、特徴inventoryと設定、有限性、既存completion markerを本番readerで検証し、新規temporary directoryから置換禁止のatomic publishを行う。mediaのhardlinkでctimeが変わるため、実行中observeとの同時操作を避けた。

### メトリクスの解釈
linked53、skipped3、selected56。skipはvideo_001/clip_007、video_002/clip_014、video_002/clip_016の完了marker欠落である。不足分は通常生成で作る必要がある。receiptに各cameraの媒体SHAとリンクした特徴SHAを残した。新規学習・forwardを伴わず収束曲線は無い。

### アーキテクチャ⇄メトリクスの因果考察
v8で変更した人物観測・root支持条件はRGB特徴の入力動画やDINOv3設定を変えないため、検証済み特徴を再利用できた。読取中変化や不正な既存targetは失敗させる。旧teacherをコピーする処理は含まれない。

### 既存実験との比較
親runで生成した53clipの特徴を、別版でも同じ入力/設定で参照する処理である。people receiptの3camera不一致は別途修復が必要であり、本runの成功は3D教師の採用を意味しない。

### 次に有効な実験
人物観測のproducer照合修正・再生成とCourt校正の確認後、欠けた3clipの特徴と全体教師を生成し、strict品質検査を通して統合する。
