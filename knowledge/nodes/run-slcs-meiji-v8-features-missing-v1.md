---
id: run-slcs-meiji-v8-features-missing-v1
type: run
title: Meiji残3clipのRGB特徴生成後、全体監査の1動画SHAが不一致
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-19'
status: failed
config:
  stage: features
  dataset: slcs/meiji_rgb_v8
  clips:
  - video_001/clip_007
  - video_002/clip_014
  - video_002/clip_016
  audit: read_only_all_56_clips
metrics:
  new_feature_clips: 3
  validated_feature_clips: 56
  validated_feature_cameras: 168
  clip_frames: 29148
  camera_token_samples: 8976
  pre_post_hash_files: 686
  matching_input_hashes: 685
  mismatching_input_hashes: 1
  audit_elapsed_seconds: 176.93
repro:
  commit: 41020d13080c5c1765c218924bbbb4d2aee2ef84
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m src.tennis_scene.scripts.build_slcs_dataset stage=features paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
    'clip_ids=[video_001/clip_007,video_002/clip_014,video_002/clip_016]' output_dir=tennis_scene/precompute/meiji_rgb_v8/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v8-features-missing-v1
  output_dir: outputs/tennis_scene/precompute/meiji_rgb_v8/s42-001
  audit: knowledge/runs/run-slcs-meiji-v8-features-missing-v1/audit.json
parents:
- run-slcs-meiji-v8-feature-reuse-v1
relations: []
tags:
- slcs
- meiji
- rgb-features
- integrity-failure
---

## 考察 / Findings

### 要約
残っていた3clipのRGB特徴生成はqueueで完了し、全56clip168cameraのfeature検証は通過した。ただし全入力の終了時照合で1本の動画のSHAが開始時と異なったため、採用監査はfailedとした。特徴生成の正常終了をデータセットの採用完了とは扱わない。

### アーキテクチャ詳細
commit41020d13080c、固定DINOv3 ViT-B/16、従来の256×448/patch16/embed768/frame_stride10で不足3clipを生成した。別CPU監査は原本Meijiとv8を明示パスで読み、全56markerの存在、manifest/media、固定producer SHA、spec、production feature reader、NPZ dtype/shape/有限値/exact frame indices/marker countを検証した。新defaultのv9設定へ依存させない。

### メトリクスの解釈
feature検証は56clip168camera・29148frame・8976camera token sampleを通過。686入力の前後dual SHAは685件一致、1件不一致。対象はv8 video_001/clip_005/media/cam2.mp4で、開始43fa76065af0efb29f9714101fab991edf46c3d5840a892b6a9caf767a75e636、終了7066c5e0f259a141c7f30759c4aa720d0eb0a83d9b865e7761a945c3fa01904d。原本側の同cameraは前後とも43fa…だった。終了後statでは原本/v8は同inode4189208、size12202440、mtime_ns1789017487427768594、ctime_ns1789754129599533438であった。今回の新規3clipとは別の既存clipである。

### アーキテクチャ⇄メトリクスの因果考察
観測事実は同じ実体を指すaliasのhash不一致であり、ファイル書換え・計算・読取・ハードウェアのどれが原因かは未確定。二実装のSHAは同じ読取byte列を処理するため、両方が一致しても前後の期待内容との一致は別途必要である。配列検証通過だけでこの失敗を無視しない。

### 既存実験との比較
親runは53clipを既存特徴から検証再利用した。本runは不足3clipを生成し、全体監査でcheckpoint以外の動画における新しい不一致を記録した。people3cameraの再生成・配列一致監査とは別の結果であり、先行の修復成功を取り消すものでも環境正常性を証明するものでもない。学習曲線は無い。

### 次に有効な実験
該当1動画の両aliasを別snapshotへ各1回取得し、軽量processの二実装SHA・外部sha256sum・snapshot間byte比較で現在内容を確認する。失敗記録は保持し、無断の繰返しやreceipt書換えで合格へ変えない。CPU監査scriptのruff/mypyは成功。該当snapshot対照と新Court実装の同値確認を別々に進める。
