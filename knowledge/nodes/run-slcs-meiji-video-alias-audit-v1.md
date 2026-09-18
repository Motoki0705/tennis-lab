---
id: run-slcs-meiji-video-alias-audit-v1
type: run
title: Meiji動画aliasの別プロセスsnapshot対照を1回実施
provider: codex
date: '2026-09-19'
status: done
config:
  device: cpu
  model: none
  capture_per_alias: 1
  live_dual_hash_per_alias: 1
  snapshot_dual_hash_per_alias: 1
  external_sha256sum_per_alias: 1
  expected_sha256: 43fa76065af0efb29f9714101fab991edf46c3d5840a892b6a9caf767a75e636
metrics:
  aliases: 2
  snapshots: 2
  matching_expected_digests: 6
  snapshot_different_bytes: 0
  bytes_per_snapshot: 12202440
  elapsed_seconds: 0.3144289999981993
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-video-alias-audit-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_video_alias_audit/s42-001
  receipt: knowledge/runs/run-slcs-meiji-video-alias-audit-v1/audit.json
  previous_failed_audit: knowledge/runs/run-slcs-meiji-v8-features-missing-v1/audit.json
parents:
- run-slcs-meiji-v8-features-missing-v1
relations: []
tags:
- slcs
- meiji
- integrity
- cpu-control
repro:
  commit: 4af1c8d1d0e5ee3d4d02d2166cddff7a1cd78334
  branch: codex/slcs-real-rgb
  command: OMP_NUM_THREADS=2 .venv/bin/python -B knowledge/runs/run-slcs-meiji-video-alias-audit-v1/audit_alias.py
    --original /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_001/clips/clip_005/media/cam2.mp4
    --alias /home/kamimura/projects/tennis-lab/data/slcs/meiji_rgb_v8/videos/video_001/clips/clip_005/media/cam2.mp4
    --output-dir /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_video_alias_audit/s42-001
---

## 考察 / Findings

### 要約
前回のv8 RGB完了監査でvideo_001/clip_005/cam2.mp4のalias側だけ終了時SHAが不一致になったため、別の小さなPythonプロセスで現在のbytesを1回だけ対照した。今回の6 digestはすべて期待値43fa7606…と一致し、2 snapshotもbyte完全一致だった。前回失敗はそのまま保持し、解決済みとは扱わない。

### アーキテクチャ詳細
numpy・torch・feature実装をimportせず、標準ライブラリとdual_sha256だけを使用。原本とv8 aliasから各1回stream copyし、各snapshotと各liveにdual SHAを1回、各liveに別プロセスsha256sumを1回実行した。最後に2 snapshotをstream比較した。再試行は実施していない。snapshotの動画2本はoutputsだけに保持し、git管理のrunにはscriptとJSONだけを置いた。

### メトリクスの解釈
2 alias・2 snapshot、計6 digestが期待SHAと一致。各snapshotは12,202,440 bytes、差分byte数0、最初の差位置はnull。取得前後stat、descriptor stat、script SHAは不変。実行時間0.314429秒。学習・forwardはなく収束曲線は存在しない。

### アーキテクチャ⇄メトリクスの因果考察
観測できたのは別プロセス・snapshotにおける現在bytesの一致だけである。動画書換え、メモリ、ハードウェア、ソフトウェア等の根本原因をこの対照から断定できない。同じ期待値へ戻ったという解釈も、元データが変化した証拠がないため行わない。

### 既存実験との比較
前回RGB監査は特徴56clip/168cameraを通過した一方、686入力の前後dual SHAでv8側1動画だけ43fa7606…から7066c5e0…となりfailedした。原本側は43fa7606…のまま、両pathは同inodeだった。今回の成功は前回の失敗を取り消さず、両結果を併記する。

### 次に有効な実験
親タスクでユーザーの環境条件と未解決の整合性問題を確認する。今回の限定対照を超える再実行や元データの修正は行っていない。kernelログにエラーが無いことだけを一般的な正常性の証拠としない。
