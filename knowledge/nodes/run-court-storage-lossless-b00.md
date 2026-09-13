---
id: run-court-storage-lossless-b00
type: run
title: B00 float32可逆圧縮と下流互換性
provider: codex
date: '2026-09-13'
status: done
config:
  data: B00 complete Court v3 owner
  storage: float32_byte_planes_v1, byte shuffle + DEFLATE, lossless
  workers: 4
metrics:
  samples: 2176
  original_bytes: 23820450240
  compressed_bytes: 9459054517
  ratio: 0.3970980574127049
  reused_verified_arrays: 6528
artifacts:
  run_dir: knowledge/runs/run-court-storage-lossless-b00
  metrics: knowledge/runs/run-court-storage-lossless-b00/metrics.json
  output_dir: /home/kamimura/projects/tennis-lab/.claude/worktrees/court-storage-ondemand/outputs/storage-ondemand/compact-scenes/B00/datasets/court
parents:
- run-court-storage-audit
relations:
- to: run-court-storage-ondemand-v4
  rel: compares
tags:
- court
- storage
session: 01a0985c-eb87-7310-9657-5411e2818d4e
repro:
  command: .venv/bin/python scripts/experiments/compact_court_storage.py --source
    data/synthetic_data_generation/scenes/B00/datasets/court --destination outputs/storage-ondemand/compact-scenes/B00/datasets/court
    --report outputs/storage-ondemand/compact-b00.json
---

## 考察 / Findings

### 要約
B00全2,176 sampleのRGB/alpha/depthを可逆圧縮し、約60.3%削減した。6,528配列すべての復元byte一致と、既存の完全dataset validatorを確認した。元のsceneは保持し、別の圧縮ownerをpublishした。

### アーキテクチャ詳細
4 byte planeへ転置したlittle-endian float32を、NumPy NPZのDEFLATEで圧縮する。shape・codec version・復元byteのSHA256を保存し、dtype・shape・値域・有限性は元のconsumer/validatorが検証する。RGBをuint8やfloat16へ変更しない。manifestの配列参照だけを明示的な.f32.npzへ変更する。CLI・学習・Review・動画可視化のreaderを対応した。

### メトリクスの解釈
圧縮ownerからSEG/LINEを全splitで生成し、計4,352 PNGが元の教師とbyte一致した。圧縮版DataModuleの3-head train batchも読み出せた（`downstream.json`）。
original_bytes / compressed_bytesはowner全体（PNG、labels、diagnosticsを含む）の実測値。metrics.jsonのwall_secondsは最終resumeの検証・公開のみで、全圧縮時間ではない。途中の実装検証でpublication byte証跡の整合性gateが動作し、途中出力をresumeしている。生成段階の時間・GPU証跡は歴史値として保持し、dense storeのpublished/reference bytesは新ownerの実バイト数へ更新する。

### アーキテクチャ⇄メトリクスの因果考察
数値を量子化せず、同じbyte位置の繰り返しを圧縮する。学習RGBは復元後に従来と同じ8bit丸めを行うため、画素の変更はない。大量の小ファイルはこの変更だけでは減らない。owner全体を単一gzipにせず、sample単位のランダムアクセスを保持した。

### 既存実験との比較
同時期のv4性能実験では、64枚を4-workerで読み込む圧縮経路は未圧縮と同等の速度だった。全datasetの長期学習や冷たいOS page cacheでは未測定。source_target_sha256は変わらず、学習側のteacher provenanceを維持できる。

### 次に有効な実験
production統合後に全sceneの移行・検証を行い、旧ownerを置換する。現時点で既存95 GiBのscene treeからデータを削除していない。容量上限cache向けにuint8 RGBの単一保存とalpha/depthの一時化を別契約で設計する。
