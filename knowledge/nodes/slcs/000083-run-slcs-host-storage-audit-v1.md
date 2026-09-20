---
task: slcs
sequence: 83
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-host-storage-audit-v1
type: run
title: WSL保存先NVMeのresetとWindows異常終了を確認
provider: codex
date: '2026-09-19'
status: done
config:
  platform: Windows host of WSL2
  access: read-only standard user
  query_days: 7
  sources:
  - System event log
  - Win32_DiskDrive
  - Get-Partition
  - WSL registry
  - Scsi Port 2 registry
  - Get-StorageReliabilityCounter
metrics:
  stornvme_129_events: 4
  kernel_power_41_events: 1
  whea_matching_events: 0
  detailed_disk_counters_available: 0
artifacts:
  run_dir: knowledge/runs/run-slcs-host-storage-audit-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_host_storage/s42-001
  summary: knowledge/runs/run-slcs-host-storage-audit-v1/summary.json
  collector: knowledge/runs/run-slcs-host-storage-audit-v1/collect_storage.ps1
parents:
- run-slcs-meiji-v9-observation-reuse-v1
relations: []
tags:
- slcs
- integrity
- windows
- storage
- diagnostics
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
---

## 考察 / Findings

### 要約
Windows Systemログを読み取り確認すると、9/18 17:15〜18:46 JSTにstornvme129のRaidPort2 resetが4件、17:54 JSTにbugcheck0x154を伴う異常終了が1件あった。Port2はD:のCT1000P310SSD8に対応し、WSLのVHDもD:に置かれている。checkpoint SHA不一致の原因やSSD故障を確定する結果ではない。

### アーキテクチャ詳細
標準ユーザーでPowerShellの読み取りqueryを個別に実行し、直近7日のWHEA/KernelPower41、storage関連Systemイベント、Win32_DiskDriveのSCSIPort、partitionのdrive letter、WSL BasePath、HKLM Scsi Port2のDriver/Identifierを取得した。詳しい信頼性counterはCIM権限不足で取得できなかった。OS設定の変更・修復・再起動はしていない。raw JSONはoutputs、gitにはkernel pointersやreport IDを除いたsummaryを保存する。

後続で同じ種類の証拠をまとめて収集するため、読み取り専用の `collect_storage.ps1` を用意した。Windows PowerShell AST parseは構文エラー0。collector自体の実診断は未実施であり、本runの個別queryによる記録と区別する。Windows PowerShellで `& '<checkout>\knowledge\runs\run-slcs-host-storage-audit-v1\collect_storage.ps1' -OutputPath '<existing directory>\new-evidence.json' -Days 7 -IncludeReliability` と実行する。既存出力を拒否し、権限不足と該当イベントなしを区別する。自動昇格や設定変更は行わない。

### メトリクスの解釈
NVMe reset4件、Power41/0x154が1件。WHEA専用queryは該当イベントなし（一般的なhardware正常性の証明ではない）。Port2のdriverはstornvme、identifierはCT1000P310SSD8、Win32のdisk1/port2とD: partitionが対応し、Ubuntu BasePathはD:\WSL\Ubuntu。概要HealthStatusはHealthyだが詳細counterは未取得。学習曲線は無い。

### アーキテクチャ⇄メトリクスの因果考察
[Microsoftの0x154説明](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x154--unexpected-store-exception)はkernel memory storeが予期しない例外を検出したことを示す。reset記録とWSL保存先の対応からstorage経路を調べる根拠は得たが、読取byte・memory・driver・device・電源等の原因分離は未完了。温度・SMART相当counter・dump解析なしにSSD故障や交換の必要性を断定しない。

### 既存実験との比較
既存のLinux側ログと限定CPU/GPU/hash対照だけでは原因を特定できなかった。本runはhost側の異常終了・NVMe resetを追加した。ViTPoseのhash不一致と同じ瞬間のresetはこのログに無く、両者の直接の因果関係を示すものではない。

### 次に有効な実験
実処理のhash読取内容を同時保存し、採用ゲートを維持した1回の観測付き実行で切り分ける。詳細storage counterが必要になった場合は、読み取り専用収集scriptをレビュー可能にし、Windowsの管理者権限が必要な操作はユーザーと調整する。
