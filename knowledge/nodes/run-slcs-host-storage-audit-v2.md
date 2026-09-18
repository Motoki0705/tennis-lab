---
id: run-slcs-host-storage-audit-v2
type: run
title: WindowsログcollectorがUNCの署名制約で実行前停止
provider: codex
date: '2026-09-19'
status: failed
config:
  access: Windows current user; no elevation
  script_transport: WSL UNC path
  query_days: 1
  include_reliability: false
  current_user_execution_policy: RemoteSigned
  script_sha256: a102fab22e926a04e26f92e7fb47d9837d47382b96d2fbeeeba4a10c054f847c
metrics:
  exit_code: 1
  collector_executed: false
  events_collected: 0
  policy_changes: 0
artifacts:
  run_dir: knowledge/runs/run-slcs-host-storage-audit-v2
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_host_storage/s42-002
  failure: knowledge/runs/run-slcs-host-storage-audit-v2/failure.json
parents:
- run-slcs-host-storage-audit-v1
- run-slcs-meiji-stream-byte-diff-v1
relations: []
tags:
- slcs
- integrity
- windows
- diagnostics
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 0b3ccdb1e77de23a9c6644dc072c61f7661a1c04
  branch: codex/slcs-real-rgb
  command: powershell.exe -NoProfile -NonInteractive -File '\\wsl.localhost\Ubuntu\home\kamimura\projects\tennis-lab\.claude\worktrees\slcs-real-rgb\knowledge\runs\run-slcs-host-storage-audit-v1\collect_storage.ps1'
    -OutputPath '\\wsl.localhost\Ubuntu\home\kamimura\projects\tennis-lab\outputs\tennis_scene\analyze\meiji_host_storage\s42-002\storage.json'
    -Days 1
---

## 考察 / Findings

### 要約
捕捉した1bit差の発生時間とhostログを照合するため、準備済み読み取り専用collectorを1回実行した。PowerShellがWSL UNC上の未署名scriptを拒否し、collector実行前にexit1となった。新しいイベント情報は得ていない。実行ポリシーを変更していない。

### アーキテクチャ詳細
親runのcollect_storage.ps1をWindows PowerShellへ-NoProfile/-NonInteractive/-Fileで渡し、1日分のSystemログとdisk/partition/WSL情報を新規outputsへ保存する予定だった。詳細counterは前回権限不足だったため今回要求していない。WSLのUNCパスから読み込む時点で署名制約により拒否された。

### メトリクスの解釈
UnauthorizedAccess、exit1、collector_executed=false。events_collected=0はログの該当件数が0という意味ではなく、収集そのものが未実行という意味である。学習曲線は無い。

読み取り専用のGet-ExecutionPolicy照会ではCurrentUserがRemoteSigned、MachinePolicy/UserPolicy/Process/LocalMachineはUndefinedだった。設定を変更せず確認した。

### アーキテクチャ⇄メトリクスの因果考察
これはWindows script実行経路の制約であり、checkpointのbyte不一致の原因を示さない。既存の個別queryから得た4件のNVMe resetと1件の異常終了記録は親runに保持しており、今回の失敗で置き換えない。

### 既存実験との比較
親runのcollectorはAST構文検査のみで実診断未実施だった。本runでUNC経由の実行が署名制約にかかることを確認した。エラーを隠す実行ポリシー変更や自動昇格は行っていない。

### 次に有効な実験
確認可能な同じscriptをWindowsローカル一時folderから実行する操作についてユーザーへ確認した。回答までその経路は実行しない。独立したLinux側のbuffered/direct read比較を先に進める。
