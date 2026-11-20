---
title: "Claude Codeのbackgroundの監視は、会話のプロセスが終わると一緒に止まり、再開した会話へ通知は届かない"
summary: "Claude Codeの会話を再開したら監視プロセスの生存とjobの状態を確かめる。以前の会話が終わると監視も止まった例がある。"
applies_to: claude
source: "#1585 のColab本実行の監視（2026-10-31〜11-01）"
created: 2026-11-01
last_verified: 2026-11-01
evidence: "会話を再開すると、前のプロセスで起動したColab監視ループはプロセス一覧に無く、そのtask出力とscratchpadの内容も消えていた。監視の終了通知は届かなかった"
citations:
  - path: ".agents/skills/parent-session/references/workers.md"
    symbol: "colabの長時間ジョブ"
    note: "監視引き継ぎの手順"
---

`run_in_background` で起動した監視は、その会話のClaude Codeプロセスに属していた。#1585 では会話プロセスの終了とともに監視も止まり、再開後の会話に監視やjobの終了通知は届かなかった。scratchpadの一時ファイルも失われた。

**使い方:** 会話を再開したら、監視プロセスの生存とjobの実状態を確認する。長時間のColab jobは、監視を引き継げるように [起動と完了の受け渡し](../../skills/parent-session/references/workers.md#colabの長時間ジョブ) に従ってjob識別子と状態を記録する。
