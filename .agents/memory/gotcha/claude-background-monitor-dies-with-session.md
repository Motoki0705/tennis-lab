---
title: Claude Codeのbackgroundの監視は、会話のプロセスが終わると一緒に止まり、再開した会話へ通知は届かない
summary: Claude Codeの会話を再開したら監視プロセスの生存とjobの状態を確かめる。以前の会話が終わると監視も止まった例がある。
applies_to: claude
source: '#1585 のColab本実行の監視（2026-10-31〜2026-11-01）'
created: 2026-11-01
last_verified: 2026-11-01
evidence: 会話を再開すると、前のプロセスで起動したColab監視ループはプロセス一覧に無く、そのtask出力とscratchpadの内容も消えていた。監視の終了通知は届かなかった
citations:
- path: .agents/skills/parent-session/references/workers.md
  symbol: '#colabの長時間ジョブ'
  note: 現行の起動・監視手順
---

#1585では、`run_in_background` で起動した監視ループが元のClaude Code会話プロセスの終了とともに止まった。再開した会話には監視の終了通知が届かず、前のtask出力とscratchpadの一時ファイルも失われた。

会話の再開時は、監視プロセスの生存とjobの状態を個別に確かめる。Colabの長時間実行の起動と監視は [workers reference](../../skills/parent-session/references/workers.md#colabの長時間ジョブ) を参照する。
