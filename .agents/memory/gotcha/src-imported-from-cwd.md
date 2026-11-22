---
title: 共有venvのpython -m srcは起動したcheckoutのソースを読む
summary: 共有.venvで使われるsrcは実行時のカレントディレクトリで決まる。試すworktreeから実行・enqueueする。
applies_to: all
source: '#540のasym実験（2026-06-20、2026-06-21に撤回）、#535の再計測、#507、#576'
created: 2026-06-20
last_verified: 2026-11-20
evidence: '#540のdeep16・wideのknowledgeノードと2026-11-20の共有venvでのfind_spec(''src'')の観察。training queueはenqueue時のPWDを記録する。'
citations:
- path: knowledge/nodes/plcs/000038-run-i540-asym-deep16.md
  symbol: 作業ディレクトリ取り違えによる no-op 計測
- path: knowledge/nodes/plcs/000039-run-i540-asym-wide.md
  symbol: 作業ディレクトリ取り違えによる no-op 計測
- path: .agents/skills/training-queue/scripts/training_queue.sh
  symbol: _json_str "$PWD"
- path: .agents/ops/storage_tier/README.md
  symbol: '#worktrees-and-service'
---

共有 `.venv` のPythonでも、通常の `python -m src.…` は起動時のcheckoutから `src` を読む。2026-11-20は共有 `.venv` でrepo外から `src` が見つからず、作業worktreeからは `src.__file__` がそのworktreeを指した。

#540のasym 2 runはメインcheckoutから起動して非対称の設定が適用されず、2026-06-21に撤回された。#535で意図したworktreeから再計測した。直接実行もqueueへの投入も試すworktreeで行い、起動時の `src.__file__` やモデルのパラメータ数で実行版を確かめる。
