---
title: メインtreeの共有 .venv は pyproject.toml / uv.lock の依存追加に自動で追従しない
summary: 共有.venvの依存はgit-syncでは更新されない。統合レビューの起動前に共有環境でjsonschemaをimportできるか確かめる。
applies_to: all
source: '#1284 / PR #1528（2026-10-30）'
created: 2026-10-30
last_verified: 2026-11-20
evidence: pyproject.tomlはjsonschemaを依存に持ち、agent_output.pyは未導入なら --json-schema requires jsonschema で停止する。git-sync READMEに依存同期の境界がある。
citations:
- path: pyproject.toml
  symbol: jsonschema>=4.26.0
- path: .agents/ops/lib/agent_output.py
  symbol: --json-schema requires jsonschema
- path: .agents/ops/git_sync/README.md
  symbol: '#更新の契約'
---

2026-10-30、メインcheckoutの共有 `.venv` に宣言済みの `jsonschema` がなく、agents_opsの一部テストと `agent_exec --json-schema` を使う統合Codexレビューが起動直後に失敗した。後者はrunを1回消費した。

ImportErrorや `sync project dependencies` が出た場合は、使っている `.venv/bin/python` でimportできるかと `pyproject.toml`・`uv.lock` の記録を照合する。統合レビューの起動前なら `import jsonschema` で確認できる。共有環境はworktree全体に影響するため、同期の判断は [git-syncの更新契約](../../ops/git_sync/README.md#更新の契約) を参照する。
