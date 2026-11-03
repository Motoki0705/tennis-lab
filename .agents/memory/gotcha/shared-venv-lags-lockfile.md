---
title: "メインtreeの共有 .venv は pyproject.toml / uv.lock の依存追加に自動で追従しない"
applies_to: all
source: "#1284 / PR #1528（2026-10-30）"
created: 2026-10-30
last_verified: 2026-10-30
evidence: "`.venv/bin/python -c 'import jsonschema'` と pyproject.toml の dependencies を比べる。agent_output.py は jsonschema が無いと `--json-schema requires jsonschema; sync project dependencies` で止まる"
---

全worktreeが symlink で共有するメインtreeの `.venv` は、PRで依存が追加されても誰かが入れるまで古いまま残る。2026-10-30 には宣言済みの `jsonschema` が無く、agents_ops テストの一部が ImportError になり、統合Codexレビュー（`agent_exec --json-schema`）が起動直後に失敗して run を1回消費した。

**使い方:** ImportError や `sync project dependencies` が出たら、まず宣言済み依存の未導入を疑う。全worktreeが共有する環境を `uv sync` で丸ごと書き換えず、`uv pip install --python .venv/bin/python <pkg>==<uv.lockの版>` で不足分だけ追加する。レビュー run の前に `import jsonschema` を確かめる。
