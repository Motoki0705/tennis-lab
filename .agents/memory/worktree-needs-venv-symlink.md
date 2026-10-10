---
title: "新しいworktreeには .venv がないので、メインtreeの .venv へのsymlinkを張る"
type: workflow
applies_to: all
source: "PR #570（2026-06-25）"
created: 2026-06-25
last_verified: 2026-10-10
evidence: "`.venv` はメインtreeの .git/info/exclude で無視されている。scripts/run_in_repo_venv.sh は git common dir からメインtreeの .venv を解決する"
---

`.venv` はgitで追跡されないので、`.claude/worktrees/<name>` に作ったworktreeには存在しない。AGENTS.md の `.venv/bin/python` がそのままでは使えない。

pre-commit の ruff / mypy は `scripts/run_in_repo_venv.sh` を通してメインtreeの `.venv` を使うため、symlinkがなくても動く。

**使い方:** worktreeを作ったら、worktreeのルートで `ln -s ../../../.venv .venv` を実行する。このsymlinkは exclude 済みなのでコミットされない。`data/` などgitignoreされた大きなディレクトリはsymlinkせず、絶対パスで参照する。
