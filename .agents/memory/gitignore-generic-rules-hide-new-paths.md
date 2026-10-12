---
title: ".gitignore の汎用パターン（lib/, runs/, *.json）が新しいディレクトリを黙って無視する"
type: gotcha
applies_to: all
source: "#533 の knowledge/runs/、#1053 の .agents/ops/lib/agent_exec.sh（2026-10-10 に base commit から欠落）"
created: 2026-06-19
last_verified: 2026-10-10
evidence: ".gitignore の `lib/`, `*.json`, `runs/` と、例外の `!knowledge/runs/`, `!.agents/ops/lib/`"
---

`.gitignore` にはPythonテンプレート由来の汎用パターン（`lib/`, `*.json`, `runs/` など）がある。新しく作ったディレクトリ名がこれに一致すると、`git add` しても何も言われずに追跡されない。コミットは成功するため、別のworktreeやcloneで初めてファイルの欠落に気づく。

**使い方:** 新しいディレクトリや拡張子を追跡に加えるときは、`git check-ignore -v <path>` で確かめる。無視されていたら、`.gitignore` に `!<path>/` の例外を追加する。コミット後は `git show --stat HEAD` で、意図したファイルが入ったかを確認する。
