---
title: ".gitignore の汎用パターン（lib/, runs/, *.json）が新しいディレクトリを黙って無視する"
summary: "汎用のignore規則で新しいlib・runs・JSONが一括stageから抜けることがある。git check-ignoreで確認する。"
applies_to: all
source: "#533 の knowledge/runs/、#1053 の .agents/ops/lib/agent_exec.sh（2026-10-10 に base commit から欠落）"
created: 2026-06-19
last_verified: 2026-11-20
evidence: ".gitignore の `lib/`, `*.json`, `runs/` と、例外の `!knowledge/runs/`, `!.agents/ops/lib/`"
citations:
  - path: ".gitignore"
    symbol: "!.agents/ops/lib/"
  - path: ".gitignore"
    symbol: "!knowledge/runs/"
---

`.gitignore` にはPythonテンプレート由来の汎用パターン（`lib/`, `*.json`, `runs/` など）がある。新しいファイルやディレクトリがこれに一致すると、`git add .` などの一括stageから抜ける。別のworktreeやcloneで初めてファイルの欠落に気づくことがある。

**使い方:** 新しいディレクトリや拡張子を追跡に加えるときは、`git check-ignore -v <path>` で確かめる。無視されていたら、`.gitignore` に `!<path>/` の例外を追加する。コミット後は `git show --stat HEAD` で、意図したファイルが入ったかを確認する。
