---
title: .gitignore の汎用パターン（lib/, runs/, *.json）が新しいディレクトリを黙って無視する
summary: 汎用のignore規則で新しいlib・runs・JSONが一括stageから抜けることがある。git check-ignoreで確認する。
applies_to: all
source: '#533 の knowledge/runs/、#1053 の .agents/ops/lib/agent_exec.sh（2026-10-10 に base commit から欠落）'
created: 2026-06-19
last_verified: 2026-11-22
evidence: .gitignore の `lib/`, `*.json`, `runs/` と、例外の `!knowledge/runs/`, `!.agents/ops/lib/`
citations:
- path: .gitignore
  symbol: '!.agents/ops/lib/'
- path: .gitignore
  symbol: '!knowledge/runs/'
---

`.gitignore` の `lib/`、`runs/`、`*.json` に新規パスが一致すると、一括stageから静かに抜ける。#533の `knowledge/runs/` と2026-10-10の #1053 の `.agents/ops/lib/agent_exec.sh` が該当した。

`git check-ignore -v <path>` で一致した規則を調べる。追跡する対象がファイルならファイルのパターンを、ディレクトリならディレクトリのパターンを解除する。無視されている親ディレクトリがあれば、配下のパスを有効にする前に親も必要な範囲で解除する。コミット後は `git show --stat HEAD` で意図したファイルが入ったか確かめる。
