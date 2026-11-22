---
title: pre-commit の mypy は staged ファイルだけを --follow-imports=skip で検査する
summary: pre-commitのmypyは対象のstagedファイルをfollow-imports=skipで検査する。全体検査と違う型エラーは同じ対象で再現する。
applies_to: all
source: 'PR #613, PR #614, PR #627 (2026-07)'
created: 2026-07-06
last_verified: 2026-11-22
evidence: .pre-commit-config.yamlのmypy entryと、pyproject.tomlのfollow_importsおよびdisallow_untyped_decoratorsのoverrideを照合する。
citations:
- path: .pre-commit-config.yaml
  symbol: mypy --follow-imports=skip
- path: pyproject.toml
  symbol: follow_imports = "normal"
  note: importの追跡を戻す例外
- path: pyproject.toml
  symbol: disallow_untyped_decorators = false
---

pre-commitのmypyは対象のstaged Pythonファイルを `--follow-imports=skip` で渡す。個別のoverrideで追跡するモジュールもある。追跡されないrepo内の依存から来る値は `Any` になり、`no-any-return` や未型付けデコレータの指摘が、全体検査や依存ファイルもstageした場合と異なることがある。

同じファイル集合とhookの引数で再現し、型が分かっている中間値には型注釈を付ける。スクリプトなどの未型付けデコレータと `src.utils.inference_recording` の追跡には個別のmypy overrideがある。
