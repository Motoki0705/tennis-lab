---
title: "pre-commit の mypy は staged ファイルだけを --follow-imports=skip で検査する"
summary: "pre-commitのmypyは対象のstagedファイルをfollow-imports=skipで検査する。全体検査と違う型エラーは同じ対象で再現する。"
applies_to: all
source: "PR #613, PR #614, PR #627 (2026-07)"
created: 2026-07-06
last_verified: 2026-11-20
evidence: ".pre-commit-config.yaml の mypy hook の entry、pyproject.toml の disallow_untyped_decorators override"
citations:
  - path: ".pre-commit-config.yaml"
    symbol: "mypy --follow-imports=skip"
  - path: "pyproject.toml"
    symbol: "disallow_untyped_decorators = false"
---

pre-commit の mypy hook は、対象ディレクトリの staged Python ファイルに `mypy --follow-imports=skip` を実行する。明示対象外のrepo内モジュールからの値が `Any` 扱いになるため、全体検査と結果が変わる。

- `@hydra_main`（`src/utils/hydra.py`）などのデコレータが untyped decorator 扱いになる。scripts 系モジュールには `pyproject.toml` で `disallow_untyped_decorators = false` の override がある。
- cv2 / PIL / 一緒にstageしていない `src.utils.*` からの戻り値は `Any` になり、`no-any-return` が出る。
- 同じファイルでも、一緒にstageしたファイルによって結果が変わる。

**使い方:** hookの検査対象と同じファイルを指定して再現する。必要な結果の型が分かっている場合は、中間値に型注釈（`x: np.ndarray = ...`）を付ける。
