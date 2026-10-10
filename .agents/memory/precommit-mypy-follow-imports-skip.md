---
title: "pre-commit の mypy は staged ファイルだけを --follow-imports=skip で検査する"
type: gotcha
applies_to: all
source: "PR #613, PR #614, PR #627 (2026-07)"
created: 2026-07-06
last_verified: 2026-10-10
evidence: ".pre-commit-config.yaml の mypy hook の entry、pyproject.toml の disallow_untyped_decorators override"
---

pre-commit の mypy hook は `mypy --follow-imports=skip` を staged ファイルだけに実行する。同じコミットに含まれないモジュールは `Any` 扱いになるため、全体の `mypy` では出ないエラーが出る。

- `@hydra_main`（`src/utils/hydra.py`）などのデコレータが untyped decorator 扱いになる。scripts 系モジュールには `pyproject.toml` で `disallow_untyped_decorators = false` の override がある。
- cv2 / PIL / 一緒にstageしていない `src.utils.*` からの戻り値は `Any` になり、`no-any-return` が出る。
- 同じファイルでも、一緒にstageしたファイルによって結果が変わる。

**使い方:** 中間値に型注釈（`x: np.ndarray = ...`）を付けるか `cast("Type", value)` を使う。モジュール全体の作法に関わる規則は、`[[tool.mypy.overrides]]` をスコープを絞って追加する。hook を通すかどうかが本当の関門で、全体の `mypy` には既存のエラーが残っている。
