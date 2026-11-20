---
title: "共有venvで import tests は外部の同名パッケージを指すことがある"
summary: "共有venvではimport testsがsite-packagesを指す場合がある。共有fixtureはconftest.py、共通の検証部品はtests/supportに置く。"
applies_to: all
source: "PR #627（2026-07-10）"
created: 2026-07-10
last_verified: 2026-11-20
evidence: "共有venvから importlib.util.find_spec('tests') の origin を確認し、git ls-files tests/__init__.py と tests README を参照する。"
citations:
  - path: "tests/README.md"
    symbol: "fixture-の共有と分離"
---

`tests/` には `__init__.py` がない。2026-11-20の共有venvでは `import tests` がsite-packagesの同名パッケージを指し、`tests.unit` で `ModuleNotFoundError` になった。

**使い方:** 共有する入力は該当範囲の `conftest.py` のfixtureにし、共通の検証部品は `tests/support/` に置く。利用方法は [tests README](../../../tests/README.md#fixture-の共有と分離) を参照する。
