---
title: "tests/ はPythonパッケージではないので、テストモジュール同士で import できない"
type: gotcha
applies_to: all
source: "PR #627（2026-07-10）"
created: 2026-07-10
last_verified: 2026-10-10
evidence: "tests/__init__.py が存在しない。共有の fixture は tests/conftest.py にある"
---

`tests/` には `__init__.py` がない。あるテストファイルから別のテストファイルの helper を import すると失敗する。

**使い方:** テスト間で共有する helper は、`tests/conftest.py`（またはサブディレクトリの `conftest.py`）の fixture にする。
