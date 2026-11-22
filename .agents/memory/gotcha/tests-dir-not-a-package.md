---
title: 共有venvで import tests は外部の同名パッケージを指すことがある
summary: 共有venvのstandalone Pythonではimport testsがsite-packagesを指す場合がある。pytestと同じimport解決を前提にしない。
applies_to: all
source: 'PR #627（2026-07-10）'
created: 2026-07-10
last_verified: 2026-11-20
evidence: 2026-11-20の共有venvでのfind_spec('tests')の観察。スナップショットではtests/__init__.pyがなく、pytestはimportlib modeを使い、standalone benchmarkはローカルtestsを明示している。
citations:
- path: pyproject.toml
  symbol: --import-mode=importlib
- path: tests/benchmarks/agents_ops/merge_gate_reads.py
  symbol: sys.modules["tests"]
- path: tests/README.md
  symbol: '#fixture-の共有と分離'
---

`tests/` に `__init__.py` がないため、共有venvでstandalone Pythonから `import tests` すると、同名の通常パッケージがローカルのnamespaceより優先される場合がある。2026-11-20はsite-packagesの `tests` が選ばれ、`tests.unit` が `ModuleNotFoundError` になった。`importlib.util.find_spec('tests')` のoriginを確認すると衝突を切り分けられる。

pytestのimportlib modeでの収集と区別する。standaloneの `tests/benchmarks/agents_ops/merge_gate_reads.py` はローカルのtests namespaceを明示して共有helperを読み込む。
