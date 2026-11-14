2件とも修正し、同じ [PR #2026](https://github.com/Motoki0705/tennis-lab/pull/2026) へpushしました。

- branch: `codex/citation-format-papers`
- commit: `7cf7a019c`
- 外部IDは共通namespaceの矛盾を結合・集約とも拒否。
- 比較探索からCを採録でき、全引用元・ref・役割・判定状態・根拠を記録。範囲外は拒否。
- pytest 233件、kg_validate、ruff/mypy、pre-commit成功。README・PR本文も更新済み。

対象HEADのCIはknowledge・Lint成功、Web UI・全体Pythonは実行中です。指摘の解消判定とmergeは親へ返します。