[PR #1800](https://github.com/Motoki0705/tennis-lab/pull/1800) を更新しました。

HEAD: `eaa4007eee293c6b66c25dd22d7a5ac58052353b`  
[CI run 42624706209](https://github.com/Motoki0705/tennis-lab/actions/runs/42624706209) は**初回成功**。Python・Lint・CPU全10本を確認済みです。

| 指摘 | 対応 | 各対象の旧実装での回帰結果 |
|---|---|---|
| pr-1800-1 | undo要求IDを保持して履歴照合を継続。未確定時に再確認を提供し、再送しない | pending消失を検出、12.35秒 |
| pr-1800-2 | 最も内側の可視blockを読位置anchorにする | 入れ子sectionの位置ずれを検出、5.09秒 |
| pr-1800-3 | details自身も状態保存・復元へ含める | open状態の消失を検出、4.50秒 |

```sh
NODE_PATH="$PWD/scripts/ci_metrics/node_modules" .venv/bin/python -m pytest \
  tests/e2e/agents_ops/explain_html/test_browser.py -n 0 --durations=5
```

修正後は5シナリオ成功、40.22秒。関連回帰231件成功、21.69秒。pre-commitも成功しました。[全コマンド・証拠](https://github.com/Motoki0705/tennis-lab/issues/1720#issuecomment-6741422097)を記録済みです。

HTTP入口・状態名・token受渡し・postMessageの許可内容は変更していません。#1721向けの公開契約変更もありません。worktreeはcleanです。修正確認・merge・merge後main確認を親へ返します。