方式 **(a)** で[PR #1883](https://github.com/Motoki0705/tennis-lab/pull/1883)を更新しました。同じ候補版での再承認を維持し、承認snapshotから手動合意・自律側の照合内容を生成します。必須節の不足は明示エラーとし、旧候補specで補完しません。

変更した6ファイル：

- 実装：`.agents/ops/issues/improvement.py`、`.agents/ops/parent/improvements.py`
- 文書：`.agents/ops/issues/README.md`、`.agents/ops/parent/improvements.md`
- テスト：`tests/unit/agents_ops/test_improvement_contract.py`、`tests/e2e/agents_ops/test_parent_improvements.py`

修正前のE2E **2件の失敗を確認**しました。修正後は再現＋契約テスト **36 passed**、関連回帰 **340 passed／399.90秒**（`-n 2`）、pre-commit・ruff・mypy成功です。再現nodeと実測はPR本文に記載しました。

branch：`codex/parent-improvement-loop`  
HEAD：`9f50015df6e489fe80249dc5c2ba581820cacc1b`  
[CI run 43201924778](https://github.com/Motoki0705/tennis-lab/actions/runs/43201924778)：**Python・Lint・CPU全10分割成功**。

mainの新規取込みはありません。PRは競合なし、作業木はcleanです。追加判断はなく、merge／closeは親へ引き継ぎます。