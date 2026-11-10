[PR #1787](https://github.com/Motoki0705/tennis-lab/pull/1787) を修正しました。compile後の最終候補に変更一覧・digest・意味判定を結び付けています。

- branch: `codex/explain-html-gemini-adapter`
- HEAD: `9b4ec527a1df3af7bbaac87aa08dfcc06e08e2f8`
- [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42568507621): **全成功**（Python・Lint・CPU全10本・lane plan）。worktree clean。

公開面の変更は次の2点です。

- `DocumentCompiler = Callable[[Document], Document]` を追加。
- `GeminiEditor(gemini, *, compile_document: DocumentCompiler)` を必須化。図なしでも恒等変換を明示します。

回帰確認の入口:

```sh
.venv/bin/python -m pytest tests/integration/agents_ops/explain_html/test_editor.py::test_long_local_edit_delivers_complete_context_to_both_calls -n 0
```

修正前はcompile拒否を無視して `DID NOT RAISE`、修正後は **1 passed / 1.36秒**。compiler指定を無視する故障注入でも検出を確認しました。関連回帰は **170 passed / 16.65秒**、pre-commit・Ruff・mypyも成功。全コマンドと実測はPR本文に記録済みです。

#1719は新constructorへcompilerを渡し、返却後の候補を再展開せずstageしてください。#1716への契約変更要望はありません。#1786の実compilerとの統合smokeは未確認で、#1719/#1721側に残ります。merge・merge後main CI確認は親へ引き継ぎます。