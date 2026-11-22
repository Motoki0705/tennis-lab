[PR #1787](https://github.com/Motoki0705/tennis-lab/pull/1787) をmain `9ad3e934c` へrebaseし、force-pushしました。

- HEAD: `cba931115233a9b93007f35229439e67e45d64f5`
- [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42581903131): **全成功**（Python・Lint・CPU全10本・lane plan）
- PRはMERGEABLE/CLEAN、worktreeもcleanです。

競合は `explain_html/README.md`・`README.ja.md`・`README.zh-CN.md` の3ファイルのみ。導入文・module表・検証節に双方の内容を統合しました。`range-diff` の既存4コミットの差分はREADME関連だけで、承認済みのeditor/runtime/promptは同一です。追加commitは統合テスト1件とREADME更新です。

追加テスト:

```sh
.venv/bin/python -m pytest tests/integration/agents_ops/explain_html/test_editor.py::test_visual_compiler_binds_verification_and_commit_to_the_final_document -n 0
```

実 `render_document`・`validate_visuals`・storeを接続し、最終候補・意味判定・digest・commitの一致を確認しました。**1 passed / 1.12秒**、関連回帰は **181 passed / 14.47秒**。compilerを恒等変換へ差し替える故障注入では `validation_mismatch` を検出。pre-commit・追加mypyも成功しています。

`StrictHTMLParser` の拒否と数値保持も確認済みです。`render_document` はDocumentCompilerへ直接渡せるためwrapperは不要で、公開API変更もありません。実API呼出しは0回です。実機モデル・ブラウザを含むサービス統合と、merge後main CI確認は親側に残ります。