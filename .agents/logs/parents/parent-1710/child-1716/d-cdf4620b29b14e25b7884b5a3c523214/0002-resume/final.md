[PR #1783](https://github.com/Motoki0705/tennis-lab/pull/1783) のmajor 2件を修正しました。

- branch: `codex/explain-html-document-contract`
- HEAD: `6c4acc2e27293753f1a08be61ce385510c961cc8`
- [CI attempt 2](https://github.com/Motoki0705/tennis-lab/actions/runs/42553072120/attempts/2): **成功**。`Python`・`Lint`・CPU全10本成功。初回は未変更のGit pagerテストがSIGPIPEで失敗し、失敗ジョブだけ再実行しました。

対応内容です。

- **pr-1783-1**: 改行前処理、pre/textarea/listing直後のLF除去、文字参照・RCDATAを反映しました。手書きDOM値とUTF-16位置による回帰6件が成功し、Chromiumでも6件一致。保存用の生HTMLは保持します。
- **pr-1783-2**: 通常のsrc/href/data/srcset参照もmanifestから所属contentへ対応付けました。未所属参照を拒否し、assetのみ変更した候補の正しい申告受理・未申告拒否を回帰8件で確認しました。

検証コマンド:

```bash
.venv/bin/python -m pytest tests/unit/agents_ops/explain_html tests/integration/agents_ops/explain_html 'tests/e2e/agents_ops/test_agents_docs.py::test_relative_links_resolve[.agents/README.md]' -n 0 --durations=0 --durations-min=0
```

**41 passed／2.29秒**（unit32・integration8・既存link1）。Ruff・mypy・pre-commitも成功しました。修正前コードでは最終回帰14件中13件が失敗します。

公開型・状態名・エラーコードの追加／変更、依存追加はありません。head内の共通resourceにもcontent IDの祖先が必要です。

指摘解消の判断・merge・未実施のmerge後main CI確認を親へ引き継ぎます。