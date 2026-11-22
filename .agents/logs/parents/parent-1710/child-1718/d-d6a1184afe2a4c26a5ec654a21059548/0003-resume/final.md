[PR #1786](https://github.com/Motoki0705/tennis-lab/pull/1786) の3件を同じbranchで修正しました。

- HEAD: `02f515301bd75ceb7ffb87ee1373292caabd31cf`
- [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42569603816): **Python・Lint・CPU全10本成功**
- ローカル: **52 passed / 5.52秒**。Ruff・mypy・pre-commit成功。worktreeはcleanです。

| 指摘 | 対応・回帰確認 |
|---|---|
| pr-1786-1 | document/visual共通で不正コメント・宣言を拒否。CSPは実head先頭へ配置し、NBSP境界も拒否。修正前はChromiumで`PWNED=1`となりassert失敗、修正後はexportに到達しません。 |
| pr-1786-2 | `image-set()`等の文字列URLを属性・style要素・stylesheetで拒否。既存の参照所有・変更申告テストへ追加し、修正前の`DID NOT RAISE`を確認しました。 |
| pr-1786-3 | 描画属性を予約し、v2は宣言されたfigureだけを処理。不正対象があっても他graphが動くことをNode・Chromiumで確認。修正前は例外とslider停止でassert失敗しました。 |

回帰は次の入口で実行しています。統合52件の正確なコマンドと修正前ログもPR本文に記録済みです。

```sh
node --test tests/unit/agents_ops/explain_html/visuals.test.mjs
node tests/e2e/agents_ops/explain_html_visuals_browser.cjs
```

v1のbytesは変更せず、v2公開後の旧snapshot・復元・ブラウザ操作を検証しました。`document.py`の既存関数・メソッドのsignatureも不変です。

指摘の解消確認・merge・merge後main CI確認は親へ引き継ぎます。