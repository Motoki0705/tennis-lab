WASB探索は深さ1・深さ2とも `complete` になりました。

[PR #2044](https://github.com/Motoki0705/tennis-lab/pull/2044)  
branch: `codex/citation-explore-cli` / commit: `e881c6f10`

- 保留114行を全解消：87行を同定、27行を理由付き未同定へ。
- 最終結果：既読5本再利用、取得失敗2本、未同定65行、判定不能38行。新規読解・agy呼出0回。
- 探索再開・再実行確認は合計176.586秒。OpenAlex使用量は[公表単価](https://help.openalex.org/access/example-costs/)から概算1,130クレジット。
- 167テスト成功、kg_validateは0 error・0 warning。pre-commit完了（Python差分なしで対象外）。
- [Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43664628289)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43664628231)成功。初回の既存テスト1件の失敗は、同じHEADで再実行して成功しました。

世代別ID・理由別件数・状態遷移をPRに記載し、先頭の `Closes #2022` を保持しました。コード変更なし、validatorは未指定で試行・完了0回です。

Web UI・Papers登録の最終受入とmergeは親側に残ります。