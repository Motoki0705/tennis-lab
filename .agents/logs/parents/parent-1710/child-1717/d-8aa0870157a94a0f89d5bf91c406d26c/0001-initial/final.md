[PR #1787](https://github.com/Motoki0705/tennis-lab/pull/1787) を作成し、レビュー可能にしました。

- branch: `codex/explain-html-gemini-adapter`
- HEAD: `e94d9ad9a617906f2bd11295bd0da2b6ef5898bf`
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42555337560): Python・Lint・CPU全10ジョブ成功
- worktreeはcleanです。

受入条件の検証結果です。

| 項目 | 確認結果 |
|---|---|
| 設定固定 | argv、不一致拒否、長文の完全な受渡しを検証。実機4呼出しすべて指定model/effortの証拠あり |
| 根拠・候補検証 | 数値・出典の保護、許可された訂正、不正Candidate/verdict、意味検査拒否を確認 |
| 中断・回収 | e2e 3件でtimeout/cancel時のプロセス回収、現行版不変、操作tool拒否を確認 |

新規13件は **8.63秒で成功**。既存回帰380件、文書等114件、更新後の依存契約での互換検証13件も成功しました。変更対象pre-commitと追加mypyも通過しています。

実機smokeは、初回 **304.01秒**（執筆271.63／意味検査32.36）、編集 **271.07秒**（執筆225.27／意味検査45.78）で通過しました。実効値は全4呼出しで **Gemini 3.8 Flash (High) / high、agy 1.4.11**。失敗分を含むモデル生成は計8回、reported totalは430,033 tokensです。

#1716の契約変更・変更要望はありません。残りは依存merge後のREADME 3言語版の競合解消・rebase、統合レビュー・mergeです。描画検査、HTTP/UI、実機品質評価全体は対象外として未実施です。