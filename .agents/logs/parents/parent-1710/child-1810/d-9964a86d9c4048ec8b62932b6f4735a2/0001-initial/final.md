根拠制約・主張の列挙・ValidationResultへの記録を実装しました。[PR #1818](https://github.com/Motoki0705/tennis-lab/pull/1818) は受入未達のdraftです。

- branch: `codex/explain-html-claim-faithfulness`、HEAD: `65d2b9664`
- 関連107テスト、pre-commit、最新HEADのCIは成功。
- 実機は検出確認3回、初版・診断12回で上限到達。旧候補の誤りは検出できました。
- 三角測量の用語、比較の条件不足、観測誤差のCLI timeoutが残り、最終promptでの再生成は未実施です。

[評価表](https://github.com/Motoki0705/tennis-lab/blob/65d2b9664a6e525cb0624169633fc0be30b27542/outputs/explain-html/faithfulness/evaluation.md)に証拠を保存しました。継続には、親による追加実機呼出上限の指定が必要です。