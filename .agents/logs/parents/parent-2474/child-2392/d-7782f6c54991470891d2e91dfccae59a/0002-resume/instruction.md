PR #2477 の正式レビュー（run parent-2474-ec9ab735b0ac4ba38804ef6581ab14c3, round 1）で major が1件出た。採用するので同じbranch・PRで修正する。

指摘「Herdrの異常応答で診断の分類と秘匿が崩れる」:
- `pane` や `agent list` の応答に `tokens: null` があると `AttributeError` が status-line hook の例外処理を抜け、traceback に checkout の絶対パスが出る。
- `pane process-info` の必須項目が欠けた場合も、Herdr の応答異常が `invalid hook input or lifecycle state` に分類される。
- 対象: `.agents/ops/herdr_labels/service.py`（L355付近）、`.agents/ops/herdr_labels/coordination.py`（L53付近）。

求める修正:
- Herdr 応答を利用前に検証して `HerdrError` に変換する（CLIが終了値0で構造の不正なJSONを返す場合を含む）。hook から traceback やローカルパスが出ないこと。ただし想定外の例外を一律に握りつぶす静かなフォールバックにはしない（区分した固定診断で非ゼロ終了する）。
- 上記のケース（`tokens: null`、`process-info` の必須項目欠落、構造不正なJSON）で固定診断が出ることを回帰テストで確認する。
- 検証・pre-commitを通してcommit・pushし、PR番号と完全なHEADを宣言して終了する。
