[PR #1793](https://github.com/Motoki0705/tennis-lab/pull/1793)を作成しました。baseは指定どおり`codex/explain-html-gemini-adapter`です。

- branch: `codex/explain-html-service`
- HEAD: `88179353ef7af6eaa2b5caf818b024cae2518578`
- [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42576854148): **Python・Lint・CPU全10ジョブ成功**
- 作業treeはclean、PRはmerge可能です。

受入条件の検証結果です。

| 項目 | 証拠 |
|---|---|
| 重複生成・commit防止、競合拒否 | 実store＋偽editorで同IDの生成・commit各1回、ID異内容・古いbase拒否 |
| cancel・undo・再起動 | Event同期で遅延結果を拒否。進行中journal復旧、commit後の応答保存障害も検証 |
| HTTP境界 | Host・Origin・token・path拒否、private非公開、公開CLIの起動・停止・再起動・export |

新規はjobs 12件＋HTTP/CLI 6件。回帰選択は**181 passed／19.75秒**、最後のHTTP修正後は**6 passed／7.37秒**。pre-commit・追加mypyも成功しました。

HTTP入口は以下です。`D = [local path]`。

- GET: `D`、`D/revisions`、`D/revisions/{revision_id}`、`D/requests/{request_id}`
- POST: `D/requests`（202）、`D/requests/{request_id}[local path]`、`D/undo`、`D/exports`
- 公開GET: `D/revisions/{revision_id}[local path]`、同`content/assets/{asset_name}`、`[local path]`
- 固定UI: `[local path]`、`[local path]`

Windows EdgeからWSLの`localhost`へ接続し、HTTP 200と説明本文のDOMを確認しました。**127.0.0.1 bindのまま到達でき、確認用serviceは停止済み**です。Gemini実API呼出しは0回です。

#1720では、起動fragmentの`document_id/token`を固定UIで取り込み、tokenをメモリに保持して`X-Explain-Token`へ付けてください。状態通知はrequest IDによるポーリングです。進行中は旧版を維持し、`succeeded`で更新、`failed/cancelled/interrupted`では公開errorを表示します。

追加の上流契約変更要望はありません。統合側には、#1786のcompilerと生成bytesの接続、実描画validator、画像要求を`image_generation_unavailable`へ分類するAssetPreparerの注入が必要です。実Gemini＋visual/UI統合と静的版の編集不可表示は未実施です。統合レビュー・merge・merge後main CIは親へ引き継ぎます。