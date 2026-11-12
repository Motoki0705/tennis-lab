PR 1を作成しました：[PR #1927](https://github.com/Motoki0705/tennis-lab/pull/1927)

- branch: `codex/parent-comment-curation`
- HEAD: `89cca103540cbf5e76485d2dd335dbc11c2e031d`
- 作業treeはクリーンです。mergeは未実施です。

受入条件への対応：

1. **保存基盤**：認証publisher・本文先頭マーカー・専用ラベルを検証します。共有stateのlockと永続journalで再POSTを防ぎ、応答喪失・重複・結び付き不正を明示エラーにします。open/closedとも読取・除外対象です。
2. **作業対象からの除外**：dispatcher、issue-supplement／issue-labeler workflows、ops-report、issues API／親CLI、project_progress、worker_status、herdr_labels、board、ラベル整理CLIを確認・対応しました。weekly-reportは現行入口が撤去済みです。境界の一覧はPR本文に記載しています。
3. **共通reader**：両媒体をcomment ID昇順で読み、出典・実URL・publisher検証結果を保持します。媒体間重複・内容不一致を拒否し、証拠側の通常コメントや別loginは診断として判定を止めます。指定された読取経路を接続しました。
4. **旧形式互換**：writerは親Issueへの投稿を維持しています。旧フィールドを補完せず、親の通常コメントも入力として保持します。旧コメントのみの合意・gate・復旧と、証拠Issue経由の読取を検証しました。

テスト増分は**36ケース**です。関連テスト **1660 passed／763.40秒**、最終修正後 **93 passed／16.51秒**。pre-commitのruff・mypyも成功しました。初回CIのworkflow検査27件の失敗は、step位置への依存を修正しています。

**最新HEADの[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43204483704)は、Python・Lint・CPU全10 shardが成功しました。**

実GitHubでの証拠Issue作成・ラベル同期は制約どおり未実施です。冪等作成は共有state内を対象とし、別ホスト／別stateからの同時作成は対象外です。

writer切替、通知判断、進捗概要、skill／worker指示、6件の再分類、通常入口の全体e2eはPR 2へ持ち越します。設計上の判断待ちはありません。validatorは未指定のため0試行です。