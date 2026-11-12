[PR #1927](https://github.com/Motoki0705/tennis-lab/pull/1927) に全体を追加し、題名と `Closes #1881` を更新しました。PR1のコミットは保持しています。

- branch: `codex/parent-comment-curation`
- HEAD: `92a95098278199b910714e58a885a90fc4d9152f`
- worktree: clean

完了条件の証拠をPR本文にまとめました。

1. 全機械writerを証拠用Issueへ接続し、共通reader・イベント対応表・進捗概要・skill/worker指示を更新。
2. 全320コメントの意味分類、原文リンク、統合・移動先を掲載。
3. 指摘ゼロレビュー5操作で親投稿0件、完了通知1件、再試行・引継ぎ時の重複抑止を検証。
4. 指定4例の要件変更・待機・人間承認・成果と残件を、通知と進捗から参照できることを検証。
5. 手動・自律入口、旧コメントのみの継続、引継ぎ、権限、レビュー・修正確認・merge gate・公開復旧を検証。

| 親 | 現状→想定通知 |
|---|---:|
| #1581 | 94→18 |
| #1710 | 73→14 |
| #1349 | 49→7 |
| #1694 | 40→5 |
| #1796 | 31→7 |
| #1367 | 21→8 |
| 計 | **308→59** |

#1796の31件は先頭prefixで再構成。追加12件は別枠2通知相当で、全320件では61通知相当です。レビュー186件を一括で削減数には数えていません。

`note` は証拠専用に残し、通知は親AIが種類・キー・根拠を指定する設計です。完了は固定キーを使い、workflow-reportと担当終了は同じ記録を参照します。

検証は **5,042 passed・1 skipped／1,355.65秒**。増分は **55件（PR1: 36、今回: 19）**。mainでも失敗する2nodeと、依存不足の2ファイルは明示除外し、詳細をPRに記載しました。pre-commitのruff/mypyも成功しています。[対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43223165519) は **Pythonを含む13/13ジョブ成功**です。

除外経路はdispatcher、supplement/labeler workflows、ops-report、weekly-report（撤去済み）、project_progress、worker_status、herdr_labels、issues API、board、ラベル整理CLIです。

追加判断待ちはありません。統合レビュー・merge・担当終了は親へ返します。merge後のラベル同期、初回証拠Issue作成、進行中の親での実確認、main CIは未実施で、手順をPRに記載しました。validatorは回数未指定のため0回です。