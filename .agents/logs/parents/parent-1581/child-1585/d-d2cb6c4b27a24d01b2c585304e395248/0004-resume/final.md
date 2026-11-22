L4割当が`Service Unavailable`で失敗し、回復は未完了です。再確認も割当0件で、job・queueは起動していません。再投入・代替GPUへの切替は行っていません。

- 運用adapterをcommit・Git/Drive保存済み。[PR #1825](https://github.com/Motoki0705/tennis-lab/pull/1825)、commit `113cd45b`、branch `codex/court-scene-pair-resume-20261109`。
- CPU68件成功（4.19秒）。実Drive代表37ファイルの厳密照合は73.1秒。科学コード・条件は変更なし。
- 対象HEADのCIは最終確認時に実行中。Knowledge libraryは成功。
- 最新候補は29,844更新。新VMでの全5条件検証、full-state復元、33,160更新とvalidation・Drive照合は未実施です。

L4供給回復後の再試行判断を親へ返します。[引継ぎ結果]([local path])[local path]