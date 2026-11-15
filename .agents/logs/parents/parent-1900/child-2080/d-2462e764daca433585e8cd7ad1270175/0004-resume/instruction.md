PR #2087 の再レビュー（run parent-1900-95a278313db14f559bcd6fac1393d645、HEAD 1074059b）は request_changes でした。前回の major 2件の直接原因は解消済みと確認されています。新しい major 1件を修正してください。

pr-2087-1（decision_logs/publication.py:284 付近、中断worktreeの未知の編集を保持して拒否する）
stage後に担当変更で中断したworktreeへ、計画外のファイル（例 recovery-notes.json）が追加された場合、既存の.gitignoreの対象だと `--exclude-standard` で検査から漏れる。内容の検査も計画のfiles内だけなので、後任の再試行は拒否せずcommit・公開へ進む。cleanup() のstatusもignoredファイルを表示せず、`git worktree remove` がそのファイルごと削除する。追加テストの ignored_plan は計画内パスの改変だけを検証していて、このケースを検出しない。
→ ignore対象を含む計画外のファイルも検査し、未知の内容があれば削除せずに保持して拒否する（cleanup も同じ基準で、未知のファイルがあるworktreeを消さない）。ignored の計画外ファイルを追加したケースのテストを加える。同じworktreeの検査・cleanupを使うほかの経路（担当中の親自身の公開の再開など）にも同じ基準が効くかを確認する。

検証: 追加・関係テストを `-n 4` 以下で実行し、pre-commit を通し、PRへpushして最新HEADのCIを確認してから終了してください。終了前に、自分が起動したプロセスが残っていないことを確認してください。最終報告には、最終HEAD、修正内容と追加テスト名、ほかの経路への影響、CIの結果を書いてください。
