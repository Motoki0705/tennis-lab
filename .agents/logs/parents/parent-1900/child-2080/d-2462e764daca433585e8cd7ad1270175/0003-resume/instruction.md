PR #2087 の統合レビュー（run parent-1900-bbdb70fe3d2f402fa94f6ae85953b8c6、gpt-6-astra / xhigh、HEAD 79b6b797）は request_changes でした。次の major 2件はどちらも妥当なので修正してください。

1. pr-2087-1（decision_logs/publication.py:509 付近、公開要求の再利用・中断後の冪等な再開）
   confirm() は batch を merged として保存してから、各記録の published を順に追記する。依存関係のある記録で、公開担当の要求（publication_request）が先に published になった直後に停止し、同じsession・世代で再試行すると、`_publish` は既存batchの照合より先に `publication_request()` を呼ぶ。旧要求は published なので再利用候補から外れ、別キーの要求が作られる。その結果、元の記録は既にmainにあるのに要求だけの追加PRができ、同じbatchの公開事実が異なる要求IDを参照する。
   → merged batch の未完了bookkeepingを先に照合し、凍結した要求を再利用する順序にする。要求の公開事実だけが保存された途中停止から再試行しても、追加PRや別IDの要求ができないことをテストで確認する。

2. pr-2087-2（decision_logs/publication.py:191 付近、担当変更後はcommit前に中断する）
   担当の確認は commit() の冒頭だけで、その後の plan()（README生成）・Git読取り・worktree作成・ファイル書込みの後、git commit の直前には再確認しない。plan() の実行中に対象へ担当が付いても、旧公開者がcommitし、batchを committed として保存してしまう。
   → 実際の書込み境界（git commit・batchのcommitted保存の直前）で再照合し、中断した状態を後任の正当な公開者が再開できるようにする。この commit 境界での担当変更を検証するテストを追加する。

あわせて、同じ種類の問題（外部・永続の書込み境界の直前で担当を再照合していない箇所、途中停止からの再試行で凍結した値を再利用しない箇所）がほかにないかを見直し、あれば同じ方針で直してください。

検証: 追加・関係テストと tests/e2e/agents_ops の関係分を `-n 4` 以下で実行し、pre-commit を通し、PRへpushして最新HEADのCIを確認してから終了してください。終了前に、自分が起動したプロセスが残っていないことを確認してください。最終報告には、最終HEAD、各指摘の修正内容と追加テスト名、見直しで見つけた同種の箇所、CIの結果を書いてください。
